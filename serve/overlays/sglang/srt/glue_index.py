"""Optional indexer Q prep + K compression fusion. Merge binding into qwenfast.so."""
import os
import torch


def install_index_prep(extension):
    if os.environ.get('QWENFAST','1')!='1' or os.environ.get('QWENFAST_ATTN_INDEX_PREP','0')!='1':return
    if not (hasattr(extension,'attn_index_prep_parallel') or hasattr(extension,'attn_index_prep')):return
    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
    if getattr(QSAIndexer,'_wp_attn_prep_installed',False):return
    orig=QSAIndexer._forward_impl
    has_parallel=hasattr(extension,"attn_index_prep_parallel")
    kernel=getattr(extension,"attn_index_prep_parallel",None)
    if kernel is None:kernel=extension.attn_index_prep

    def forward(self,hidden_states,positions,forward_batch,indexer_metadata):
        md=indexer_metadata;mode=forward_batch.forward_mode
        modes=mode.is_decode() or getattr(mode,'is_target_verify',lambda:False)() or getattr(mode,'is_draft_extend_v2',lambda:False)()
        pool=md.token_to_kv_pool
        slots=md.pending_ring_slots
        groups=getattr(md,'graph_ring_group_locs',None)
        writes=getattr(md,'graph_write_locs',None)
        n=md.get_token_to_batch_idx().numel()
        if not (modes and md.is_cuda_graph and 1<=n<=8 and (has_parallel or n<=2) and hidden_states.shape[0]>=n
                and self.index_n_heads==4 and self.index_head_dim==128 and self.compress_ratio==4
                and hidden_states.is_cuda and hidden_states.dtype==torch.bfloat16
                and self.rotary_emb.rotary_dim==64 and self.rotary_emb.is_neox_style
                and self.rotary_emb.cos_sin_cache.is_contiguous()
                and self._use_fused_compress(pool) and slots is not None
                and slots.is_cuda and slots.device==hidden_states.device
                and slots.dtype==torch.int64 and slots.numel()>=n and slots.is_contiguous()
                and groups is not None and groups.is_cuda and groups.device==hidden_states.device
                and groups.dtype==torch.int32 and groups.ndim==2 and groups.shape[1]==4 and groups.is_contiguous()
                and writes is not None and writes.is_cuda and writes.device==hidden_states.device
                and writes.dtype==torch.int32 and writes.is_contiguous()
                and positions.is_cuda and positions.device==hidden_states.device
                and positions.dtype==torch.int64 and positions.ndim in (1,2)
                and positions.shape[-1]>=n and (positions.ndim==1 or positions.shape[0]==3)
                and positions.stride(-1)==1
                and pool.index_state_dtype==torch.bfloat16
                and not torch.compiler.is_compiling()):
            return orig(self,hidden_states,positions,forward_batch,md)
        state=pool.get_qsa_key_state_buffer(self.layer_id)
        compressed=pool.get_qsa_compressed_k_buffer(self.layer_id)
        if not (state.is_contiguous() and compressed.is_contiguous()
                and state.dtype==torch.bfloat16 and compressed.dtype==torch.bfloat16):
            return orig(self,hidden_states,positions,forward_batch,md)
        logical=md.decode_logical_positions
        if logical is None:logical=md.get_seqlens_expanded()-1
        logical=logical[:n]
        pos=positions[:,:n] if positions.ndim==2 else positions[:n]
        qk,_=self.index_qk_proj(hidden_states[:n])
        q=kernel(qk,self.q_layernorm.weight.data,self.k_layernorm.weight.data,
             self.rotary_emb.cos_sin_cache,self._rope_axis_map(qk.device),pos,slots[:n],
             state.view(-1,128),pool.qsa_rope_position_buffer,groups,writes,
             compressed.view(-1,128),64,self.q_layernorm.variance_epsilon,self.k_layernorm.variance_epsilon)
        cache,table,lengths,maxlen=md.get_decode_mqa_inputs(self.layer_id)
        return self.select_decode_tokens(q,cache,table,lengths,maxlen,logical,md.get_seqlens_int32())

    QSAIndexer._forward_impl=forward
    QSAIndexer._wp_attn_prep_installed=True


def install_index_mqa(extension):
    if (os.environ.get('QWENFAST','1')!='1' or os.environ.get('QWENFAST_ATTN_MQA','0')!='1'
            or not hasattr(extension,'attn_index_mqa')):return
    from sglang.srt.layers.attention.qsa import qsa_indexer as qi
    if getattr(qi,'_wp_attn_mqa_installed',False):return
    original=qi.qsa_mqa_decode

    def decode(q,cache,table,lens,max_model_len,score_scale=None):
        if not (q.is_cuda and q.dtype==torch.bfloat16 and q.is_contiguous() and q.ndim==3
                and 1<=q.shape[0]<=8 and q.shape[1:]==(8,128)
                and cache.is_cuda and cache.device==q.device
                and cache.dtype==torch.bfloat16 and cache.is_contiguous() and cache.ndim==4
                and cache.shape[1:]==(16,1,128) and table.is_cuda and table.device==q.device
                and table.dtype==torch.int32 and table.stride(1)==1
                and lens.is_cuda and lens.device==q.device and lens.dtype==torch.int32 and lens.is_contiguous()
                and 1<=max_model_len<=table.shape[1]*16 and not torch.compiler.is_compiling()):
            return original(q,cache,table,lens,max_model_len,score_scale)
        return extension.attn_index_mqa(q,cache,table,lens,max_model_len,128,score_scale or 128**.5)

    qi.qsa_mqa_decode=decode
    qi._wp_attn_mqa_installed=True


def install_cached_row_starts():
    """Eliminate fast_topk's per-call zeros launch; initialize once outside capture."""
    if os.environ.get('QWENFAST','1')!='1' or os.environ.get('QWENFAST_ATTN_TOPKZERO','1')!='1':return
    from sglang.kernels.ops.attention import fast_topk as ft
    if getattr(ft,'_wp_attn_zero_installed',False):return
    original=ft.fast_topk
    zeros={}

    def fast_topk(score,lengths,topk,row_starts=None):
        if row_starts is None and topk==512 and score.is_cuda and 1<=score.shape[0]<=8 and not torch.compiler.is_compiling():
            key=score.device
            z=zeros.get(key)
            if z is None and not torch.cuda.is_current_stream_capturing():
                z=torch.zeros(8,device=score.device,dtype=torch.int32)
                # One-time warmup synchronization makes this immutable tensor safe
                # on every later capture stream. Never synchronize inside capture.
                torch.cuda.current_stream(score.device).synchronize()
                zeros[key]=z
            if z is not None:row_starts=z[:score.shape[0]]
        return original(score,lengths,topk,row_starts=row_starts)

    ft.fast_topk=fast_topk
    ft._wp_attn_zero_installed=True


def install_index_linear(extension):
    if (os.environ.get('QWENFAST','1')!='1' or os.environ.get('QWENFAST_ATTN_INDEX_LINEAR','0')!='1'
            or not hasattr(extension,'attn_index_linear')):return
    from sglang.srt import qwenfast as qf
    if getattr(qf,'_wp_attn_index_linear_installed',False):return
    original=qf.try_smallm

    def try_smallm(x,w,bias=None):
        if (bias is None and x.is_cuda and x.dtype==torch.bfloat16 and x.ndim==2
                and x.shape[0]==4 and x.shape[1]==2560 and x.is_contiguous()
                and w.is_cuda and w.device==x.device and w.dtype==torch.bfloat16 and w.shape==(640,2560) and w.is_contiguous()
                and not torch.compiler.is_compiling()):
            return extension.attn_index_linear(x,w)
        return original(x,w,bias)

    qf.try_smallm=try_smallm
    qf._wp_attn_index_linear_installed=True
