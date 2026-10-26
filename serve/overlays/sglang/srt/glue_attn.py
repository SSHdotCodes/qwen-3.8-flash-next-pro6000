"""Install WP-Attn overrides after qwenfast has installed its ordinary overrides.

Place this module, attn_kernels.py, and norm_store.py next to qwenfast.py, then call
`from sglang.srt.glue_attn import install; install()` after model classes are defined.
Each feature is independently gated and falls back to the frozen implementation.
"""
import os
import torch


def install():
    if os.environ.get('QWENFAST','1')!='1':return
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qb
    from sglang.srt.models import qwen4_exp as qm
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from .attn_kernels import pack_counts
    from .norm_store import prepare_store
    if getattr(qm,'_wp_attn_installed',False):return
    qm._wp_attn_installed=True
    use_pack=os.environ.get('QWENFAST_ATTN_PACK','1')=='1'
    use_store=os.environ.get('QWENFAST_ATTN_STORE','1')=='1'
    use_overlap=os.environ.get('QWENFAST_ATTN_OVERLAP','1')=='1'
    old_paged=qb.QwenSparseAttnBackend._forward_trtllm_sparse
    old_attn=qm.Qwen4ExpAttentionDecoderLayer.self_attention

    if use_pack:
        def paged(self,q,k_buffer,v_buffer,layer,forward_batch,metadata,topk_indices,trtllm_decode):
            m,topk=topk_indices.shape
            if not (1<=m<=8 and q.dtype==torch.bfloat16 and q.is_cuda
                    and q.shape[1:]==(24,256) and k_buffer.shape[1:]==(2,256)
                    and k_buffer.dtype in (torch.float8_e4m3fn,torch.bfloat16)
                    and v_buffer.dtype==k_buffer.dtype and k_buffer.is_contiguous()
                    and v_buffer.is_contiguous() and v_buffer.shape==k_buffer.shape and 1<=topk<=4096):
                return old_paged(self,q,k_buffer,v_buffer,layer,forward_batch,metadata,topk_indices,trtllm_decode)
            page=64;pages=(topk+page-1)//page;stride=pages*page
            counts=metadata.fa2_valid_counts if metadata.is_cuda_graph else torch.empty(m,dtype=torch.int32,device=q.device)
            if counts is None:
                return old_paged(self,q,k_buffer,v_buffer,layer,forward_batch,metadata,topk_indices,trtllm_decode)
            _,tables=self._get_trtllm_sparse_tables(m,pages,page,q.device)
            capacity=self._cuda_graph_max_tokens if metadata.is_cuda_graph else m
            pk,pv=self._get_fa2_scratch(max(capacity,m)*stride,2,256,q.dtype,k_buffer.device)
            req=(metadata.row_req_pool_indices if metadata.row_req_pool_indices is not None else forward_batch.req_pool_indices)
            pack_counts(k_buffer,v_buffer,self.req_to_token_pool.req_to_token,req,topk_indices,
                        metadata.sequence_lengths,pk,pv,counts,stride)
            kc=pk[:m*stride].view(-1,page,2,256).permute(0,2,1,3)
            vc=pv[:m*stride].view(-1,page,2,256).permute(0,2,1,3)
            if self._trtllm_workspace is None:
                self._trtllm_workspace=torch.zeros(128*1024*1024,dtype=torch.uint8,device=q.device)
            output=trtllm_decode(query=q.contiguous(),kv_cache=(kc,vc),workspace_buffer=self._trtllm_workspace,
                                 block_tables=tables,seq_lens=counts,max_seq_len=stride,
                                 bmm1_scale=layer.scaling,bmm2_scale=1.)
            return output.reshape(m,-1)
        qb.QwenSparseAttnBackend._forward_trtllm_sparse=paged

    if use_store or use_overlap:
        def attention(self,positions,hidden_states,forward_batch):
            if not (self.is_qsa and hidden_states.is_cuda and hidden_states.dtype==torch.bfloat16
                    and hidden_states.ndim==2 and 1<=hidden_states.shape[0]<=8
                    and not torch.compiler.is_compiling()):
                return old_attn(self,positions,hidden_states,forward_batch)
            sparse=resolve_qsa_sparse_backend(qm.get_attn_backend())
            if not isinstance(sparse,qb.QwenSparseAttnBackend):
                return old_attn(self,positions,hidden_states,forward_batch)
            overlap=(self.alt_stream is not None and qm.get_is_capture_mode()
                     and hidden_states.shape[0]<qm._QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD)
            if not use_overlap and not use_store:
                return old_attn(self,positions,hidden_states,forward_batch)
            pool=sparse.token_to_kv_pool
            mode=forward_batch.forward_mode
            paged=(mode.is_decode() or sparse._is_speculative_paged_mode(mode))
            if not paged:
                return old_attn(self,positions,hidden_states,forward_batch)
            current=torch.cuda.current_stream()
            if overlap:self.alt_stream.wait_stream(current)
            kcache=pool.get_key_buffer(self.attn.layer_id);vcache=pool.get_value_buffer(self.attn.layer_id)
            loc=forward_batch.out_cache_loc
            fuse=(use_store and isinstance(loc,torch.Tensor) and loc.is_cuda and loc.device==hidden_states.device
                  and loc.numel()==hidden_states.shape[0]
                  and loc.dtype in (torch.int32,torch.int64) and loc.is_contiguous()
                  and tuple(kcache.shape[1:])==(2,256) and kcache.is_contiguous()
                  and vcache.is_contiguous() and kcache.dtype==torch.float8_e4m3fn
                  and vcache.dtype==kcache.dtype and self.attn_output_gate
                  and self.num_heads==24 and self.num_kv_heads==2 and self.head_dim==256
                  and self.rotary_emb.rotary_dim==64 and self.rotary_emb.is_neox_style
                  and positions.is_cuda and positions.device==hidden_states.device
                  and positions.dtype==torch.int64 and positions.stride(-1)==1
                  and positions.ndim in (1,2) and self.rotary_emb.cos_sin_cache.dtype==torch.float32)
            if fuse:
                qkv,_=self.qkv_proj(hidden_states)
                q,k,v,gate=prepare_store(qkv,self.q_norm.weight.data,self.k_norm.weight.data,
                    self.rotary_emb.cos_sin_cache,positions,loc,kcache,vcache,
                    self.q_norm.variance_epsilon,mrope_axis_map=(self.rotary_emb.axis_map if positions.ndim==2 else None))
            else:
                q,k,v,gate=self._prepare_qkv_gate(positions,hidden_states,forward_batch)
            if overlap:
                with torch.cuda.stream(self.alt_stream):
                    indices=self._compute_qsa_topk_indices(hidden_states,positions,forward_batch)
                # KV storage uses separate buffers and can run beside the indexer.
                if not fuse and use_overlap:
                    pool.set_kv_buffer(self.attn,loc,k.view(-1,2,256),v.view(-1,2,256))
                current.wait_stream(self.alt_stream)
                indices.record_stream(current)
            else:
                indices=self._compute_qsa_topk_indices(hidden_states,positions,forward_batch)
            saved=fuse or (overlap and use_overlap)
            out=self.attn(q,k,v,forward_batch,save_kv_cache=not saved,topk_indices=indices)
            if gate is not None:out=qm.fused_sigmoid_mul(out,gate,inplace=True)
            output,_=self.o_proj(out)
            return output
        qm.Qwen4ExpAttentionDecoderLayer.self_attention=attention
