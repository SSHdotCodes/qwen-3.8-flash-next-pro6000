"""Exercise the actual patched pool allocation and its layer/all-layer views."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import sglang.srt.mem_cache.memory_pool as shipped
path = Path(shipped.__file__)
spec = importlib.util.spec_from_file_location("wp_gdn_memory_pool", path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
for dtype in (torch.float32, torch.bfloat16):
    for enabled in (False, True):
        os.environ["QWENFAST_GDN_REPLAY"] = "1" if enabled else "0"
        shape = SimpleNamespace(conv=[(10240, 3)], temporal=(48, 128, 128),
                                num_k_heads_per_tp=16, conv_kernel=4,
                                disable_conv_window_dedup=False)
        cache = SimpleNamespace(is_kda=False, shape=shape,
                                dtype=SimpleNamespace(conv=torch.bfloat16, temporal=dtype))
        pool = mod.MambaPool(size=2, spec_state_size=1, cache_params=cache,
                             mamba_layer_ids=[0], device="cuda",
                             speculative_num_draft_tokens=4, speculative_eagle_topk=1)
        all_layers = pool.get_speculative_mamba2_params_all_layers()
        layer = pool.mamba2_layer_cache(0)
        assert all_layers.temporal.dtype == dtype and layer.temporal.dtype == dtype
        assert (all_layers.gdn_replay_active is not None) == enabled
        assert (layer.gdn_replay_active is not None) == enabled
        if enabled:
            assert layer.gdn_replay_k.shape == (2, 4, 16, 128)
            assert layer.gdn_replay_d.dtype == torch.float32
            assert layer.gdn_replay_k.data_ptr() == all_layers.gdn_replay_k.data_ptr()
        print(f"pool activation and layer views: dtype={dtype}, enabled={enabled}, reason={pool.gdn_replay_reason}", flush=True)
        del layer, all_layers, pool
print("pool activation checks passed", flush=True)
