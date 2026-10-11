"""Does vLLM's fused reshape_and_cache_flash write correctly into (a) a contiguous per-layer cache
and (b) a strided per-layer view of the block-major elastic buffer? Compares with index_put.
    python -m tests.kv_write_check
"""
import torch
import vllm._custom_ops as ops

torch.manual_seed(0)
NB, L, BS, H, D = 64, 4, 16, 8, 128
dev, dt = "cuda", torch.bfloat16
T = 37
slots = torch.randperm(NB * BS, device=dev)[:T]
k = torch.randn(T, H, D, device=dev, dtype=dt)
v = torch.randn(T, H, D, device=dev, dtype=dt)
one = torch.ones((), device=dev, dtype=torch.float32)

for name, make in (("contiguous", lambda: (torch.zeros(NB, BS, H, D, device=dev, dtype=dt),
                                           torch.zeros(NB, BS, H, D, device=dev, dtype=dt), None)),
                   ("block-major view", None)):
    if make is None:
        buf = torch.zeros(NB, L, 2, BS, H, D, device=dev, dtype=dt)
        kc, vc = buf[:, 2, 0], buf[:, 2, 1]
    else:
        kc, vc, buf = make()
    ref_k, ref_v = kc.clone(), vc.clone()
    ref_k[slots // BS, slots % BS] = k
    ref_v[slots // BS, slots % BS] = v
    ops.reshape_and_cache_flash(k, v, kc, vc, slots, "auto", one, one)
    torch.cuda.synchronize()
    ok = torch.equal(kc, ref_k) and torch.equal(vc, ref_v)
    leak = ""
    if buf is not None:  # nothing outside layer 2 may be touched
        others = torch.cat([buf[:, l] for l in range(L) if l != 2])
        leak = f" | other layers untouched: {bool((others == 0).all())}"
    print(f"{name:17s}: fused write == index_put: {ok}{leak}")
