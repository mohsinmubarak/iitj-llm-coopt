"""Does vLLM's FlashAttention accept a block-major KV buffer (per-layer views strided over blocks)?

Compares attention over a block-major buffer [NB, L, 2, bs, H, D] (per-layer view buf[:, l, 0])
with the same data in an ordinary contiguous per-layer cache. Small tensors: runs beside a busy GPU.
    python -m tests.strided_kv_check
"""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
NB, L, BS, HKV, HQ, D = 64, 4, 16, 8, 32, 128
dev, dt = "cuda", torch.bfloat16
buf = torch.randn(NB, L, 2, BS, HKV, D, device=dev, dtype=dt)
layer = 2
k_strided, v_strided = buf[:, layer, 0], buf[:, layer, 1]
k_contig, v_contig = k_strided.contiguous(), v_strided.contiguous()
print("strided view strides:", k_strided.stride(), "contiguous:", k_strided.is_contiguous())

seq_lens = torch.tensor([37, 200, 5], dtype=torch.int32, device=dev)
q_lens = [1, 3, 5]
cu = torch.tensor([0, 1, 4, 9], dtype=torch.int32, device=dev)
table = torch.randperm(NB, device=dev)[:3 * 16].view(3, 16).to(torch.int32)
q = torch.randn(sum(q_lens), HQ, D, device=dev, dtype=dt)
kw = dict(q=q, cu_seqlens_q=cu, max_seqlen_q=5, seqused_k=seq_lens, max_seqlen_k=200,
          softmax_scale=D ** -0.5, causal=True, block_table=table, fa_version=2)
a = flash_attn_varlen_func(k=k_contig, v=v_contig, **kw)
b = flash_attn_varlen_func(k=k_strided, v=v_strided, **kw)
print("max |diff| strided vs contiguous:", (a - b).abs().max().item())

# the write path the model uses for strided caches: index by (block, offset)
slots = torch.tensor([5, 17, 900], device=dev)
new = torch.randn(3, HKV, D, device=dev, dtype=dt)
k_strided[slots // BS, slots % BS] = new
print("write through strided view reaches the buffer:",
      torch.equal(buf[slots // BS, layer, 0, slots % BS], new))
