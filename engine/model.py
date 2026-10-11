"""Llama forward pass with a paged KV cache.

Our own implementation (no vLLM model code): HF safetensors weights, fused QKV and gate/up
projections, Llama-3 RoPE scaling, and attention through vLLM's bundled FlashAttention kernel
(`flash_attn_varlen_func` with `block_table`). That is the same kernel the vLLM baselines use,
so engine-vs-vLLM differences come from scheduling/control, not from the attention kernel.

One forward call handles any mix of prefill chunks, single-token decodes and (γ+1)-token
speculative verifications: each sequence contributes `q_len` new tokens appended after
`seq_len - q_len` cached tokens, and causal masking is aligned bottom-right by the kernel.
Numerics follow HF transformers (fp32 RMSNorm and RoPE tables, cast to bf16) so that greedy
outputs can be checked token for token against `generate()` (experiment M4).
"""
import glob
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass
class LlamaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool
    eos_token_id: int | list
    max_position_embeddings: int

    @classmethod
    def from_json(cls, path):
        c = json.load(open(path))
        c.setdefault("head_dim", c["hidden_size"] // c["num_attention_heads"])
        return cls(**{k: c.get(k) for k in cls.__dataclass_fields__})

    def kv_bytes_per_token(self, dtype=torch.bfloat16):
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype.itemsize


@dataclass
class Batch:
    """Flattened inputs for one forward step (all tensors on the GPU)."""
    input_ids: torch.Tensor       # [T] int64, new tokens of all sequences back to back
    positions: torch.Tensor       # [T] int64
    cu_seqlens_q: torch.Tensor    # [B+1] int32, offsets of each sequence's new tokens
    seq_lens: torch.Tensor        # [B] int32, cached + new tokens per sequence
    block_table: torch.Tensor     # [B, max_blocks] int32
    slot_mapping: torch.Tensor    # [T] int64, cache slot where each new token's K/V is written
    logits_indices: torch.Tensor  # [N] int64, rows of the hidden states that need logits
    max_seqlen_q: int
    max_seqlen_k: int


def llama3_inv_freq(cfg: LlamaConfig):
    dim = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    rs = cfg.rope_scaling
    if not rs or rs.get("rope_type") != "llama3":
        return inv_freq
    factor, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], \
        rs["original_max_position_embeddings"]
    low_wavelen, high_wavelen = old / lo, old / hi
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old / wavelen - lo) / (hi - lo)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def rms_norm(x, weight, eps):
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Llama:
    def __init__(self, cfg: LlamaConfig, weights: dict, device="cuda", dtype=torch.bfloat16):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.n_heads, self.n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        w = weights  # fused tensors are popped as we go so peak memory stays ~1x the model
        self.embed = w["model.embed_tokens.weight"]
        self.norm = w["model.norm.weight"]
        self.lm_head = self.embed if cfg.tie_word_embeddings else w["lm_head.weight"]
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": w[p + "input_layernorm.weight"],
                "ln2": w[p + "post_attention_layernorm.weight"],
                "qkv": torch.cat([w.pop(p + "self_attn.q_proj.weight"), w.pop(p + "self_attn.k_proj.weight"),
                                  w.pop(p + "self_attn.v_proj.weight")], 0),
                "o": w[p + "self_attn.o_proj.weight"],
                "gate_up": torch.cat([w.pop(p + "mlp.gate_proj.weight"), w.pop(p + "mlp.up_proj.weight")], 0),
                "down": w[p + "mlp.down_proj.weight"],
            })
        # RoPE tables like HF: fp32 angles, cast to the model dtype.
        pos = torch.arange(cfg.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(pos, llama3_inv_freq(cfg))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(device=device, dtype=dtype)
        self.sin = emb.sin().to(device=device, dtype=dtype)
        self.k_cache, self.v_cache = [], []
        self.block_size = None

    @classmethod
    def from_pretrained(cls, repo, device="cuda", dtype=torch.bfloat16):
        path = repo if os.path.isdir(repo) else snapshot_download(
            repo, allow_patterns=["*.json", "*.safetensors"])
        cfg = LlamaConfig.from_json(os.path.join(path, "config.json"))
        weights = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, framework="pt", device=str(device)) as sf:
                for name in sf.keys():
                    weights[name] = sf.get_tensor(name).to(dtype)
        model = cls(cfg, weights, device, dtype)
        del weights
        torch.cuda.empty_cache()
        return model

    def allocate_kv_cache(self, num_blocks, block_size=16):
        """One [num_blocks, block_size, n_kv, head_dim] K and V tensor per layer."""
        assert block_size % 16 == 0, "FlashAttention paged KV needs block_size % 16 == 0"
        shape = (num_blocks, block_size, self.n_kv, self.head_dim)
        self.k_cache = [torch.empty(shape, device=self.device, dtype=self.dtype)
                        for _ in range(self.cfg.num_hidden_layers)]
        self.v_cache = [torch.empty_like(k) for k in self.k_cache]
        self.block_size = block_size

    def use_kv_views(self, k_list, v_list, block_size):
        """Use externally owned per-layer caches (e.g. strided views of one block-major buffer)."""
        self.k_cache, self.v_cache, self.block_size = list(k_list), list(v_list), block_size

    def weight_slots(self):
        """(container, key) of every weight tensor; a tied lm_head is the embedding, listed once."""
        slots = [(self.__dict__, "embed"), (self.__dict__, "norm")]
        if not self.cfg.tie_word_embeddings:
            slots.append((self.__dict__, "lm_head"))
        for layer in self.layers:
            slots += [(layer, k) for k in layer]
        return slots

    def retie(self):
        if self.cfg.tie_word_embeddings:
            self.lm_head = self.embed

    def kv_blocks_for_bytes(self, nbytes, block_size=16):
        return int(nbytes // (self.cfg.kv_bytes_per_token(self.dtype) * block_size))

    @torch.inference_mode()
    def forward(self, b: Batch, logits_dtype=torch.float32, out=None) -> torch.Tensor:
        """Returns logits for the rows in `b.logits_indices` (fp32 by default). CUDA graphs pass `out`,
        one shared bf16 [max_rows, vocab] buffer, so ~50 captured graphs don't each own a logits tensor."""
        T = b.input_ids.shape[0]
        H, Hkv, D = self.n_heads, self.n_kv, self.head_dim
        cos = self.cos[b.positions].unsqueeze(1)
        sin = self.sin[b.positions].unsqueeze(1)
        # (block, offset) indexing also works for strided per-layer views of a block-major buffer
        blk, off = b.slot_mapping // self.block_size, b.slot_mapping % self.block_size
        x = F.embedding(b.input_ids, self.embed)
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L["ln1"], self.cfg.rms_norm_eps)
            qkv = F.linear(h, L["qkv"])
            q, k, v = qkv.split([H * D, Hkv * D, Hkv * D], dim=-1)
            q = q.view(T, H, D)
            k = k.view(T, Hkv, D)
            v = v.view(T, Hkv, D)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            kc, vc = self.k_cache[i], self.v_cache[i]
            kc[blk, off] = k
            vc[blk, off] = v
            attn = flash_attn_varlen_func(
                q=q, k=kc, v=vc, cu_seqlens_q=b.cu_seqlens_q, max_seqlen_q=b.max_seqlen_q,
                seqused_k=b.seq_lens, max_seqlen_k=b.max_seqlen_k, softmax_scale=self.scale,
                causal=True, block_table=b.block_table, fa_version=2)
            x = x + F.linear(attn.reshape(T, H * D), L["o"])
            h = rms_norm(x, L["ln2"], self.cfg.rms_norm_eps)
            gate, up = F.linear(h, L["gate_up"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(gate) * up, L["down"])
        x = rms_norm(x[b.logits_indices], self.norm, self.cfg.rms_norm_eps)
        if out is not None:
            return torch.matmul(x, self.lm_head.t(), out=out[: x.shape[0]])
        return F.linear(x, self.lm_head).to(logits_dtype)
