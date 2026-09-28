"""Frozen three-BF16-product linear kernel with FP32 accumulation.

Used only for explicitly selected, numerically qualified feature rebuilding.
This is not equivalent to casting the entire ESM model to BF16.
"""
import types
import torch
import triton
import triton.language as tl

@triton.jit
def linear_kernel(X, W, Y, Bias, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in range(tl.cdiv(K, BK)):
        ks = kk * BK + rk
        x = tl.load(X + rm[:, None] * K + ks[None, :], (rm[:, None] < M) & (ks[None, :] < K), 0)
        w = tl.load(W + rn[None, :] * K + ks[:, None], (rn[None, :] < N) & (ks[:, None] < K), 0)
        xh = x.to(tl.bfloat16)
        wh = w.to(tl.bfloat16)
        xl = (x - xh.to(tl.float32)).to(tl.bfloat16)
        wl = (w - wh.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(xh, wh) + tl.dot(xh, wl) + tl.dot(xl, wh)
    if HAS_BIAS:
        acc += tl.load(Bias + rn, rn < N, 0)[None, :]
    tl.store(Y + rm[:, None] * N + rn[None, :], acc, (rm[:, None] < M) & (rn[None, :] < N))

def accelerated_linear(self, x):
    if x.dtype != torch.float32 or not x.is_cuda:
        return torch.nn.functional.linear(x, self.weight, self.bias)
    shape = x.shape[:-1]
    k = x.shape[-1]
    x = x.contiguous().view(-1, k)
    n = self.weight.shape[0]
    y = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    linear_kernel[triton.cdiv(x.shape[0], 64), triton.cdiv(n, 128)](x, self.weight, y, self.bias if self.bias is not None else y, x.shape[0], n, k, self.bias is not None, 64, 128, 32, num_warps=8, num_stages=3)
    return y.view(*shape, n)

def install(model):
    modules = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            modules.append((mod, mod.forward))
            mod.forward = types.MethodType(accelerated_linear, mod)
    return modules
