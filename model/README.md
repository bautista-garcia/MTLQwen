# Model creation

## Weight lookup helpers

The constructor tracks the current GGUF in `g` and the current tensor-name prefix in `r`. Three helpers bind weights without copying them:

- `w(name)` looks up `r + name` and returns its tensor view, `(k, n)` shape, and quantization type.
- `t(name)` returns only that tensor view for a specialized kernel.
- `p(name)` creates a `Linear` containing the view and its matching decode and prefill pipelines.

## Weight objects

```cpp
struct Tensor {
  std::shared_ptr<Buffer> buffer;
  uint64_t offset = 0, bytes = 0;
};

struct Linear {
  Tensor weight;
  Pipeline* pipeline[4]{};
  uint32_t k = 0, n = 0;
  uint8_t outputsPerGroup = 0;
};
```

Every weight is ultimately a `Tensor` view. `Linear` adds its shape and the decode, decode-add, prefill, and small-prefill pipelines required by the
standard matrix-multiplication path; specialized kernels consume plain tensors directly.

## Layer composition

This is what gets built during weight loading from views into the memory-mapped GGUF and pipelines selected from each weight's type and shape:

```text
Model
├── Tensor embedding
├── Tensor output norm
├── Linear output head
│
├── Layer[32]
│   ├── bool fullAttention
│   ├── kvIndex
│   ├── Tensor inputNorm
│   ├── Tensor postNorm
│   │
│   ├── MlpWeights
│   │   ├── Linear gate
│   │   ├── Linear up
│   │   ├── Linear down
│   │   └── fused decode pipeline
│   │
│   ├── AttentionWeights
│   │   ├── Linear Q
│   │   ├── Linear K
│   │   ├── Linear V
│   │   ├── Linear output
│   │   ├── Tensor Q norm
│   │   └── Tensor K norm
│   │
│   └── GdnWeights
│       ├── Linear QKV
│       ├── Linear gate
│       ├── Linear output
│       ├── Tensor beta
│       ├── Tensor alpha
│       ├── Tensor convolution
│       ├── Tensor norm
│       ├── Tensor dt
│       └── Tensor A
│
├── MtpWeights
└── DflashWeights
```

All target layers contain norms and MLP weights. `fullAttention` determines whether the attention or GDN branch is populated.

## Runtime initialization

1. Target and optional DFlash RoPE tables are generated once for the complete context.
2. One shared control allocation holds batch metadata, tokens, and RNG state; private arenas hold MTP seeds and the two banks of GDN state.
3. `SparseKV` creates the virtual KV address space for every slot. Physical pages are allocated and mapped only when sessions require them.
