# Released steering vectors (paper 1 artifacts)

Tensor-only releases of every vector used in the channel-factorization
paper. Each `.safetensors` file holds a single tensor under the key
`"vector"`: **per-layer direction rows — row L is injected at the
output of layer L** (shape `[n_layers, d_model]`, see
`manifest.json` for shapes, dtypes and sha256 prefixes).

```python
from safetensors.torch import load_file
v = load_file("task_suppression_qwen3-4b.safetensors")["vector"]  # [36, 2560]
# inject s * v[20] at layer-20 output, decode positions only
```

Provenance: baked with [hidden-directions](https://github.com/moudrkat/hidden-directions)
from a private behavior recipe (task-offering suppression) or public
data (MWE behaviors — also rebuildable via
`experiments/skop_residual/bake_mwe.py`). The tensors encode only the
direction; no prompt content is recoverable from them. The private
recipe is not required to reproduce any paper number — scripts +
these vectors + the generic prompt sets published in the scripts
suffice for every row of Table 1.

| file | model | used in |
|---|---|---|
| task_suppression_qwen3-4b | Qwen3-4B-Instruct-2507 | Table 1, Figs 1–5 |
| task_suppression_qwen3-8b | Qwen3-8B | Table 1, Fig 2 |
| task_suppression_qwen2.5-7b | Qwen2.5-7B-Instruct | Table 1, Fig 2 (H2) |
| task_suppression_llama31-8b | Llama-3.1-8B-Instruct | Table 1, Fig 2 |
| task_suppression_gemma-4-e4b | Gemma-4-E4B | Table 1, Fig 2 (H4) |
| websearch_overtrigger_qwen3-4b | Qwen3-4B | Table 1, Fig 3 (H1) |
| sycophancy_qwen3-4b | Qwen3-4B | Table 1, Fig 3 (H1) |
| confidence_qwen3-4b | Qwen3-4B | Table 1, Fig 3 (H1) |
| refusal_qwen3-4b | Qwen3-4B | Table 1, Fig 3 (H1) |
| random_ctl_r1536_s1_qwen3-4b | Qwen3-4B | Table 1, Fig 3 (control) |
