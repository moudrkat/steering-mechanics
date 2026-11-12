# steering-mechanics

How steering vectors behave inside the model: how strong is too strong, and where the damage comes from.
Experiments run against a live model through [brainscope](https://github.com/moudrkat/brainscope)'s HTTP API.

> Work in progress. Lab notebook with every result and its caveats: [FINDINGS.md](FINDINGS.md).

## Dose: compare steering strength in relative units

The raw scale of a steering vector means nothing across layers or models: it multiplies a vector with its own
norm, added to a residual stream with its own norm. The comparable unit is the fraction of the residual stream
you perturb:

    relative_dose(L) = ‖scale · V[L]‖ / ‖h[L]‖

Measured on Qwen3-4B-Instruct, layer 20 (‖V‖ = 13.2, ‖h‖ = 54.9 over a 12k-token context):

| relative dose | what happens |
|---|---|
| 0.72 | working point |
| 1.13–1.44 | degradation starts |
| 1.93 | collapse |

Measure ‖h‖ at the context length you deploy at. On short prompts a few sink tokens dominate the mean: the same
layer reads 375 on five short prompts and 54.9 at 12k tokens, which makes every dose look seven times smaller
than it is.

Whether the same relative dose means the same thing on other models is not settled yet.

## Run it

```bash
pip install -e .
make demo        # renders the measured results in examples/ into fig/, no GPU
```

Live experiments need a running brainscope:

```bash
pip install brainscope
brainscope --model Qwen/Qwen3-4B-Instruct-2507 --port 8010     # on a GPU box
export BRAINSCOPE_BASE=http://<gpu-host>:8010
```

## Related

- [hidden-directions](https://github.com/moudrkat/hidden-directions): make, calibrate and audit a steering vector
- [hotwire-vllm](https://github.com/moudrkat/hotwire-vllm): serve it per request in vLLM

## License

MIT
