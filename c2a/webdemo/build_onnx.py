"""Build ``controller.onnx`` from ``controller_weights.npz``.

Runs in the light demo venv (``.venv-webdemo``: numpy + onnx), never in JAX.
The graph is written node by node, mirroring the Flax modules exactly, so
nothing depends on a JAX->ONNX converter:

* code window ``[1, 5]`` -> ``Embed`` -> ``[1, 80]``
* raw proprio ``[1, 277]`` -> ``(x - mean) / std`` (the PPO running stats)
* four stacked ``flax.linen.GRUCell`` steps on ``[emb, proprio_n]``::

      r = sigmoid(W_ir x + b_ir + W_hr h)
      z = sigmoid(W_iz x + b_iz + W_hz h)
      n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
      h' = (1 - z) * n + z * h

* VAE head on the last layer: ``Dense -> SiLU -> LayerNorm`` x4, then the
  latent mean (deterministic mode, z = mu)
* frozen MIMIC decoder on ``[z, proprio_n]``: ``Dense -> SiLU -> LayerNorm``
  x4, then ``Dense(76)``; action = ``tanh(loc)`` with ``loc = out[:38]``.

Inputs:  ``codes`` int64[B,5], ``proprio`` f32[B,277], ``h0..h3`` f32[B,H].
Outputs: ``action`` f32[B,38], ``h0_out..h3_out``, ``z`` f32[B,16].
The browser runs B=1; the simulator-gap test batches all bodies.

Usage::

    .venv-webdemo/bin/python -m c2a.webdemo.build_onnx [--fp16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from c2a.paths import get_repo_root

OUT = get_repo_root() / "outputs/webdemo"
OPSET = 17


class Graph:
    """Tiny helper that accumulates nodes and initializers with unique names."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def _name(self, stem: str) -> str:
        self._n += 1
        return f"{stem}_{self._n}"

    def const(self, key: str, arr: np.ndarray | None = None) -> str:
        arr = self.w[key] if arr is None else arr
        name = key.replace("/", ".")
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def op(self, op_type: str, inputs: list[str], stem: str | None = None,
           out: str | None = None, **attrs) -> str:
        out = out or self._name(stem or op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def dense(self, x: str, prefix: str, bias: bool = True) -> str:
        y = self.op("MatMul", [x, self.const(f"{prefix}/kernel")], stem=prefix)
        if bias and f"{prefix}/bias" in self.w:
            y = self.op("Add", [y, self.const(f"{prefix}/bias")], stem=prefix)
        return y

    def silu(self, x: str) -> str:
        return self.op("Mul", [x, self.op("Sigmoid", [x])], stem="silu")

    def layer_norm(self, x: str, prefix: str) -> str:
        return self.op("LayerNormalization",
                       [x, self.const(f"{prefix}/scale"), self.const(f"{prefix}/bias")],
                       stem=prefix, axis=-1, epsilon=1e-6)

    def gru(self, x: str, h: str, prefix: str, out: str) -> str:
        r = self.op("Sigmoid", [self.op("Add", [self.dense(x, f"{prefix}/ir"),
                                                self.dense(h, f"{prefix}/hr")])])
        z = self.op("Sigmoid", [self.op("Add", [self.dense(x, f"{prefix}/iz"),
                                                self.dense(h, f"{prefix}/hz")])])
        n = self.op("Tanh", [self.op("Add", [
            self.dense(x, f"{prefix}/in"),
            self.op("Mul", [r, self.dense(h, f"{prefix}/hn")])])])
        # h' = n + z * (h - n)  ==  (1 - z) * n + z * h
        zh = self.op("Mul", [z, self.op("Sub", [h, n])])
        return self.op("Add", [n, zh], out=out)


def build(weights: dict[str, np.ndarray]) -> onnx.ModelProto:
    g = Graph(weights)
    emb_table = weights["code_embedding/embedding"]
    num_codes, emb_dim = emb_table.shape
    hidden_sizes = [weights[f"rnn_cells_{i}/hn/kernel"].shape[0]
                    for i in range(4) if f"rnn_cells_{i}/hn/kernel" in weights]
    stack = (weights["rnn_cells_0/ir/kernel"].shape[0] - 277) // emb_dim
    n_head = sum(1 for k in weights if k.startswith("vae_head_") and k.endswith("/kernel")
                 and k.split("/")[0][len("vae_head_"):].isdigit())
    n_dec = sum(1 for k in weights if k.startswith("decoder_module/hidden_")
                and k.endswith("/kernel"))
    nu = weights[f"decoder_module/hidden_{n_dec - 1}/kernel"].shape[1] // 2

    # Embedding of the forward code window, flattened in window order.
    emb = g.op("Gather", [g.const("code_embedding/embedding"), "codes"], axis=0)
    emb = g.op("Reshape", [emb, g.const("shape_emb", np.array([-1, stack * emb_dim]))])
    pn = g.op("Div", [g.op("Sub", ["proprio", g.const("norm/proprio_mean")]),
                      g.const("norm/proprio_std")], stem="proprio_n")

    x = g.op("Concat", [emb, pn], axis=-1, stem="gru_in")
    for i in range(len(hidden_sizes)):
        x = g.gru(x, f"h{i}", f"rnn_cells_{i}", out=f"h{i}_out")

    y = x
    for i in range(n_head):
        y = g.layer_norm(g.silu(g.dense(y, f"vae_head_{i}")), f"vae_head_ln_{i}")
    z = g.dense(y, "vae_head_mean")
    g.nodes.append(helper.make_node("Identity", [z], ["z"]))

    d = g.op("Concat", [z, pn], axis=-1, stem="dec_in")
    for i in range(n_dec - 1):
        d = g.layer_norm(g.silu(g.dense(d, f"decoder_module/hidden_{i}")),
                         f"decoder_module/LayerNorm_{i}")
    d = g.dense(d, f"decoder_module/hidden_{n_dec - 1}")
    loc = g.op("Slice", [d, g.const("loc_start", np.array([0])),
                         g.const("loc_end", np.array([nu])),
                         g.const("loc_axis", np.array([1]))])
    g.op("Tanh", [loc], out="action")

    f32 = TensorProto.FLOAT
    inputs = [helper.make_tensor_value_info("codes", TensorProto.INT64, ["batch", stack]),
              helper.make_tensor_value_info("proprio", f32, ["batch", 277])]
    inputs += [helper.make_tensor_value_info(f"h{i}", f32, ["batch", s])
               for i, s in enumerate(hidden_sizes)]
    outputs = [helper.make_tensor_value_info("action", f32, ["batch", nu])]
    outputs += [helper.make_tensor_value_info(f"h{i}_out", f32, ["batch", s])
                for i, s in enumerate(hidden_sizes)]
    outputs.append(helper.make_tensor_value_info(
        "z", f32, ["batch", weights["vae_head_mean/kernel"].shape[1]]))

    graph = helper.make_graph(g.nodes, "code2act_mlb_controller", inputs, outputs, g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="c2a.webdemo.build_onnx")
    model.ir_version = 8
    meta = {"num_codes": num_codes, "code_stack_size": stack,
            "hidden_sizes": ",".join(map(str, hidden_sizes)), "action_size": nu}
    for k, v in meta.items():
        model.metadata_props.append(onnx.StringStringEntryProto(key=k, value=str(v)))
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, default=OUT / "controller_weights.npz")
    ap.add_argument("--out", type=Path, default=OUT / "controller.onnx")
    args = ap.parse_args()
    weights = dict(np.load(args.weights))
    model = build(weights)
    used = {i.name.replace(".", "/") for i in model.graph.initializer}
    # The latent log-variance only matters when sampling; the demo is
    # deterministic (z = mu), exactly like every paper rollout.
    unused = sorted(k for k in weights if k not in used
                    and not k.startswith("vae_head_logvar/"))
    if unused:
        raise RuntimeError(f"weights not consumed by the graph: {unused}")
    onnx.save(model, args.out)
    meta = {p.key: p.value for p in model.metadata_props}
    manifest = {
        "checkpoint": "model_checkpoints/260629_212713_669229_step208",
        "code_stack_size": int(meta["code_stack_size"]),
        "hidden_sizes": [int(h) for h in meta["hidden_sizes"].split(",")],
        "action_size": int(meta["action_size"]),
        "proprio_size": 277,
        "ctrl_dt": 0.01,
        "window_convention": "ctrl",
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), "
          f"{len(model.graph.node)} nodes")


if __name__ == "__main__":
    main()
