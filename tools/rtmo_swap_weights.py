"""rtmo_swap_weights.py — rebuild an RTMO ONNX export with another checkpoint's weights.

OpenMMLab publishes RTMO ONNX exports only for the body7 checkpoints, whose
training data includes datasets with non-commercial terms. The COCO-only
checkpoint is published only as a PyTorch .pth. Both have the same
architecture, so this script swaps the COCO weights into the body7 graph
without needing mmpose, mmcv or mmdeploy.

The export folded BatchNorm into convolutions, merged the RepVGG branches, and
turned some head weights into constants. Rather than reproduce the exporter, the
mapping is learned by exact numeric match:
  1. derive every tensor the export could contain from the SOURCE checkpoint
     (raw, transposed, conv+BN folded at eps 1e-3 and 1e-5, RepVGG merged,
     Linear+BN1d folded, q/k/v split)
  2. match each ONNX weight to the derivation that reproduces it
  3. apply the same derivations to the NEW checkpoint
GAU gamma rows and beta+pos_enc constants are handled explicitly. Fixed tensors
(the sincos encoding, shape constants, anchors) are left alone. The weights the
graph never uses are training-only (aux keypoint regression, sigma branch, OKS
sigmas). Check: swapping body7 into its own graph reproduces its outputs
exactly (0.0 px).

Needs onnx and torch (the Python 3.12 env):

    <env>/bin/python tools/rtmo_swap_weights.py \\
        rtmo-s_body7/end2end.onnx rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth \\
        rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth rtmo-s_coco/end2end.onnx

Checkpoints: download.openmmlab.com/mmpose/v1/projects/rtmo/<name>.pth.
"""

import collections
import sys

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

def fuse(sd, p, eps=1e-3):
    w = sd[p + ".conv.weight"].double()
    g, b = sd[p + ".bn.weight"].double(), sd[p + ".bn.bias"].double()
    m, v = sd[p + ".bn.running_mean"].double(), sd[p + ".bn.running_var"].double()
    s = g / torch.sqrt(v + eps)
    cb = sd.get(p + ".conv.bias")
    cb = torch.zeros_like(m) if cb is None else cb.double()
    return w * s[:, None, None, None], (cb - m) * s + b


def candidates(sd, eps):
    """name -> float32 numpy array."""
    out = {}
    def add(k, t):
        a = t.detach().double().numpy() if torch.is_tensor(t) else np.asarray(t, float)
        out[k] = a.astype(np.float32)
        if a.ndim == 2:
            out[k + "^T"] = a.T.astype(np.float32).copy()
    for k, t in sd.items():
        if t.dtype.is_floating_point and t.numel() > 1:
            add(k, t)
            if k.endswith("in_proj_weight") or k.endswith("in_proj_bias"):
                for i, c in enumerate(t.chunk(3, 0)):
                    add(f"{k}[{i}]", c)
    convs = {k[:-len(".conv.weight")] for k in sd if k.endswith(".conv.weight") and k[:-len(".conv.weight")] + ".bn.weight" in sd}
    for p in convs:
        w, b = fuse(sd, p, eps)
        add(p + "#fw", w); add(p + "#fb", b)
    reps = {k.rsplit(".branch_3x3", 1)[0] for k in sd if ".branch_3x3." in k}
    for p in reps:
        w3, b3 = fuse(sd, p + ".branch_3x3", eps)
        w1, b1 = fuse(sd, p + ".branch_1x1", eps)
        add(p + "#rw", w3 + F.pad(w1, [1, 1, 1, 1])); add(p + "#rb", b3 + b1)
    # Linear + BatchNorm1d (pose_to_kpts)
    for p in {k.rsplit(".0.weight", 1)[0] for k in sd if k.endswith(".0.weight") and k.replace(".0.weight", ".1.running_var") in sd}:
        W, B = sd[p + ".0.weight"].double(), sd[p + ".0.bias"].double()
        g, be = sd[p + ".1.weight"].double(), sd[p + ".1.bias"].double()
        m, v = sd[p + ".1.running_mean"].double(), sd[p + ".1.running_var"].double()
        for e in (1e-5, 1e-3):
            s = g / torch.sqrt(v + e)
            add(f"{p}#lw{e}", W * s[:, None]); add(f"{p}#lb{e}", (B - m) * s + be)
            add(f"{p}#bns{e}", s); add(f"{p}#bnb{e}", be - m * s)
    return out


def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    src_onnx, src_pth, new_pth, out = sys.argv[1:5]
    m = onnx.load(src_onnx)
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
    A, B = load(src_pth), load(new_pth)
    both = lambda sd: {**{k + "@3": v for k, v in candidates(sd, 1e-3).items()},
                       **{k + "@5": v for k, v in candidates(sd, 1e-5).items()}}
    ca, cb = both(A), both(B)
    by = collections.defaultdict(list)
    for k, a in ca.items():
        by[a.shape].append(k)
    FIXED = {"neck.pos_enc_0"}  # sincos encoding, not learned
    
    def resolve(name, a):
        ks = [k for k in by.get(a.shape, []) if np.allclose(ca[k], a, rtol=1e-4, atol=1e-6)]
        if not ks:
            return None
        ks.sort(key=lambda k: float(np.abs(ca[k] - a).max()))
        # a tie between different modules is only safe if they agree in the new checkpoint
        rivals = [k for k in ks if k.rsplit("@", 1)[0] != ks[0].rsplit("@", 1)[0]]
        if any(not np.allclose(cb[k], cb[ks[0]], rtol=1e-4, atol=1e-6) for k in rivals):
            raise SystemExit(f"ambiguous {name}: {ks}")
        return ks[0]
    
    swapped, kept = 0, []
    for init in m.graph.initializer:
        a = numpy_helper.to_array(init)
        if a.dtype != np.float32 or init.name in FIXED:
            kept.append(init.name); continue
        if a.size == 1:
            k = init.name if init.name in A else None
            if k is None:
                kept.append(init.name); continue
            if not np.isclose(float(A[k]), float(a.ravel()[0])): raise SystemExit(f"scalar mismatch {k}")
            new = B[k].numpy().astype(np.float32).reshape(a.shape)
        else:
            k = resolve(init.name, a)
            if k is None:
                kept.append(init.name); continue
            new = cb[k].reshape(a.shape)
        init.CopyFrom(numpy_helper.from_array(new, init.name)); swapped += 1
    
    g, be, pe = (x.numpy() for x in (A["head.dcc.gau.gamma"], A["head.dcc.gau.beta"], A["head.dcc.pos_enc"]))
    g2, be2, pe2 = (x.numpy() for x in (B["head.dcc.gau.gamma"], B["head.dcc.gau.beta"], B["head.dcc.pos_enc"]))
    consts = 0
    for n in m.graph.node:
        if n.op_type != "Constant": continue
        t = n.attribute[0].t; a = numpy_helper.to_array(t)
        if a.dtype != np.float32: continue
        new = None
        for i in (0, 1):
            if a.size == 128 and np.allclose(a.ravel(), g[i]): new = g2[i]
            if a.size == 17 * 128 and np.allclose(a.reshape(17, 128), be[i] + pe, atol=1e-5): new = be2[i] + pe2
        if new is not None:
            t.CopyFrom(numpy_helper.from_array(np.asarray(new, np.float32).reshape(a.shape), t.name)); consts += 1
    print(f"swapped {swapped} initializers, {consts} constants; kept as is: {kept}")
    assert consts == 4
    onnx.checker.check_model(m)
    onnx.save(m, out)


if __name__ == "__main__":
    main()
