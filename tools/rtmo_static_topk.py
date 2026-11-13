"""rtmo_static_topk.py — make an RTMO ONNX export fully static, for Core ML.

The mmdeploy export ends in NonMaxSuppression plus a score threshold, which
give data-dependent shapes. ONNX Runtime's Core ML provider cannot take
unbounded dimensions, so most of the model falls back to the CPU (~100 ms/frame
on an M2 either way).

This replaces the NMS result with a fixed list: the top K of the 2000
candidates, which the graph has already sorted by score. Keypoints are then
decoded for all K, and the input is fixed at 1x3x640x640. The outputs become
dets [1, K+1, 5] and keypoints [1, K+1, 17, 3]; the extra row is the export's
zero padding. NMS (IoU 0.45) and the score threshold (0.3) move to the caller,
which rtmlib already does after the model anyway.

Measured on an M2, 60 deadlift4 frames: CPU 108 ms/frame, Core ML (ALL) 23,
CPU+GPU 25, CPU+Neural Engine 76. The lifter's keypoints after NMS are
identical to the original model's (under 0.01 px).

    .venv312/bin/python tools/rtmo_static_topk.py models/rtmo/coco/end2end.onnx \\
        models/rtmo/coco/static_k20.onnx 20
"""

import collections
import sys

import numpy as np
import onnx
import onnxsim
from onnx import helper, numpy_helper, shape_inference


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    src, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    m = onnx.load(src)
    nms = next(n for n in m.graph.node if n.op_type == "NonMaxSuppression")
    sel = nms.output[0]
    m.graph.node.remove(nms)
    idx = np.array([[0, 0, i] for i in range(k)], np.int64)  # (batch, class, box)
    m.graph.node.insert(0, helper.make_node("Constant", [], [sel], value=numpy_helper.from_array(idx, sel)))
    m, ok = onnxsim.simplify(m, overwrite_input_shapes={"input": [1, 3, 640, 640]})
    m = shape_inference.infer_shapes(m)
    vi = {v.name: [d.dim_value if d.HasField("dim_value") else "?" for d in v.type.tensor_type.shape.dim]
          for v in list(m.graph.value_info) + list(m.graph.output)}
    dyn = sum("?" in v for v in vi.values())
    print("simplified:", ok, "outputs:", [(o.name, vi[o.name]) for o in m.graph.output])
    print("dynamic tensors left:", dyn)
    print(collections.Counter(n.op_type for n in m.graph.node).most_common(8))
    if dyn:
        raise SystemExit("graph still has dynamic shapes")
    onnx.save(m, out)


if __name__ == "__main__":
    main()
