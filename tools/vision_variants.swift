// vision_variants.swift — wrists from three Apple Vision requests, side by side.
//
// Follow-on to vision_pose.swift for the bench-press investigation (app
// LEDGER D25). On each frame it runs:
//   body2d  VNDetectHumanBodyPoseRequest    (what the app uses)
//   body3d  VNDetectHumanBodyPose3DRequest  (iOS 17 / macOS 14), wrists projected
//           to the image with pointInImage
//   hand    VNDetectHumanHandPoseRequest    (up to 4 hands), each hand's .wrist joint
//
//   swift tools/vision_variants.swift <frames_dir> <out.json>
//
// Output per frame, pixels with the origin top left:
//   body2d: [{kp: COCO-17 [[x, y, c]], box}]  (same shape as vision_pose.swift)
//   body3d: [{lw: [x, y], rw: [x, y], torso: [x, y], height_m}]  (no per-joint confidence)
//   hand:   [{wrist: [x, y, c], chirality: "left" | "right" | "unknown", conf}]
// plus per-request total seconds.

import CoreGraphics
import Foundation
import ImageIO
import Vision

let cocoOrder: [VNHumanBodyPoseObservation.JointName] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
    .leftWrist, .rightWrist, .leftHip, .rightHip,
    .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
]

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("usage: vision_variants.swift <frames_dir> <out.json>\n".data(using: .utf8)!)
    exit(2)
}
let framesDir = URL(fileURLWithPath: args[1])
let files = try FileManager.default.contentsOfDirectory(atPath: framesDir.path)
    .filter { $0.hasSuffix(".jpg") }.sorted()

var frames: [[String: Any]] = []
var secs: [String: Double] = ["body2d": 0, "body3d": 0, "hand": 0]
var size = [0, 0]

func timed(_ key: String, _ f: () throws -> Void) rethrows {
    let t0 = DispatchTime.now().uptimeNanoseconds
    try f()
    secs[key, default: 0] += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
}

for name in files {
    guard let src = CGImageSourceCreateWithURL(framesDir.appendingPathComponent(name) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
    let w = Double(img.width), h = Double(img.height)
    size = [img.width, img.height]
    func px(_ p: CGPoint) -> [Double] { [Double(p.x) * w, (1 - Double(p.y)) * h] }

    let b2 = VNDetectHumanBodyPoseRequest()
    let b3 = VNDetectHumanBodyPose3DRequest()
    let hd = VNDetectHumanHandPoseRequest()
    hd.maximumHandCount = 4
    // Separate handlers so each request's time is its own.
    try timed("body2d") { try VNImageRequestHandler(cgImage: img).perform([b2]) }
    try? timed("body3d") { try VNImageRequestHandler(cgImage: img).perform([b3]) }
    try timed("hand") { try VNImageRequestHandler(cgImage: img).perform([hd]) }

    var body2d: [[String: Any]] = []
    for obs in b2.results ?? [] {
        guard let pts = try? obs.recognizedPoints(.all) else { continue }
        var kp: [[Double]] = []
        var xs: [Double] = [], ys: [Double] = []
        for j in cocoOrder {
            if let p = pts[j], p.confidence > 0 {
                let q = px(p.location)
                kp.append(q + [Double(p.confidence)])
                xs.append(q[0]); ys.append(q[1])
            } else { kp.append([0, 0, 0]) }
        }
        body2d.append(["kp": kp, "box": xs.isEmpty ? [0.0, 0, 0, 0] : [xs.min()!, ys.min()!, xs.max()!, ys.max()!]])
    }

    var body3d: [[String: Any]] = []
    for obs in b3.results ?? [] {
        var e: [String: Any] = ["height_m": Double(obs.bodyHeight)]
        if let l = try? obs.pointInImage(.leftWrist) { e["lw"] = px(l.location) }
        if let r = try? obs.pointInImage(.rightWrist) { e["rw"] = px(r.location) }
        var tx: [Double] = [], ty: [Double] = []
        for j in [VNHumanBodyPose3DObservation.JointName.leftShoulder, .rightShoulder, .leftHip, .rightHip] {
            if let p = try? obs.pointInImage(j) { let q = px(p.location); tx.append(q[0]); ty.append(q[1]) }
        }
        if !tx.isEmpty { e["torso"] = [tx.reduce(0, +) / Double(tx.count), ty.reduce(0, +) / Double(ty.count)] }
        body3d.append(e)
    }

    var hands: [[String: Any]] = []
    for obs in hd.results ?? [] {
        guard let wr = try? obs.recognizedPoint(.wrist), wr.confidence > 0 else { continue }
        let chir: String
        switch obs.chirality {
        case .left: chir = "left"
        case .right: chir = "right"
        default: chir = "unknown"
        }
        hands.append(["wrist": px(wr.location) + [Double(wr.confidence)], "chirality": chir,
                      "conf": Double(obs.confidence)])
    }
    frames.append(["frame": name, "body2d": body2d, "body3d": body3d, "hand": hands])
}

let out: [String: Any] = ["frame_size": size, "seconds": secs, "frames": frames]
try JSONSerialization.data(withJSONObject: out).write(to: URL(fileURLWithPath: args[2]))
let n = Double(max(files.count, 1))
print(String(format: "variants: %d frames  body2d %.1f  body3d %.1f  hand %.1f ms/frame",
             files.count, 1000 * secs["body2d"]! / n, 1000 * secs["body3d"]! / n, 1000 * secs["hand"]! / n))
