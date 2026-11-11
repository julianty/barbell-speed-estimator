// vision_pose.swift — run Apple Vision body pose over a directory of frames.
//
// Part of the pose-backend comparison (tools/compare_pose_backends.py). Runs
// VNDetectHumanBodyPoseRequest on every .jpg in a frames directory and writes
// one JSON file with each person's joints remapped to COCO-17 order, so the
// existing wrist-midpoint tracker (COCO 9 + 10) can consume it unchanged.
//
//   swift tools/vision_pose.swift <frames_dir> <out.json> [pad]
//
// pad (default 0): a border, as a fraction of the frame height, added on every
// side before Vision runs. Joints are mapped back to the unpadded frame. Vision
// loses a bench presser at lockout, filmed from the feet and cut off at the
// frame edges, and a 0.25 border recovers most of those frames (app LEDGER D25).
//
// Coordinates are pixels, origin top-left (Vision's are normalized with origin
// bottom-left). Joints Vision does not report come back as [0, 0, 0].
// model_s times only the request, not JPEG decode, to match the YOLO sweep's
// model_s.

import Foundation
import Vision
import CoreGraphics
import ImageIO

let cocoOrder: [VNHumanBodyPoseObservation.JointName] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow,
    .leftWrist, .rightWrist, .leftHip, .rightHip,
    .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
]

let args = CommandLine.arguments
guard args.count == 3 || args.count == 4 else {
    FileHandle.standardError.write("usage: vision_pose.swift <frames_dir> <out.json> [pad]\n".data(using: .utf8)!)
    exit(2)
}
let padFraction = args.count == 4 ? Double(args[3]) ?? 0 : 0
let framesDir = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])

let files = try FileManager.default.contentsOfDirectory(atPath: framesDir.path)
    .filter { $0.hasSuffix(".jpg") }
    .sorted()

var frames: [[String: Any]] = []
var modelS = 0.0
var frameSize: [Int] = [0, 0]

for name in files {
    let url = framesDir.appendingPathComponent(name)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        frames.append(["frame": name, "people": []])
        continue
    }
    let w = Double(img.width), h = Double(img.height)
    frameSize = [img.width, img.height]
    let pad = (padFraction * h).rounded()
    let pw = w + 2 * pad, ph = h + 2 * pad
    var input = img
    if pad > 0 {
        let ctx = CGContext(data: nil, width: Int(pw), height: Int(ph), bitsPerComponent: 8,
                            bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: pw, height: ph))
        ctx.draw(img, in: CGRect(x: pad, y: pad, width: w, height: h))
        input = ctx.makeImage()!
    }

    let request = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(cgImage: input, options: [:])
    let t0 = DispatchTime.now().uptimeNanoseconds
    try handler.perform([request])
    modelS += Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9

    var people: [[String: Any]] = []
    for obs in request.results ?? [] {
        guard let pts = try? obs.recognizedPoints(.all) else { continue }
        var kp: [[Double]] = []
        var xs: [Double] = [], ys: [Double] = []
        for j in cocoOrder {
            if let p = pts[j], p.confidence > 0 {
                let x = Double(p.location.x) * pw - pad
                let y = (1 - Double(p.location.y)) * ph - pad
                kp.append([x, y, Double(p.confidence)])
                xs.append(x); ys.append(y)
            } else {
                kp.append([0, 0, 0])
            }
        }
        let box = xs.isEmpty ? [0.0, 0, 0, 0] : [xs.min()!, ys.min()!, xs.max()!, ys.max()!]
        people.append(["kp": kp, "box": box, "score": Double(obs.confidence)])
    }
    frames.append(["frame": name, "people": people])
}

let out: [String: Any] = [
    "backend": "vision",
    "frame_size": frameSize,
    "model_s": modelS,
    "frames": frames,
]
let data = try JSONSerialization.data(withJSONObject: out)
try data.write(to: outURL)
print("vision: \(files.count) frames, \(String(format: "%.2f", 1000 * modelS / Double(max(files.count, 1)))) ms/frame")
