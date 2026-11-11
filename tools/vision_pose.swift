// vision_pose.swift — run Apple Vision body pose over a directory of frames.
//
// Part of the pose-backend comparison (tools/compare_pose_backends.py). Runs
// VNDetectHumanBodyPoseRequest on every .jpg in a frames directory and writes
// one JSON file with each person's joints remapped to COCO-17 order, so the
// existing wrist-midpoint tracker (COCO 9 + 10) can consume it unchanged.
//
//   swift tools/vision_pose.swift <frames_dir> <out.json>
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
guard args.count == 3 else {
    FileHandle.standardError.write("usage: vision_pose.swift <frames_dir> <out.json>\n".data(using: .utf8)!)
    exit(2)
}
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

    let request = VNDetectHumanBodyPoseRequest()
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
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
                let x = Double(p.location.x) * w
                let y = (1 - Double(p.location.y)) * h
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
