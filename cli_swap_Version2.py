#!/usr/bin/env python3
"""
Simple desktop CLI face swap using OpenCV + MediaPipe.
Install: pip install opencv-python mediapipe numpy
Usage:
  python swap.py --source s.jpg --target t.jpg --out out.jpg
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse

mp_face = mp.solutions.face_detection

def detect_face_keypoints(img):
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = img.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        wbox = int(bbox.width * w)
        hbox = int(bbox.height * h)
        return (x, y, wbox, hbox)

def simple_swap(src, tgt):
    s_face = detect_face_keypoints(src)
    t_face = detect_face_keypoints(tgt)
    if s_face is None or t_face is None:
        print("No face detected in one of the images")
        return None
    sx, sy, sw, sh = s_face
    tx, ty, tw, th = t_face
    src_crop = src[sy:sy+sh, sx:sx+sw]
    src_resized = cv2.resize(src_crop, (tw, th))
    mask = 255 * np.ones(src_resized.shape[:2], src_resized.dtype)
    center = (tx + tw//2, ty + th//2)
    output = cv2.seamlessClone(src_resized, tgt, mask, center, cv2.NORMAL_CLONE)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = cv2.imread(args.source)
    tgt = cv2.imread(args.target)
    result = simple_swap(src, tgt)
    if result is not None:
        cv2.imwrite(args.out, result)
        print("Wrote", args.out)

if __name__ == "__main__":
    main()