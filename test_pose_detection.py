#!/usr/bin/env python3
"""
Test pose detection on user's video to validate improvements.
"""

import cv2
import sys
from pathlib import Path
from pose_detector import PoseDetector


def test_video(video_path="5542.mov", conf_threshold=0.25, keypoint_conf_threshold=0.3):
    """
    Test pose detection on a video file.

    Args:
        video_path: Path to video file
        conf_threshold: Detection confidence threshold
        keypoint_conf_threshold: Keypoint confidence threshold
    """
    # Check if video exists
    video_file = Path(video_path)
    if not video_file.exists():
        print(f"Error: Video file not found: {video_path}")
        print("Please provide the correct path to 5542.mov")
        return

    print(f"Testing pose detection on: {video_path}")
    print(f"Detection threshold: {conf_threshold}")
    print(f"Keypoint threshold: {keypoint_conf_threshold}")
    print("-" * 60)

    # Initialize detector
    try:
        detector = PoseDetector("yolov8m-pose.pt")
        if not detector.available:
            print("Error: Pose detector not available")
            return
        print(f"✓ Loaded YOLOv8m-pose model")
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    # Open video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Error: Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video opened: {total_frames} frames @ {fps:.1f} FPS")
    print("-" * 60)

    # Process video
    detected_frames = 0
    confidence_scores = []
    wrist_detections = 0
    keypoint_stats = {i: {'detected': 0, 'avg_conf': []} for i in range(17)}

    keypoint_names = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"
    ]

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Detect pose
        pose = detector.detect(frame, conf_threshold=conf_threshold, keypoint_conf_threshold=keypoint_conf_threshold)

        if pose is not None:
            detected_frames += 1

            # Calculate average keypoint confidence
            valid_keypoints = pose.keypoints[pose.keypoints[:, 2] > keypoint_conf_threshold]
            if len(valid_keypoints) > 0:
                avg_conf = valid_keypoints[:, 2].mean()
                confidence_scores.append(avg_conf)

            # Track wrist detections (keypoints 9 and 10)
            left_wrist = pose.keypoints[9]
            right_wrist = pose.keypoints[10]
            if left_wrist[2] > keypoint_conf_threshold or right_wrist[2] > keypoint_conf_threshold:
                wrist_detections += 1

            # Track per-keypoint statistics
            for i in range(17):
                kp = pose.keypoints[i]
                if kp[2] > keypoint_conf_threshold:
                    keypoint_stats[i]['detected'] += 1
                    keypoint_stats[i]['avg_conf'].append(kp[2])

        # Progress update every 100 frames
        if frame_num % 100 == 0:
            detection_rate = detected_frames / frame_num
            print(f"Processed {frame_num}/{total_frames} frames | Detection rate: {detection_rate:.1%}")

    cap.release()

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total frames: {total_frames}")
    print(f"Detected frames: {detected_frames} ({detected_frames/total_frames:.1%})")

    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        min_confidence = min(confidence_scores)
        max_confidence = max(confidence_scores)
        print(f"Average keypoint confidence: {avg_confidence:.2f}")
        print(f"Min/Max confidence: {min_confidence:.2f} / {max_confidence:.2f}")

    print(f"Wrist detections: {wrist_detections} ({wrist_detections/total_frames:.1%})")

    # Per-keypoint statistics
    print("\n" + "-" * 60)
    print("PER-KEYPOINT DETECTION RATES")
    print("-" * 60)

    for i in range(17):
        stats = keypoint_stats[i]
        detection_rate = stats['detected'] / total_frames
        avg_conf = sum(stats['avg_conf']) / len(stats['avg_conf']) if stats['avg_conf'] else 0.0

        # Highlight important keypoints for table tennis
        important = i in [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
        marker = "★" if important else " "

        print(f"{marker} {keypoint_names[i]:12s}: {detection_rate:5.1%} (avg conf: {avg_conf:.2f})")

    # Success criteria evaluation
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 60)

    detection_rate = detected_frames / total_frames
    wrist_rate = wrist_detections / total_frames
    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    criteria = [
        ("Detection rate > 70%", detection_rate > 0.70, f"{detection_rate:.1%}"),
        ("Avg keypoint confidence > 0.5", avg_conf > 0.5, f"{avg_conf:.2f}"),
        ("Wrist detection > 60%", wrist_rate > 0.60, f"{wrist_rate:.1%}"),
    ]

    for criterion, passed, value in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {criterion} [{value}]")

    all_passed = all(passed for _, passed, _ in criteria)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CRITERIA PASSED - Pose detection is working well!")
    else:
        print("✗ SOME CRITERIA FAILED - Further tuning may be needed")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "5542.mov"

    test_video(video_path)
