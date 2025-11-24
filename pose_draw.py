import numpy as np
import cv2
import math
import torch

# OpenPose Keypoint Colors and Pairs

# Body 25 / COCO 18?
# We focus on COCO 18 (17 + Neck) or Body 25.
# Common OpenPose Body 18:
# 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist
# 8:RHip, 9:RKnee, 10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:REye, 15:LEye, 16:REar, 17:LEar
# 18: Bkg?
# 
# But our data is COCO 17 (0:Nose, ..., 5:LShoulder, 6:RShoulder ...).
# Note: retarget_pose.py uses:
# 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
# 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow ...
# This matches COCO output from many OpenPose implementations.

# COCO 17 Pairs
BODY_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), # Face
    (5, 6), # Shoulders
    (5, 7), (7, 9), # Left Arm
    (6, 8), (8, 10), # Right Arm
    (11, 12), # Hips
    (11, 13), (13, 15), # Left Leg
    (12, 14), (14, 16), # Right Leg
    (5, 11), (6, 12) # Torso
]

# Colors (BGR)
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85)
]

# Hand Pairs (21 points)
# 0: Wrist
# Thumb: 1,2,3,4
# Index: 5,6,7,8
# Middle: 9,10,11,12
# Ring: 13,14,15,16
# Pinky: 17,18,19,20
HAND_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def draw_body(canvas, keypoints, threshold=0.4):
    """
    Draws COCO 17 body keypoints.
    """
    H, W, C = canvas.shape
    
    # Check if keypoints are normalized (0-1) and denormalize if so
    if keypoints.size > 0:
        max_val = np.max(keypoints[:, :2])
        if max_val <= 1.0 and max_val > 0:
            keypoints = keypoints.copy()
            keypoints[:, 0] *= W
            keypoints[:, 1] *= H

    # Draw Lines
    for i, (p1_idx, p2_idx) in enumerate(BODY_PAIRS):
        if p1_idx >= len(keypoints) or p2_idx >= len(keypoints):
            continue
            
        p1 = keypoints[p1_idx]
        p2 = keypoints[p2_idx]
        
        if p1[2] < threshold or p2[2] < threshold:
            continue
            
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        color = COLORS[i % len(COLORS)]
        cv2.line(canvas, (x1, y1), (x2, y2), color, 3)

    # Draw Points
    for i, p in enumerate(keypoints):
        if i >= 18: break # Limit to 18 points for body
        if p[2] < threshold: continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(canvas, (x, y), 4, COLORS[i % len(COLORS)], -1)

    # Draw Feet (17-22 if present)
    # 17:LBigToe, 18:LSmallToe, 19:LHeel
    # 20:RBigToe, 21:RSmallToe, 22:RHeel
    # Connect Ankle(15) to Heel(19) and BigToe(17)
    # Connect Ankle(16) to Heel(22) and BigToe(20)
    # Connect Heel to Toe?
    if len(keypoints) > 22:
        feet_pairs = [
            (15, 19), (19, 17), (17, 18), # Left Foot
            (16, 22), (22, 20), (20, 21)  # Right Foot
        ]
        for p1_idx, p2_idx in feet_pairs:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
                if p1[2] >= threshold and p2[2] >= threshold:
                    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
                    
        for i in range(17, 23):
            if i < len(keypoints) and keypoints[i][2] >= threshold:
                cv2.circle(canvas, (int(keypoints[i][0]), int(keypoints[i][1])), 3, (0, 255, 255), -1)

def draw_hand(canvas, keypoints, threshold=0.4):
    """
    Draws Hand Keypoints.
    """
    if keypoints is None or len(keypoints) < 21:
        return

    H, W, C = canvas.shape
    
    # Check if keypoints are normalized (0-1) and denormalize if so
    if keypoints.size > 0:
        max_val = np.max(keypoints[:, :2])
        if max_val <= 1.0 and max_val > 0:
            keypoints = keypoints.copy()
            keypoints[:, 0] *= W
            keypoints[:, 1] *= H

    # Draw Lines
    for p1_idx, p2_idx in HAND_PAIRS:
        p1 = keypoints[p1_idx]
        p2 = keypoints[p2_idx]
        
        if p1[2] < threshold or p2[2] < threshold:
            continue
            
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red color for hands usually? Or multi-colored.

    # Draw Points
    for p in keypoints:
        if p[2] < threshold: continue
        x, y = int(p[0]), int(p[1])
        cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)

def draw_pose_frame(pose_data, width, height, threshold=0.4):
    """
    Draws a single frame of pose data.
    """
    # Initialize black canvas
    # Ensure dimensions are valid
    if width <= 0: width = 512
    if height <= 0: height = 512
    
    width = int(width)
    height = int(height)
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    if "people" not in pose_data:
        return canvas
        
    for person in pose_data["people"]:
        # Draw Body
        body_kpts = person.get("pose_keypoints_2d", [])
        if body_kpts:
            body_arr = np.array(body_kpts).reshape(-1, 3)
            draw_body(canvas, body_arr, threshold)
            
        # Draw Left Hand
        lh_kpts = person.get("hand_left_keypoints_2d", [])
        if lh_kpts:
            lh_arr = np.array(lh_kpts).reshape(-1, 3)
            draw_hand(canvas, lh_arr, threshold)
            
        # Draw Right Hand
        rh_kpts = person.get("hand_right_keypoints_2d", [])
        if rh_kpts:
            rh_arr = np.array(rh_kpts).reshape(-1, 3)
            draw_hand(canvas, rh_arr, threshold)
            
    return canvas

def batch_draw_pose(pose_list, threshold=0.4):
    """
    Draws a list of pose frames and returns a batch tensor.
    """
    if isinstance(pose_list, dict): # Single frame
        pose_list = [pose_list]
        
    if not pose_list:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    
    # Get dimensions from first frame
    width = pose_list[0].get("canvas_width", 512)
    height = pose_list[0].get("canvas_height", 512)
    
    frames = []
    for pose in pose_list:
        # Use per-frame dimension if available, else fallback
        w = pose.get("canvas_width", width)
        h = pose.get("canvas_height", height)
        
        canvas = draw_pose_frame(pose, w, h, threshold)
        
        # Convert BGR to RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        canvas = canvas.astype(np.float32) / 255.0
        frames.append(canvas)
        
    # Stack frames: (B, H, W, C)
    batch_tensor = torch.from_numpy(np.stack(frames, axis=0))
    
    return batch_tensor

