import numpy as np
import cv2
import math
import torch
import matplotlib.colors

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6, scale_for_xinsr=False):
    H, W, C = canvas.shape

    # --- 计算基础粗细 (使用固定值4作为基准) ---
    base_stickwidth = 4 
    stickwidth = base_stickwidth # 默认值

    # --- 应用 xinsr 缩放 ---
    if scale_for_xinsr:
        target_max_side = max(H, W) # 使用图像最大边长
        # 借用 OpenPose Editor 的公式
        xinsr_stick_scale = 1 if target_max_side < 500 else min(2 + (target_max_side // 1000), 7)
        stickwidth = base_stickwidth * xinsr_stick_scale
        # print(f"SDPose Node: Applying Xinsr scale ({xinsr_stick_scale:.1f}) to stickwidth. New width: {stickwidth}") # Debug print

    body_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18: continue
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]]); X = np.array([keypoints[idx1][1], keypoints[idx2][1]]); mX = np.mean(X); mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < 1: continue
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])

        for i in range(18):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)

    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)

    if len(keypoints) >= 113:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

        for i in range(92, 113):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    if len(keypoints) >= 134:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)

        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    if len(keypoints) >= 92:
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas

def draw_pose_frame(pose_data, width, height, threshold=0.4):
    """
    Draws a single frame of pose data using OpenPose style.
    Converts COCO 17 + Hands to WholeBody format (134 points).
    """
    # Initialize black canvas
    if width <= 0: width = 512
    if height <= 0: height = 512
    
    width = int(width)
    height = int(height)
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    if "people" not in pose_data:
        return canvas
        
    for person in pose_data["people"]:
        # --- Prepare WholeBody Arrays ---
        # Format:
        # 0-17: Body 18 (COCO 18)
        # 18-23: Feet (6)
        # 24-91: Face (68) - Not supported yet
        # 92-112: Right Hand (21)
        # 113-133: Left Hand (21)
        # Total: 134
        
        kpts_full = np.zeros((134, 2), dtype=np.float32)
        scores_full = np.zeros((134,), dtype=np.float32)
        
        # 1. Parse Body (COCO 17 -> Body 18)
        body_kpts = person.get("pose_keypoints_2d", [])
        if body_kpts:
            body_arr = np.array(body_kpts).reshape(-1, 3)
            
            # Check/Fix Normalization
            if body_arr.size > 0:
                max_val = np.max(body_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    body_arr[:, 0] *= width
                    body_arr[:, 1] *= height
            
            if len(body_arr) >= 17:
                # Map COCO 17 to Body 18
                # 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar
                # 5:LSho, 6:RSho, 7:LElb, 8:RElb, 9:LWri, 10:RWri
                # 11:LHip, 12:RHip, 13:LKnee, 14:RKnee, 15:LAnk, 16:RAnk
                
                # Target: 0:Nose, 1:Neck, 2:RSho, 3:RElb, 4:RWri, 5:LSho, 6:LElb, 7:LWri
                # 8:RHip, 9:RKnee, 10:RAnk, 11:LHip, 12:LKnee, 13:LAnk
                # 14:REye, 15:LEye, 16:REar, 17:LEar
                
                # Nose
                kpts_full[0] = body_arr[0, :2]; scores_full[0] = body_arr[0, 2]
                
                # Neck = Avg(LSho, RSho)
                if body_arr[5, 2] > 0 and body_arr[6, 2] > 0:
                    kpts_full[1] = (body_arr[5, :2] + body_arr[6, :2]) / 2
                    scores_full[1] = min(body_arr[5, 2], body_arr[6, 2])
                
                # Right Arm
                kpts_full[2] = body_arr[6, :2]; scores_full[2] = body_arr[6, 2]
                kpts_full[3] = body_arr[8, :2]; scores_full[3] = body_arr[8, 2]
                kpts_full[4] = body_arr[10, :2]; scores_full[4] = body_arr[10, 2]
                
                # Left Arm
                kpts_full[5] = body_arr[5, :2]; scores_full[5] = body_arr[5, 2]
                kpts_full[6] = body_arr[7, :2]; scores_full[6] = body_arr[7, 2]
                kpts_full[7] = body_arr[9, :2]; scores_full[7] = body_arr[9, 2]
                
                # Right Leg
                kpts_full[8] = body_arr[12, :2]; scores_full[8] = body_arr[12, 2]
                kpts_full[9] = body_arr[14, :2]; scores_full[9] = body_arr[14, 2]
                kpts_full[10] = body_arr[16, :2]; scores_full[10] = body_arr[16, 2]
                
                # Left Leg
                kpts_full[11] = body_arr[11, :2]; scores_full[11] = body_arr[11, 2]
                kpts_full[12] = body_arr[13, :2]; scores_full[12] = body_arr[13, 2]
                kpts_full[13] = body_arr[15, :2]; scores_full[13] = body_arr[15, 2]
                
                # Eyes/Ears
                kpts_full[14] = body_arr[2, :2]; scores_full[14] = body_arr[2, 2] # REye
                kpts_full[15] = body_arr[1, :2]; scores_full[15] = body_arr[1, 2] # LEye
                kpts_full[16] = body_arr[4, :2]; scores_full[16] = body_arr[4, 2] # REar
                kpts_full[17] = body_arr[3, :2]; scores_full[17] = body_arr[3, 2] # LEar

            # Feet (Indices 17-22 in COCO 17/WholeBody format are sometimes appended)
            # Our Retarget node might append feet (indices 17,18,19,20,21,22) to the body array.
            # COCO WholeBody Feet indices: 18:LBigToe, 19:LSmallToe, 20:LHeel, 21:RBigToe, 22:RSmallToe, 23:RHeel
            # If input has > 17 points, map them.
            if len(body_arr) >= 23:
                 # COCO17 with Feet:
                 # 17: LBigToe, 18: LSmallToe, 19: LHeel
                 # 20: RBigToe, 21: RSmallToe, 22: RHeel
                 
                 # Map to WholeBody:
                 # 18: LBigToe, 19: LSmallToe, 20: LHeel
                 # 21: RBigToe, 22: RSmallToe, 23: RHeel
                 
                 # Left Foot
                 kpts_full[18] = body_arr[17, :2]; scores_full[18] = body_arr[17, 2]
                 kpts_full[19] = body_arr[18, :2]; scores_full[19] = body_arr[18, 2]
                 kpts_full[20] = body_arr[19, :2]; scores_full[20] = body_arr[19, 2]
                 
                 # Right Foot
                 kpts_full[21] = body_arr[20, :2]; scores_full[21] = body_arr[20, 2]
                 kpts_full[22] = body_arr[21, :2]; scores_full[22] = body_arr[21, 2]
                 kpts_full[23] = body_arr[22, :2]; scores_full[23] = body_arr[22, 2]

        # 2. Hands
        # Right Hand
        rh_kpts = person.get("hand_right_keypoints_2d", [])
        if rh_kpts:
            rh_arr = np.array(rh_kpts).reshape(-1, 3)
             # Check/Fix Normalization
            if rh_arr.size > 0:
                max_val = np.max(rh_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    rh_arr[:, 0] *= width
                    rh_arr[:, 1] *= height
                    
            if len(rh_arr) >= 21:
                kpts_full[92:113] = rh_arr[:21, :2]
                scores_full[92:113] = rh_arr[:21, 2]
                
        # Left Hand
        lh_kpts = person.get("hand_left_keypoints_2d", [])
        if lh_kpts:
            lh_arr = np.array(lh_kpts).reshape(-1, 3)
             # Check/Fix Normalization
            if lh_arr.size > 0:
                max_val = np.max(lh_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    lh_arr[:, 0] *= width
                    lh_arr[:, 1] *= height
                    
            if len(lh_arr) >= 21:
                kpts_full[113:134] = lh_arr[:21, :2]
                scores_full[113:134] = lh_arr[:21, 2]
        
        # Draw
        draw_wholebody_keypoints_openpose_style(canvas, kpts_full, scores_full, threshold=threshold)
            
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
