import numpy as np
import cv2
import math
import torch
import matplotlib.colors

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6, scale_for_xinsr=False, stickwidth=4):
    """
    Draw wholebody keypoints in DWPose style (matching SDPose_gradio.py)
    Expected keypoint format:
    - Body: 0-17 (18 keypoints in OpenPose format, neck at index 1)
    - Foot: 18-23 (6 keypoints)
    - Face: 24-91 (68 landmarks)
    - Right hand: 92-112 (21 keypoints)
    - Left hand: 113-133 (21 keypoints)
    
    Args:
        stickwidth: 骨骼线条粗细，默认为4
    """
    H, W, C = canvas.shape
    
    # Body connections (1-indexed, will convert to 0-indexed)
    body_limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]
    ]
    
    # Hand connections (same for both hands)
    hand_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],      # index
        [0, 9], [9, 10], [10, 11], [11, 12], # middle
        [0, 13], [13, 14], [14, 15], [15, 16], # ring
        [0, 17], [17, 18], [18, 19], [19, 20], # pinky
    ]
    
    # Colors in BGR format - from RED to BLUE, then to purple/magenta
    # After BGR->RGB conversion: limb 0 = red, limb 12 (neck-nose) = blue
    colors = [
        [0, 0, 255],     # 0: red
        [0, 85, 255],    # 1: orange-red
        [0, 170, 255],   # 2: orange
        [0, 255, 255],   # 3: yellow
        [0, 255, 170],   # 4: yellow-green
        [0, 255, 85],    # 5: green-yellow
        [0, 255, 0],     # 6: green
        [85, 255, 0],    # 7: cyan-green
        [170, 255, 0],   # 8: cyan
        [255, 255, 0],   # 9: light cyan
        [255, 170, 0],   # 10: blue-cyan
        [255, 85, 0],    # 11: blue-green
        [255, 0, 0],     # 12: blue (neck-nose)
        [255, 0, 85],    # 13: blue-purple
        [255, 0, 170],   # 14: purple
        [255, 0, 255],   # 15: magenta
        [170, 0, 255],   # 16: purple-red
        [85, 0, 255],    # 17: red-purple
    ]
    
    # --- Draw body limbs (0-17) ---
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18:
                continue
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]])
            X = np.array([keypoints[idx1][1], keypoints[idx2][1]])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < 1:
                continue
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
    
    # --- Draw body keypoints (0-17) ---
    if len(keypoints) >= 18:
        for i in range(18):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)
    
    # --- Draw foot keypoints (18-23) - NO CONNECTIONS, only dots ---
    # Use colors[i % len(colors)] matching reference implementation
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, colors[i % len(colors)], thickness=-1)

    # --- Draw right hand (92-112) ---
    if len(keypoints) >= 113:
        eps = 0.01
        # Draw hand edges with HSV rainbow colors
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw right hand keypoints - BLUE color (255, 0, 0) in BGR
        # After BGR->RGB conversion in batch_draw_pose, this becomes RGB blue (0, 0, 255)
        for i in range(92, 113):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (255, 0, 0), thickness=-1)
    
    # --- Draw left hand (113-133) ---
    if len(keypoints) >= 134:
        eps = 0.01
        # Draw hand edges with HSV rainbow colors
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold):
                continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1])
            x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                    color = matplotlib.colors.hsv_to_rgb([ie / float(len(hand_edges)), 1.0, 1.0]) * 255
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw left hand keypoints - BLUE color (255, 0, 0) in BGR
        # After BGR->RGB conversion in batch_draw_pose, this becomes RGB blue (0, 0, 255)
        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 4, (255, 0, 0), thickness=-1)
    
    # --- Draw face keypoints (24-91) - white dots only, no lines ---
    if len(keypoints) >= 92:
        eps = 0.01
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold:
                continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > eps and y > eps and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas

def draw_pose_frame(pose_data, width, height, threshold=0.4, stickwidth=4, y_offset=0):
    """
    Draws a single frame of pose data using OpenPose style.
    Handles OpenPose 18 format from SDPose output.
    
    Args:
        stickwidth: 骨骼线条粗细，默认为4
        y_offset: Y轴偏移量（用于底部对齐），默认为0
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
        # 0-17: Body 18 (OpenPose 18 format)
        # 18-23: Feet (6)
        # 24-91: Face (68) - Not supported yet
        # 92-112: Right Hand (21)
        # 113-133: Left Hand (21)
        # Total: 134
        
        kpts_full = np.zeros((134, 2), dtype=np.float32)
        scores_full = np.zeros((134,), dtype=np.float32)
        
        # 1. Parse Body (OpenPose 18 format)
        body_kpts = person.get("pose_keypoints_2d", [])
        if body_kpts:
            body_arr = np.array(body_kpts).reshape(-1, 3)
            
            # Check/Fix Normalization
            if body_arr.size > 0:
                max_val = np.max(body_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    # Values are normalized, multiply by canvas dimensions
                    body_arr[:, 0] *= width
                    body_arr[:, 1] *= height
            
            # Input is already in OpenPose 18 format, copy directly
            if len(body_arr) >= 18:
                # OpenPose 18: 0:Nose, 1:Neck, 2:RSho, 3:RElb, 4:RWri, 5:LSho, 6:LElb, 7:LWri, 
                #              8:RHip, 9:RKnee, 10:RAnk, 11:LHip, 12:LKnee, 13:LAnk, 
                #              14:REye, 15:LEye, 16:REar, 17:LEar
                for i in range(18):
                    kpts_full[i] = body_arr[i, :2]
                    scores_full[i] = body_arr[i, 2]
        
        # 2. Parse Feet (from separate foot_keypoints_2d field)
        foot_kpts = person.get("foot_keypoints_2d", [])
        if foot_kpts:
            foot_arr = np.array(foot_kpts).reshape(-1, 3)
            
            # Check/Fix Normalization
            if foot_arr.size > 0:
                max_val = np.max(foot_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    foot_arr[:, 0] *= width
                    foot_arr[:, 1] *= height
            
            # SDPose foot format: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
            # Map to WholeBody indices 18-23
            if len(foot_arr) >= 6:
                # Right foot (indices 18-20 in our output)
                kpts_full[18] = foot_arr[0, :2]; scores_full[18] = foot_arr[0, 2]  # RBigToe
                kpts_full[19] = foot_arr[1, :2]; scores_full[19] = foot_arr[1, 2]  # RSmallToe
                kpts_full[20] = foot_arr[2, :2]; scores_full[20] = foot_arr[2, 2]  # RHeel
                
                # Left foot (indices 21-23 in our output)
                kpts_full[21] = foot_arr[3, :2]; scores_full[21] = foot_arr[3, 2]  # LBigToe
                kpts_full[22] = foot_arr[4, :2]; scores_full[22] = foot_arr[4, 2]  # LSmallToe
                kpts_full[23] = foot_arr[5, :2]; scores_full[23] = foot_arr[5, 2]  # LHeel
        
        # 3. Parse Hands
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
        
        # 4. Parse Face (if available)
        face_kpts = person.get("face_keypoints_2d", [])
        if face_kpts:
            face_arr = np.array(face_kpts).reshape(-1, 3)
            
            # Check/Fix Normalization
            if face_arr.size > 0:
                max_val = np.max(face_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    face_arr[:, 0] *= width
                    face_arr[:, 1] *= height
            
            # Map 68 face landmarks to indices 24-91
            if len(face_arr) >= 68:
                kpts_full[24:92] = face_arr[:68, :2]
                scores_full[24:92] = face_arr[:68, 2]
        
        # Apply Y offset for bottom alignment (only to valid keypoints)
        if y_offset != 0:
            # Only offset keypoints that have valid scores
            for i in range(len(kpts_full)):
                if scores_full[i] > 0:
                    kpts_full[i, 1] += y_offset
        
        # Draw
        draw_wholebody_keypoints_openpose_style(canvas, kpts_full, scores_full, threshold=threshold, stickwidth=stickwidth)
            
    return canvas

def get_keypoints_from_pose(pose_data, width, height, threshold=0.0):
    """
    从pose_data中提取所有有效关键点的坐标，用于计算边界
    
    Args:
        pose_data: 姿态数据字典
        width: 画布宽度
        height: 画布高度
        threshold: 置信度阈值，只返回置信度高于此值的点
    
    返回所有有效点的列表 [(x, y, score), ...]
    """
    points = []
    
    if "people" not in pose_data:
        return points
    
    for person in pose_data["people"]:
        # Body keypoints
        body_kpts = person.get("pose_keypoints_2d", [])
        if body_kpts:
            body_arr = np.array(body_kpts).reshape(-1, 3).copy()
            if body_arr.size > 0:
                max_val = np.max(body_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    body_arr[:, 0] *= width
                    body_arr[:, 1] *= height
            for pt in body_arr:
                if pt[2] >= threshold:  # 使用 threshold 判断有效点
                    points.append((pt[0], pt[1], pt[2]))
        
        # Foot keypoints
        foot_kpts = person.get("foot_keypoints_2d", [])
        if foot_kpts:
            foot_arr = np.array(foot_kpts).reshape(-1, 3).copy()
            if foot_arr.size > 0:
                max_val = np.max(foot_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    foot_arr[:, 0] *= width
                    foot_arr[:, 1] *= height
            for pt in foot_arr:
                if pt[2] >= threshold:
                    points.append((pt[0], pt[1], pt[2]))
        
        # Right hand
        rh_kpts = person.get("hand_right_keypoints_2d", [])
        if rh_kpts:
            rh_arr = np.array(rh_kpts).reshape(-1, 3).copy()
            if rh_arr.size > 0:
                max_val = np.max(rh_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    rh_arr[:, 0] *= width
                    rh_arr[:, 1] *= height
            for pt in rh_arr:
                if pt[2] >= threshold:
                    points.append((pt[0], pt[1], pt[2]))
        
        # Left hand
        lh_kpts = person.get("hand_left_keypoints_2d", [])
        if lh_kpts:
            lh_arr = np.array(lh_kpts).reshape(-1, 3).copy()
            if lh_arr.size > 0:
                max_val = np.max(lh_arr[:, :2])
                if max_val <= 1.0 and max_val > 0:
                    lh_arr[:, 0] *= width
                    lh_arr[:, 1] *= height
            for pt in lh_arr:
                if pt[2] >= threshold:
                    points.append((pt[0], pt[1], pt[2]))
        
        # Face keypoints - 不参与底部对齐计算（通常脸不是最低点）
        # 如果需要，可以取消下面的注释
        # face_kpts = person.get("face_keypoints_2d", [])
        # if face_kpts:
        #     face_arr = np.array(face_kpts).reshape(-1, 3).copy()
        #     if face_arr.size > 0:
        #         max_val = np.max(face_arr[:, :2])
        #         if max_val <= 1.0 and max_val > 0:
        #             face_arr[:, 0] *= width
        #             face_arr[:, 1] *= height
        #     for pt in face_arr:
        #         if pt[2] >= threshold:
        #             points.append((pt[0], pt[1], pt[2]))
    
    return points


def batch_draw_pose(pose_list, threshold=0.4, stickwidth=4, align_to_bottom=False):
    """
    Draws a list of pose frames and returns a batch tensor.
    
    Args:
        pose_list: 姿态数据列表
        threshold: 置信度阈值
        stickwidth: 骨骼线条粗细
        align_to_bottom: 是否将所有帧的最低点对齐到画布底部
    """
    if isinstance(pose_list, dict): # Single frame
        pose_list = [pose_list]
        
    if not pose_list:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    
    # Get dimensions from first frame
    width = pose_list[0].get("canvas_width", 512)
    height = pose_list[0].get("canvas_height", 512)
    
    # Calculate Y offset for bottom alignment
    y_offset = 0
    if align_to_bottom:
        # Find the lowest point (maximum Y value) across ALL frames
        global_max_y = -float('inf')
        total_frames = len(pose_list)
        frames_with_points = 0
        
        for frame_idx, pose in enumerate(pose_list):
            w = pose.get("canvas_width", width)
            h = pose.get("canvas_height", height)
            # 使用相同的 threshold 来获取有效关键点
            points = get_keypoints_from_pose(pose, w, h, threshold=threshold)
            if points:
                frames_with_points += 1
                # Find maximum Y (lowest point) in this frame
                frame_max_y = max(pt[1] for pt in points)
                global_max_y = max(global_max_y, frame_max_y)
        
        # Calculate offset to align bottom point to canvas bottom
        if global_max_y > -float('inf') and global_max_y > 0:
            y_offset = height - global_max_y
    
    frames = []
    for pose in pose_list:
        # Use per-frame dimension if available, else fallback
        w = pose.get("canvas_width", width)
        h = pose.get("canvas_height", height)
        
        canvas = draw_pose_frame(pose, w, h, threshold, stickwidth=stickwidth, y_offset=y_offset)
        
        # Convert BGR to RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1
        canvas = canvas.astype(np.float32) / 255.0
        frames.append(canvas)
        
    # Stack frames: (B, H, W, C)
    batch_tensor = torch.from_numpy(np.stack(frames, axis=0))
    
    return batch_tensor
