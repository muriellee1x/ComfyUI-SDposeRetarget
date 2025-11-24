import numpy as np
import copy
import json
import math

# --- Constants and Topology ---

# COCO 17 Body Keypoints
# 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
# 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow
# 9: LWrist, 10: RWrist, 11: LHip, 12: RHip
# 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle

# Feet (Assuming appended if present, or separate)
# COCO WholeBody Feet: 17: LBigToe, 18: LSmallToe, 19: LHeel, 20: RBigToe, 21: RSmallToe, 22: RHeel
# (Note: indices might vary, but we will detect based on length)

# Hierarchy for Body Retargeting (Parent -> Child)
# Root is virtual MidHip
BODY_HIERARCHY = {
    # Torso / Head
    # We treat MidHip as root.
    # MidHip -> Neck (virtual) -> Nose (0)
    # Nose -> Eyes, Ears? 
    # Let's use a simplified tree for retargeting to avoid disconnection
    
    # Spine/Head path
    # MidHip -> Spine -> Neck -> Nose
    # We don't have Spine/Neck in COCO 17 explicitly, usually average of shoulders/hips.
    
    # Limbs
    5: 7, 7: 9,        # Left Arm: LShoulder -> LElbow -> LWrist
    6: 8, 8: 10,       # Right Arm: RShoulder -> RElbow -> RWrist
    11: 13, 13: 15,    # Left Leg: LHip -> LKnee -> LAnkle
    12: 14, 14: 16,    # Right Leg: RHip -> RKnee -> RAnkle
    
    # Feet (if present, indices 17-22)
    15: [17, 18, 19], # LAnkle -> LToes/Heel
    16: [20, 21, 22]  # RAnkle -> RToes/Heel
}

# Hand Topology (21 points)
# 0: Wrist
# 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def parse_pose_data(data):
    """
    Parses the input data (list or dict) and extracts the first person's pose.
    Returns a normalized dictionary with numpy arrays.
    """
    if isinstance(data, list):
        if len(data) == 0: return None
        frame_data = data[0]
    else:
        frame_data = data
    
    if "people" in frame_data:
        people = frame_data["people"]
        if len(people) == 0: return None
        person = people[0]
    else:
        # Assume it might be the person dict itself
        person = frame_data

    # Helper to convert list to (N, 3) array
    def to_array(kpts):
        if not kpts: return None
        arr = np.array(kpts).reshape(-1, 3)
        return arr

    body = to_array(person.get("pose_keypoints_2d", []))
    left_hand = to_array(person.get("hand_left_keypoints_2d", []))
    right_hand = to_array(person.get("hand_right_keypoints_2d", []))
    
    # Filter face if needed (we ignore it for retargeting calculations but keep structure if needed? 
    # The prompt says "receive ... not include face ... if receive face ignore")
    # We just don't process face keys.

    canvas_width = frame_data.get("canvas_width", 0)
    canvas_height = frame_data.get("canvas_height", 0)

    # Check for normalized coordinates and denormalize if necessary
    all_kpts = []
    if body is not None: all_kpts.append(body)
    if left_hand is not None: all_kpts.append(left_hand)
    if right_hand is not None: all_kpts.append(right_hand)
    
    is_normalized = False
    if all_kpts:
        # Flatten to check values
        flat_list = [k[:, :2].flatten() for k in all_kpts] # Only check x,y
        flat = np.concatenate(flat_list)
        if len(flat) > 0:
            max_val = np.max(flat)
            # If max value is <= 1.0 and there is at least some data > 0
            if max_val <= 1.0 and max_val > 0:
                 is_normalized = True

    if is_normalized:
        # Determine scale
        # If canvas dimensions are missing or small, default to 512 (common square format)
        # or 1024 if you prefer high res, but 512 is safer for SD.
        scale_w = canvas_width if canvas_width > 1 else 512
        scale_h = canvas_height if canvas_height > 1 else 512
        
        # Update width/height if they were missing/invalid
        if canvas_width <= 1: canvas_width = int(scale_w)
        if canvas_height <= 1: canvas_height = int(scale_h)
        
        # Denormalize
        if body is not None:
            body[:, 0] *= scale_w
            body[:, 1] *= scale_h
        if left_hand is not None:
            left_hand[:, 0] *= scale_w
            left_hand[:, 1] *= scale_h
        if right_hand is not None:
            right_hand[:, 0] *= scale_w
            right_hand[:, 1] *= scale_h

    return {
        "body": body,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height
    }

def get_bone_length(kpts, i1, i2):
    if kpts is None or i1 >= len(kpts) or i2 >= len(kpts):
        return 0.0
    p1 = kpts[i1]
    p2 = kpts[i2]
    if p1[2] < 0.05 or p2[2] < 0.05: # Low confidence
        return 0.0
    return np.linalg.norm(p1[:2] - p2[:2])

def calculate_hand_scale(src_hand, dst_hand):
    """
    Calculates the average scale factor between two hands.
    Handles both 21-point (with wrist) and 20-point (without wrist) hand arrays.
    """
    if src_hand is None or dst_hand is None:
        return 1.0
    
    len_src = len(src_hand)
    len_dst = len(dst_hand)
    
    # If lengths differ significantly, return 1.0 or try to match?
    # Assume same format for ref and target
    
    total_len_src = 0
    total_len_dst = 0
    count = 0
    
    # Determine mapping offset
    # If len == 20, indices 0..19 correspond to standard 1..20.
    # If len == 21, indices 0..20 correspond to standard 0..20.
    
    src_has_wrist = (len_src == 21)
    dst_has_wrist = (len_dst == 21)
    
    for i1, i2 in HAND_BONES:
        # Map standard indices to array indices
        idx1_src, idx2_src = i1, i2
        idx1_dst, idx2_dst = i1, i2
        
        # Adjust for Source
        if not src_has_wrist:
            if i1 == 0: continue # Skip wrist connections if wrist missing
            idx1_src -= 1
            idx2_src -= 1
            
        # Adjust for Target
        if not dst_has_wrist:
            if i1 == 0: continue
            idx1_dst -= 1
            idx2_dst -= 1
            
        if idx1_src < 0 or idx2_src >= len_src or idx1_dst < 0 or idx2_dst >= len_dst:
            continue

        l_src = get_bone_length(src_hand, idx1_src, idx2_src)
        l_dst = get_bone_length(dst_hand, idx1_dst, idx2_dst)
        
        if l_src > 0 and l_dst > 0:
            total_len_src += l_src
            total_len_dst += l_dst
            count += 1
            
    if count < 3 or total_len_src == 0: # Need at least some matches
        return 1.0
        
    return total_len_dst / total_len_src

def retarget_pose_main(ref_pose, target_pose, video_poses, threshold=0.4):
    """
    Main entry point for retargeting.
    """
    # 1. Parse Reference and Target
    # If ref_pose is a list (video), take the first frame
    ref_data = parse_pose_data(ref_pose[0] if isinstance(ref_pose, list) else ref_pose)
    target_data = parse_pose_data(target_pose[0] if isinstance(target_pose, list) else target_pose)
    
    if not ref_data or not target_data:
        print("Error: Missing reference or target pose data.")
        return video_poses # Return original if failed
    
    # 2. Calculate Scaling Factors
    
    # --- Body Scaling ---
    # We calculate scale factors for specific limbs.
    # Map: (Parent, Child) -> ScaleRatio
    limb_scales = {}
    
    # List of bones to compute scale for
    # (ParentIdx, ChildIdx)
    bones_to_scale = [
        (5, 7), (7, 9), # L Arm
        (6, 8), (8, 10), # R Arm
        (11, 13), (13, 15), # L Leg
        (12, 14), (14, 16), # R Leg
        (5, 6), # Shoulders width
        (11, 12), # Hips width
        (5, 11), (6, 12) # Torso length (approx)
    ]
    
    avg_scale_sum = 0
    avg_scale_count = 0
    
    for p, c in bones_to_scale:
        l_ref = get_bone_length(ref_data["body"], p, c)
        l_tgt = get_bone_length(target_data["body"], p, c)
        
        if l_ref > 0 and l_tgt > 0:
            ratio = l_tgt / l_ref
            limb_scales[(p, c)] = ratio
            avg_scale_sum += ratio
            avg_scale_count += 1
        else:
            limb_scales[(p, c)] = None

    global_scale = avg_scale_sum / avg_scale_count if avg_scale_count > 0 else 1.0
    
    # Fill missing scales with global scale
    for k in limb_scales:
        if limb_scales[k] is None:
            limb_scales[k] = global_scale

    # --- Hand Scaling ---
    l_hand_scale = calculate_hand_scale(ref_data["left_hand"], target_data["left_hand"])
    r_hand_scale = calculate_hand_scale(ref_data["right_hand"], target_data["right_hand"])
    
    # If hand scaling failed (e.g. hand missing in target), fallback to global or wrist scale?
    # Using global scale is safer than 1.0
    if l_hand_scale == 1.0 and ref_data["left_hand"] is not None:
        l_hand_scale = global_scale
    if r_hand_scale == 1.0 and ref_data["right_hand"] is not None:
        r_hand_scale = global_scale
        
    # 3. Process Video Frames
    result_frames = []
    
    # Prepare Target Root (MidHip)
    tgt_body = target_data["body"]
    tgt_mid_hip = None
    if tgt_body is not None and len(tgt_body) > 12:
        if tgt_body[11][2] > 0 and tgt_body[12][2] > 0:
             tgt_mid_hip = (tgt_body[11][:2] + tgt_body[12][:2]) / 2
    
    # Iterate over video frames
    frames_list = video_poses if isinstance(video_poses, list) else [video_poses]
    
    for frame_idx, raw_frame in enumerate(frames_list):
        # Deep copy structure to preserve format
        new_frame = copy.deepcopy(raw_frame)
        
        # Parse current frame data
        frame_parsed = parse_pose_data(raw_frame)
        if not frame_parsed:
            result_frames.append(new_frame)
            continue

        src_body = frame_parsed["body"]
        src_l_hand = frame_parsed["left_hand"]
        src_r_hand = frame_parsed["right_hand"]
        
        if src_body is None:
            result_frames.append(new_frame)
            continue

        # --- Retarget Body ---
        # Strategy: Reconstruct body starting from MidHip
        
        # 1. Offset Calculation
        src_mid_hip = None
        if src_body is not None and len(src_body) > 12:
             if src_body[11][2] > 0 and src_body[12][2] > 0:
                src_mid_hip = (src_body[11][:2] + src_body[12][:2]) / 2
        
        offset = np.array([0.0, 0.0])
        if src_mid_hip is not None and tgt_mid_hip is not None:
            offset = tgt_mid_hip - src_mid_hip
        
        # 2. Prepare Transformed Body
        # We work with a copy that has been translated
        translated_body = src_body.copy()
        for i in range(len(translated_body)):
            if translated_body[i][2] > 0:
                translated_body[i][:2] += offset
                
        # Final Body to store results
        final_body = translated_body.copy()
        
        # 3. Apply Scaling
        if src_mid_hip is not None:
             # Hips
            hip_scale = limb_scales.get((11, 12), global_scale)
            center_current = (translated_body[11][:2] + translated_body[12][:2]) / 2
            
            for hip_idx in [11, 12]:
                if translated_body[hip_idx][2] > 0:
                    vec = translated_body[hip_idx][:2] - center_current
                    final_body[hip_idx][:2] = center_current + vec * hip_scale
            
            # Torso / Shoulders
            shoulder_center = (translated_body[5][:2] + translated_body[6][:2]) / 2
            spine_vec = shoulder_center - center_current
            
            torso_scale_L = limb_scales.get((5, 11), global_scale)
            torso_scale_R = limb_scales.get((6, 12), global_scale)
            avg_torso_scale = (torso_scale_L + torso_scale_R) / 2
            
            new_shoulder_center = center_current + spine_vec * avg_torso_scale 
            
            shoulder_width_scale = limb_scales.get((5, 6), global_scale)
            for s_idx in [5, 6]:
                if translated_body[s_idx][2] > 0:
                    vec = translated_body[s_idx][:2] - shoulder_center
                    final_body[s_idx][:2] = new_shoulder_center + vec * shoulder_width_scale
            
            # Limbs (Arms/Legs)
            queue = [
                (11, 13), (13, 15), # L Leg
                (12, 14), (14, 16), # R Leg
                (5, 7), (7, 9),     # L Arm
                (6, 8), (8, 10)     # R Arm
            ]
            
            for p, c in queue:
                if translated_body[p][2] > 0 and translated_body[c][2] > 0:
                    # Vector from translated parent to translated child
                    # IMPORTANT: We must use the vector from the translated input to preserve rotation.
                    vec = translated_body[c][:2] - translated_body[p][:2]
                    scale = limb_scales.get((p, c), global_scale)
                    
                    # Attach to the NEW parent position
                    final_body[c][:2] = final_body[p][:2] + vec * scale
            
            # Feet (Indices 17-22)
            # 15 -> 17, 18, 19
            # 16 -> 20, 21, 22
            if len(translated_body) > 22:
                foot_groups = [(15, [17, 18, 19]), (16, [20, 21, 22])]
                for ankle, toes in foot_groups:
                    for toe in toes:
                        if toe < len(translated_body) and translated_body[ankle][2] > 0 and translated_body[toe][2] > 0:
                            vec = translated_body[toe][:2] - translated_body[ankle][:2]
                            final_body[toe][:2] = final_body[ankle][:2] + vec * global_scale
                            
            # Head/Face (Indices 0-4)
            if final_body[0][2] > 0:
                 neck_old = (translated_body[5][:2] + translated_body[6][:2]) / 2
                 neck_new = (final_body[5][:2] + final_body[6][:2]) / 2
                 
                 for h_idx in range(5):
                     if h_idx < len(translated_body) and translated_body[h_idx][2] > 0:
                         vec = translated_body[h_idx][:2] - neck_old
                         final_body[h_idx][:2] = neck_new + vec * global_scale

        # --- Retarget Hands ---
        # Anchor: Body Wrist
        # L Wrist = 9, R Wrist = 10
        
        # Left Hand
        if src_l_hand is not None and len(src_l_hand) > 0:
            # Anchor is New Body Wrist
            target_wrist = final_body[9][:2]
            # Old Anchor is Old Translated Body Wrist
            w_old = translated_body[9][:2]
            
            new_l_hand = src_l_hand.copy()
            for i in range(len(new_l_hand)):
                 if new_l_hand[i][2] > 0:
                     p_old = src_l_hand[i][:2] + offset # Translate first
                     vec = p_old - w_old
                     new_l_hand[i][:2] = target_wrist + vec * l_hand_scale
            
            new_frame["people"][0]["hand_left_keypoints_2d"] = new_l_hand.flatten().tolist()

        # Right Hand
        if src_r_hand is not None and len(src_r_hand) > 0:
            target_wrist = final_body[10][:2]
            w_old = translated_body[10][:2]
            
            new_r_hand = src_r_hand.copy()
            for i in range(len(new_r_hand)):
                 if new_r_hand[i][2] > 0:
                     p_old = src_r_hand[i][:2] + offset
                     vec = p_old - w_old
                     new_r_hand[i][:2] = target_wrist + vec * r_hand_scale
            
            new_frame["people"][0]["hand_right_keypoints_2d"] = new_r_hand.flatten().tolist()

        # Update Body in Frame
        new_frame["people"][0]["pose_keypoints_2d"] = final_body.flatten().tolist()

        # Update Canvas Size to Target Size
        if target_data["canvas_width"] > 0:
            new_frame["canvas_width"] = target_data["canvas_width"]
            new_frame["canvas_height"] = target_data["canvas_height"]
        
        # Append Result
        result_frames.append(new_frame)

    return result_frames
