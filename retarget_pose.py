# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
from tqdm import tqdm
import math 
from typing import NamedTuple
import copy
import json
try:
    from .pose_utils.pose2d_utils import AAPoseMeta
except ImportError:
    AAPoseMeta = None

# load skeleton name and bone lines
keypoint_list = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist", # No.4
        "LShoulder",
        "LElbow",
        "LWrist", # No.7
        "RHip",
        "RKnee",
        "RAnkle", # No.10
        "LHip",
        "LKnee",
        "LAnkle", # No.13
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LToe",
        "RToe",
]


limbSeq = [
    [2, 3], [2, 6],     # shoulders
    [3, 4], [4, 5],     # left arm
    [6, 7], [7, 8],     # right arm
    [2, 9], [9, 10], [10, 11],    # right leg 
    [2, 12], [12, 13], [13, 14],  # left leg
    [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], # face (nose, eyes, ears)
    [14, 19], # left foot
    [11, 20] #  right foot
]

eps = 0.01

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


# for each limb, calculate src & dst bone's length
# and calculate their ratios 
def get_length(skeleton, limb):
    
    k1_index, k2_index = limb
    
    H, W = skeleton['height'], skeleton['width']
    keypoints = skeleton['keypoints_body']
    keypoint1 = keypoints[k1_index - 1]
    keypoint2 = keypoints[k2_index - 1]

    if keypoint1 is None or keypoint2 is None:
        return None, None, None
    
    X = np.array([keypoint1[0], keypoint2[0]]) * float(W)
    Y = np.array([keypoint1[1], keypoint2[1]]) * float(H)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    
    return X, Y, length



def get_handpose_meta(keypoints, delta, src_H, src_W):

    new_keypoints = []
    
    # 首先计算手腕位置（第0个点），用于后续的异常检测
    wrist_pixel = None
    if len(keypoints) > 0 and keypoints[0] is not None and keypoints[0].score > 0:
        if not (abs(keypoints[0].x) < 1e-6 and abs(keypoints[0].y) < 1e-6):
            wrist_x = keypoints[0].x * src_W + delta[0]
            wrist_y = keypoints[0].y * src_H + delta[1]
            wrist_pixel = (wrist_x, wrist_y)

    for idx, keypoint in enumerate(keypoints):
        if keypoint is None:
            new_keypoints.append(None)
            continue
        if keypoint.score <= 0:  # score=0 或 score=-1 都表示无效点
            new_keypoints.append(None)
            continue
        
        # 检查坐标是否有效（不是全0或无效值）
        # keypoint.x 和 keypoint.y 是归一化坐标（0-1范围）
        # 只过滤掉明确为 0 的点（使用很小的阈值避免浮点数精度问题）
        if abs(keypoint.x) < 1e-6 and abs(keypoint.y) < 1e-6:
            new_keypoints.append(None)
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * src_W + delta[0])
        y = int(y * src_H + delta[1])
        
        # 安全检查：如果手指关键点距离手腕太远，认为是异常数据
        # 手部关键点到手腕的距离不应超过画布尺寸的一定比例
        if wrist_pixel is not None and idx > 0:
            dist_to_wrist = math.sqrt((x - wrist_pixel[0])**2 + (y - wrist_pixel[1])**2)
            max_dist = max(src_W, src_H) * 0.25  # 手指长度不应超过画布的25%
            if dist_to_wrist > max_dist:
                # 异常点，跳过
                new_keypoints.append(None)
                continue

        new_keypoints.append(                
                Keypoint(
                    x=x,
                    y=y,
                    score=keypoint.score,
                ))

    return new_keypoints


def deal_hand_keypoints(hand_res, r_ratio, l_ratio, hand_score_th = 0.5):

    left_hand = []
    right_hand = []
    
    # 检查手腕位置（第0个点）是否有效
    # 手部坐标已经是归一化的（0-1范围），检查 score 而不是坐标值
    # 如果整个手部的所有点 score 都是 0 或 -1，说明没有检测到手部
    # score=-1 是特殊标记，表示原始输入数据为空数组（完全没有检测到该手）
    left_wrist_valid = hand_res['left'][0][2] > 0  # score=-1 或 0 都认为无效
    right_wrist_valid = hand_res['right'][0][2] > 0  # score=-1 或 0 都认为无效
    
    # 获取手腕位置用于异常检测（归一化坐标）
    left_wrist = (hand_res['left'][0][0], hand_res['left'][0][1]) if left_wrist_valid else None
    right_wrist = (hand_res['right'][0][0], hand_res['right'][0][1]) if right_wrist_valid else None
    
    # 手指到手腕的最大允许距离（归一化坐标，约占画面15%）
    max_finger_dist = 0.15

    left_delta_x = hand_res['left'][0][0] * (l_ratio - 1) 
    left_delta_y = hand_res['left'][0][1] * (l_ratio - 1)

    right_delta_x = hand_res['right'][0][0] * (r_ratio - 1)
    right_delta_y = hand_res['right'][0][1] * (r_ratio - 1)

    length = len(hand_res['left'])

    for i in range(length):
        # left hand
        # score 必须 > 0 且 >= threshold (score=-1 或 0 都表示无效)
        left_point_valid = left_wrist_valid and hand_res['left'][i][2] > 0 and hand_res['left'][i][2] >= hand_score_th
        
        # 检查手指点到手腕距离是否合理（仅对非手腕点检查）
        if left_point_valid and i > 0 and left_wrist is not None:
            dx = hand_res['left'][i][0] - left_wrist[0]
            dy = hand_res['left'][i][1] - left_wrist[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > max_finger_dist:
                left_point_valid = False  # 异常点，标记为无效
        
        if not left_point_valid:
            left_hand.append(
                Keypoint(
                    x=-1,
                    y=-1,
                    score=0,
                )
            )
        else:
            left_hand.append(
                Keypoint(
                    x=hand_res['left'][i][0] * l_ratio - left_delta_x,
                    y=hand_res['left'][i][1] * l_ratio - left_delta_y,
                    score = hand_res['left'][i][2]
                )
            )

        # right hand
        # score 必须 > 0 且 >= threshold (score=-1 或 0 都表示无效)
        right_point_valid = right_wrist_valid and hand_res['right'][i][2] > 0 and hand_res['right'][i][2] >= hand_score_th
        
        # 检查手指点到手腕距离是否合理（仅对非手腕点检查）
        if right_point_valid and i > 0 and right_wrist is not None:
            dx = hand_res['right'][i][0] - right_wrist[0]
            dy = hand_res['right'][i][1] - right_wrist[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > max_finger_dist:
                right_point_valid = False  # 异常点，标记为无效
        
        if not right_point_valid:
            right_hand.append(
                Keypoint(
                    x=-1,
                    y=-1,
                    score=0,
                )
            )
        else:
            right_hand.append(
                Keypoint(
                    x=hand_res['right'][i][0] * r_ratio - right_delta_x,
                    y=hand_res['right'][i][1] * r_ratio - right_delta_y,
                    score = hand_res['right'][i][2]
                )
            )

    return left_hand, right_hand


def get_scaled_pose(canvas, src_canvas, keypoints, keypoints_hand, bone_ratio_list, delta_ground_x, delta_ground_y,
                                       rescaled_src_ground_x, body_flag, id, scale_min, threshold = 0.4, hand_ratio=None, keypoints_foot=None):

    H, W = canvas
    src_H, src_W = src_canvas

    new_length_list = [ ] 
    angle_list = [ ]

    # keypoints from 0-1 to H/W range
    for idx in range(len(keypoints)):
        if keypoints[idx] is None or len(keypoints[idx]) == 0:
            continue

        keypoints[idx] = [keypoints[idx][0] * src_W, keypoints[idx][1] * src_H, keypoints[idx][2]]

    # first traverse, get new_length_list and angle_list
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None or len(keypoint1) == 0 or len(keypoint2) == 0:
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) #* float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) #* float(H)

        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5

        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    # Keep foot length within 0.5x calf length
    foot_lower_leg_ratio = 0.5
    if new_length_list[8] != None and new_length_list[18] != None:
        if new_length_list[18] > new_length_list[8] * foot_lower_leg_ratio:
            new_length_list[18] = new_length_list[8] * foot_lower_leg_ratio

    if new_length_list[11] != None and new_length_list[17] != None:
        if new_length_list[17] > new_length_list[11] * foot_lower_leg_ratio:
            new_length_list[17] = new_length_list[11] * foot_lower_leg_ratio

    # second traverse, calculate new keypoints
    rescale_keypoints = keypoints.copy()

    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        # update dst_keypoints
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if rescale_keypoints[k1_index - 1] is None or rescale_keypoints[k2_index - 1] is None or \
            len(rescale_keypoints[k1_index - 1]) == 0 or len(rescale_keypoints[k2_index - 1]) == 0:
            continue

        # calculate end_keypoint
        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))
        
        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        # update keypoints
        rescale_keypoints[k2_index - 1] = [end_keypoint_x, end_keypoint_y, rescale_keypoints[k2_index - 1][2]]

    if id == 0:
        if body_flag == 'full_body' and rescale_keypoints[8] != None and rescale_keypoints[11] != None:
            delta_ground_x_offset_first_frame = (rescale_keypoints[8][0] + rescale_keypoints[11][0]) / 2 - rescaled_src_ground_x
            delta_ground_x += delta_ground_x_offset_first_frame
        elif body_flag == 'half_body' and rescale_keypoints[1] != None:
            delta_ground_x_offset_first_frame = rescale_keypoints[1][0] - rescaled_src_ground_x
            delta_ground_x += delta_ground_x_offset_first_frame

    # offset all keypoints
    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0 :
            continue
        rescale_keypoints[idx][0] -= delta_ground_x
        rescale_keypoints[idx][1] -= delta_ground_y

        # rescale keypoints to original size
        rescale_keypoints[idx][0] /= scale_min
        rescale_keypoints[idx][1] /= scale_min

    # Scale hand proportions based on actual hand keypoints or body skeletal ratios
    # 优先使用手部关键点计算的缩放系数
    if hand_ratio is not None and hand_ratio['right'] is not None:
        r_ratio = hand_ratio['right'] / scale_min
    else:
        r_ratio = 0.5*max(bone_ratio_list[0], bone_ratio_list[1]) / scale_min
    
    if hand_ratio is not None and hand_ratio['left'] is not None:
        l_ratio = hand_ratio['left'] / scale_min
    else:
        l_ratio = 0.5*max(bone_ratio_list[0], bone_ratio_list[1]) / scale_min
    
    left_hand, right_hand = deal_hand_keypoints(keypoints_hand, r_ratio, l_ratio, hand_score_th = threshold)

    left_hand_new = left_hand.copy()
    right_hand_new = right_hand.copy()
    
    # 注意：hand 坐标是归一化到源画布的，需要使用 src_W, src_H 进行转换
    # keypoints 是原始像素坐标，rescale_keypoints 是变换后的坐标
    # 变换公式: transformed = (original - delta_ground) / scale_min
    # 逆变换公式: original = transformed * scale_min + delta_ground
    if rescale_keypoints[4] == None and rescale_keypoints[7] == None:
        # 当两个手腕都缺失时，也需要将手部关键点转换为像素坐标
        # 使用源画布尺寸 src_H, src_W，因为手部坐标是相对于源画布归一化的
        left_hand_new = get_handpose_meta(left_hand, np.array([0, 0]), src_H, src_W)
        right_hand_new = get_handpose_meta(right_hand, np.array([0, 0]), src_H, src_W)

    elif rescale_keypoints[4] == None and rescale_keypoints[7] != None:
        # 只有 LWrist (7) 存在
        # 注意：在 retarget 过程中，源视频的手部数据可能需要映射到目标骨骼的对应手腕
        # 经过测试，需要交叉匹配：left_hand 对应 RWrist(4)，right_hand 对应 LWrist(7)
        right_hand_delta = np.array([
            rescale_keypoints[7][0] - right_hand[0].x * src_W,
            rescale_keypoints[7][1] - right_hand[0].y * src_H
        ]) if right_hand[0].x != -1 else np.array([0, 0])
        right_hand_new = get_handpose_meta(right_hand, right_hand_delta, src_H, src_W)
        # 左手无法计算（RWrist 不存在），必须清空为全 None，避免错误保留原始数据
        left_hand_new = [None] * len(left_hand)

    elif rescale_keypoints[4] != None and rescale_keypoints[7] == None:
        # 只有 RWrist (4) 存在
        left_hand_delta = np.array([
            rescale_keypoints[4][0] - left_hand[0].x * src_W,
            rescale_keypoints[4][1] - left_hand[0].y * src_H
        ]) if left_hand[0].x != -1 else np.array([0, 0])
        left_hand_new = get_handpose_meta(left_hand, left_hand_delta, src_H, src_W)
        # 右手无法计算（LWrist 不存在），必须清空为全 None，避免错误保留原始数据
        right_hand_new = [None] * len(right_hand)

    else:
        # 交叉匹配：经过测试发现需要交换左右手的目标手腕
        # left_hand (源数据的 hand_left) → 移动到 RWrist (index 4) 位置
        # right_hand (源数据的 hand_right) → 移动到 LWrist (index 7) 位置
        # 
        # 原因：retarget 过程中骨骼的左右可能发生了翻转，
        # 导致源视频的左手需要映射到目标骨骼的右手腕位置
        
        # left_hand 移动到 RWrist (index 4)
        left_hand_delta = np.array([
            rescale_keypoints[4][0] - left_hand[0].x * src_W,
            rescale_keypoints[4][1] - left_hand[0].y * src_H
        ]) if left_hand[0].x != -1 else np.array([0, 0])
        
        # right_hand 移动到 LWrist (index 7)
        right_hand_delta = np.array([
            rescale_keypoints[7][0] - right_hand[0].x * src_W,
            rescale_keypoints[7][1] - right_hand[0].y * src_H
        ]) if right_hand[0].x != -1 else np.array([0, 0])

        left_hand_new = get_handpose_meta(left_hand, left_hand_delta, src_H, src_W)
        right_hand_new = get_handpose_meta(right_hand, right_hand_delta, src_H, src_W)

    # get normalized keypoints_body
    # 注意：rescale_keypoints 是基于源画布尺度的像素坐标，需要使用 src_W, src_H 归一化
    # 同样，手部坐标也是基于源画布尺度的，所以 frame_info 的尺寸应该使用源画布尺寸
    norm_body_keypoints = [ ]
    for body_keypoint in rescale_keypoints:
        if body_keypoint != None:
            norm_body_keypoints.append([body_keypoint[0] / src_W , body_keypoint[1] / src_H, body_keypoint[2]])
        else:
            norm_body_keypoints.append(None)

    # 处理脚部关键点（完整的6个点）
    # 顺序: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
    # 固定匹配：索引 0-2 对应 RAnkle，索引 3-5 对应 LAnkle
    norm_foot_keypoints = []
    if keypoints_foot is not None and len(keypoints_foot) >= 6:
        # 获取原始 ankle 位置（像素坐标，keypoints 已转换）
        r_ankle_orig = keypoints[10] if keypoints[10] is not None else None
        l_ankle_orig = keypoints[13] if keypoints[13] is not None else None
        
        # 获取 rescaled ankle 位置
        r_ankle_new = rescale_keypoints[10] if rescale_keypoints[10] is not None else None
        l_ankle_new = rescale_keypoints[13] if rescale_keypoints[13] is not None else None
        
        # 获取脚部骨骼缩放比例
        r_foot_ratio = bone_ratio_list[18] if len(bone_ratio_list) > 18 and bone_ratio_list[18] is not None and bone_ratio_list[18] > 0 else 1.0
        l_foot_ratio = bone_ratio_list[17] if len(bone_ratio_list) > 17 and bone_ratio_list[17] is not None and bone_ratio_list[17] > 0 else 1.0
        
        for i in range(6):
            foot_kp = keypoints_foot[i]
            if foot_kp is not None and len(foot_kp) >= 3 and foot_kp[2] > 0:
                # 原始脚部像素坐标
                foot_orig_px = [foot_kp[0] * src_W, foot_kp[1] * src_H]
                
                # 固定匹配：索引 0-2 (R foot) 对应 RAnkle，索引 3-5 (L foot) 对应 LAnkle
                if i < 3:
                    # 右脚关键点 (RBigToe, RSmallToe, RHeel) 对应 RAnkle
                    ankle_orig = r_ankle_orig
                    ankle_new = r_ankle_new
                    foot_ratio = r_foot_ratio
                else:
                    # 左脚关键点 (LBigToe, LSmallToe, LHeel) 对应 LAnkle
                    ankle_orig = l_ankle_orig
                    ankle_new = l_ankle_new
                    foot_ratio = l_foot_ratio
                
                if ankle_orig is not None and ankle_new is not None:
                    # 计算脚部关键点相对于 ankle 的偏移
                    rel_x = foot_orig_px[0] - ankle_orig[0]
                    rel_y = foot_orig_px[1] - ankle_orig[1]
                    
                    # 应用脚部骨骼缩放比例到相对偏移，计算新位置
                    new_foot_px = [
                        ankle_new[0] + rel_x * foot_ratio,
                        ankle_new[1] + rel_y * foot_ratio
                    ]
                    
                    # 归一化到 0-1
                    norm_foot_keypoints.append([new_foot_px[0] / src_W, new_foot_px[1] / src_H, foot_kp[2]])
                else:
                    # 没有 ankle 参考，清零脚部关键点（不保留没有父节点支持的脚部关键点）
                    norm_foot_keypoints.append([0, 0, 0])
            else:
                norm_foot_keypoints.append([0, 0, 0])
    else:
        norm_foot_keypoints = [[0, 0, 0]] * 6

    # OpenPose 约定：hand_left = 人的左手（图像右侧），hand_right = 人的右手（图像左侧）
    # left_hand_new 使用 LWrist (7) 偏移计算 → 在图像右侧 → 对应 hand_left
    # right_hand_new 使用 RWrist (4) 偏移计算 → 在图像左侧 → 对应 hand_right
    frame_info = {
                    'height': src_H,
                    'width': src_W,
                    'keypoints_body': norm_body_keypoints,
                    'keypoints_left_hand' : left_hand_new,
                    'keypoints_right_hand' : right_hand_new,
                    'keypoints_foot': norm_foot_keypoints,  # 新增：完整的6个脚部关键点
                }

    return frame_info


def rescale_skeleton(H, W, keypoints, bone_ratio_list):

    rescale_keypoints = keypoints.copy()

    new_length_list = [ ] 
    angle_list = [ ]

    # keypoints from 0-1 to H/W range
    for idx in range(len(rescale_keypoints)):
        if rescale_keypoints[idx] is None or len(rescale_keypoints[idx]) == 0:
            continue

        rescale_keypoints[idx] = [rescale_keypoints[idx][0] * W, rescale_keypoints[idx][1] * H]

    # first traverse, get new_length_list and angle_list
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        keypoint1 = rescale_keypoints[k1_index - 1]
        keypoint2 = rescale_keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None or len(keypoint1) == 0 or len(keypoint2) == 0:
            new_length_list.append(None)
            angle_list.append(None)
            continue

        Y = np.array([keypoint1[0], keypoint2[0]]) #* float(W)
        X = np.array([keypoint1[1], keypoint2[1]]) #* float(H)

        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5


        new_length = length * bone_ratio_list[idx]
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        new_length_list.append(new_length)
        angle_list.append(angle)

    # # second traverse, calculate new keypoints
    for idx, (k1_index, k2_index) in enumerate(limbSeq):
        # update dst_keypoints
        start_keypoint = rescale_keypoints[k1_index - 1]
        new_length = new_length_list[idx]
        angle = angle_list[idx]

        if rescale_keypoints[k1_index - 1] is None or rescale_keypoints[k2_index - 1] is None or \
            len(rescale_keypoints[k1_index - 1]) == 0 or len(rescale_keypoints[k2_index - 1]) == 0:
            continue

        # calculate end_keypoint
        delta_x = new_length * math.cos(math.radians(angle))
        delta_y = new_length * math.sin(math.radians(angle))
        
        end_keypoint_x = start_keypoint[0] - delta_x
        end_keypoint_y = start_keypoint[1] - delta_y

        # update keypoints
        rescale_keypoints[k2_index - 1] = [end_keypoint_x, end_keypoint_y]

    return rescale_keypoints


def fix_lack_keypoints_use_sym(skeleton):

    keypoints = skeleton['keypoints_body']
    H, W = skeleton['height'], skeleton['width']

    limb_points_list = [
                        [3, 4, 5],
                        [6, 7, 8],
                        [12, 13, 14, 19],
                        [9, 10, 11, 20],
    ]

    for limb_points in limb_points_list:
        miss_flag = False
        for point in limb_points:
            if keypoints[point - 1] is None:
                miss_flag = True
                continue
            if miss_flag:
                skeleton['keypoints_body'][point - 1] = None

    repair_limb_seq_left = [
        [3, 4], [4, 5],     # left arm
        [12, 13], [13, 14],  # left leg
        # [14, 19] # left foot - 已移除：不再自动修复脚部关键点
    ]

    repair_limb_seq_right = [
        [6, 7], [7, 8],     # right arm
        [9, 10], [10, 11],    # right leg 
        # [11, 20] # right foot - 已移除：不再自动修复脚部关键点
    ]

    repair_limb_seq = [repair_limb_seq_left, repair_limb_seq_right]

    for idx_part, part in enumerate(repair_limb_seq):
        for idx, limb in enumerate(part):

            k1_index, k2_index = limb
            keypoint1 = keypoints[k1_index - 1]
            keypoint2 = keypoints[k2_index - 1]

            if keypoint1 != None and keypoint2 is None:
                # reference to symmetric limb
                sym_limb = repair_limb_seq[1-idx_part][idx]
                k1_index_sym, k2_index_sym = sym_limb
                keypoint1_sym = keypoints[k1_index_sym - 1]
                keypoint2_sym = keypoints[k2_index_sym - 1]
                ref_length = 0

                if keypoint1_sym != None and keypoint2_sym != None:
                    X = np.array([keypoint1_sym[0], keypoint2_sym[0]]) * float(W)
                    Y = np.array([keypoint1_sym[1], keypoint2_sym[1]]) * float(H)
                    ref_length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                else:
                    ref_length_left, ref_length_right = 0, 0
                    if keypoints[1] != None and keypoints[8] != None:
                        X = np.array([keypoints[1][0], keypoints[8][0]]) * float(W)
                        Y = np.array([keypoints[1][1], keypoints[8][1]]) * float(H)
                        ref_length_left = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        if idx <= 1: # arms
                            ref_length_left /= 2
                    
                    if keypoints[1] != None and keypoints[11] != None:
                        X = np.array([keypoints[1][0], keypoints[11][0]]) * float(W)
                        Y = np.array([keypoints[1][1], keypoints[11][1]]) * float(H)
                        ref_length_right = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                        if idx <= 1: # arms
                            ref_length_right /= 2
                        elif idx == 4: # foot
                            ref_length_right /= 5

                    ref_length = max(ref_length_left, ref_length_right)
                    
                if ref_length != 0:
                    skeleton['keypoints_body'][k2_index - 1] = [0, 0] #init
                    skeleton['keypoints_body'][k2_index - 1][0] = skeleton['keypoints_body'][k1_index - 1][0]
                    skeleton['keypoints_body'][k2_index - 1][1] = skeleton['keypoints_body'][k1_index - 1][1] + ref_length / H
    return skeleton


def rescale_shorten_skeleton(ratio_list, src_length_list, dst_length_list):

    # 对称骨骼取最大值（保证左右对称）
    modify_bone_list = [
        [0, 1],
        [2, 4],
        [3, 5],
        [6, 9],
        [7, 10],
        [8, 11],
    ]

    for modify_bone in modify_bone_list:
        new_ratio = max(ratio_list[modify_bone[0]], ratio_list[modify_bone[1]])
        ratio_list[modify_bone[0]] = new_ratio
        ratio_list[modify_bone[1]] = new_ratio
    
    # 左右脚：如果都有效则取均值，否则取有效的那个
    if ratio_list[17] is not None and ratio_list[17] > 0 and ratio_list[18] is not None and ratio_list[18] > 0:
        foot_ratio_avg = (ratio_list[17] + ratio_list[18]) / 2
        ratio_list[17] = foot_ratio_avg
        ratio_list[18] = foot_ratio_avg
    elif ratio_list[17] is not None and ratio_list[17] > 0:
        ratio_list[18] = ratio_list[17]
    elif ratio_list[18] is not None and ratio_list[18] > 0:
        ratio_list[17] = ratio_list[18]
    
    if ratio_list[13]!= None and ratio_list[15]!= None:
        ratio_eye_avg = (ratio_list[13] + ratio_list[15]) / 2
        ratio_list[13] = ratio_eye_avg
        ratio_list[15] = ratio_eye_avg

    if ratio_list[14]!= None and ratio_list[16]!= None:
        ratio_eye_avg = (ratio_list[14] + ratio_list[16]) / 2
        ratio_list[14] = ratio_eye_avg
        ratio_list[16] = ratio_eye_avg

    return ratio_list, src_length_list, dst_length_list



def check_full_body(keypoints, threshold = 0.4):

    body_flag = 'half_body'

    # 1. If ankle points exist, confidence is greater than the threshold, and points do not exceed the frame, return full_body
    if keypoints[10] != None and keypoints[13] != None and keypoints[8] != None and keypoints[11] != None:
        if (keypoints[10][1] <= 1 and keypoints[13][1] <= 1) and (keypoints[10][2] >= threshold and keypoints[13][2] >= threshold) and \
            (keypoints[8][1] <= 1 and keypoints[11][1] <= 1) and (keypoints[8][2] >= threshold and keypoints[11][2] >= threshold):
            body_flag = 'full_body'
            return body_flag

    # 2. If hip points exist, return three_quarter_body
    if (keypoints[8] != None and keypoints[11] != None):
        if (keypoints[8][1] <= 1 and keypoints[11][1] <= 1) and (keypoints[8][2] >= threshold and keypoints[11][2] >= threshold):
            body_flag = 'three_quarter_body'
            return body_flag
    
    return body_flag


def check_full_body_both(flag1, flag2):
    body_flag_dict = {
        'full_body': 2,
        'three_quarter_body' : 1,
        'half_body': 0
    }

    body_flag_dict_reverse = {
        2: 'full_body', 
        1: 'three_quarter_body',
        0: 'half_body'
    }

    flag1_num = body_flag_dict[flag1]
    flag2_num = body_flag_dict[flag2]
    flag_both_num = min(flag1_num, flag2_num)
    return body_flag_dict_reverse[flag_both_num]


def write_to_poses(data_to_json, none_idx, dst_shape, bone_ratio_list, delta_ground_x, delta_ground_y, rescaled_src_ground_x, body_flag, scale_min, hand_ratio=None):
    outputs = []
    length = len(data_to_json)
    for id in tqdm(range(length)):

        src_height, src_width = data_to_json[id]['height'], data_to_json[id]['width']
        width, height = dst_shape
        keypoints = data_to_json[id]['keypoints_body']
        for idx in range(len(keypoints)):
            if idx in none_idx:
                keypoints[idx] = None
        new_keypoints = keypoints.copy()

        # get hand keypoints
        left_hand_data = data_to_json[id]['keypoints_left_hand']
        right_hand_data = data_to_json[id]['keypoints_right_hand']
        
        # 检查手部坐标是否已经归一化（如果最大坐标值 <= 1，则已归一化）
        # 只有当坐标是像素坐标时才需要归一化
        def need_normalize(hand_kps, width, height):
            """检查手部坐标是否需要归一化"""
            for kp in hand_kps:
                if kp is not None and len(kp) >= 2:
                    # 如果任何坐标值大于1，说明是像素坐标，需要归一化
                    if kp[0] > 1.0 or kp[1] > 1.0:
                        return True
            return False
        
        # Normalize hand coordinates to 0-1 range (only if they are pixel coordinates)
        if need_normalize(left_hand_data, src_width, src_height):
            for hand_idx in range(len(left_hand_data)):
                left_hand_data[hand_idx][0] = left_hand_data[hand_idx][0] / src_width
                left_hand_data[hand_idx][1] = left_hand_data[hand_idx][1] / src_height

        if need_normalize(right_hand_data, src_width, src_height):
            for hand_idx in range(len(right_hand_data)):
                right_hand_data[hand_idx][0] = right_hand_data[hand_idx][0] / src_width
                right_hand_data[hand_idx][1] = right_hand_data[hand_idx][1] / src_height
        
        # 固定匹配：hand_left 对应 LWrist (7)，hand_right 对应 RWrist (4)
        # 不再使用距离匹配来交换左右手，避免一只手缺失时导致位置错误
        keypoints_hand = {'left': left_hand_data, 'right': right_hand_data}
        current_hand_ratio = hand_ratio

        # 获取脚部关键点数据
        keypoints_foot = data_to_json[id].get('keypoints_foot', None)
        
        frame_info = get_scaled_pose((height, width), (src_height, src_width), new_keypoints, keypoints_hand, bone_ratio_list, delta_ground_x, delta_ground_y, rescaled_src_ground_x, body_flag, id, scale_min, hand_ratio=current_hand_ratio, keypoints_foot=keypoints_foot)
        outputs.append(frame_info)

    return outputs


def calculate_scale_ratio(skeleton, skeleton_edit, scale_ratio_flag):
    if scale_ratio_flag:

        headw = max(skeleton['keypoints_body'][0][0], skeleton['keypoints_body'][14][0], skeleton['keypoints_body'][15][0], skeleton['keypoints_body'][16][0], skeleton['keypoints_body'][17][0]) - \
                    min(skeleton['keypoints_body'][0][0], skeleton['keypoints_body'][14][0], skeleton['keypoints_body'][15][0], skeleton['keypoints_body'][16][0], skeleton['keypoints_body'][17][0])
        headw_edit = max(skeleton_edit['keypoints_body'][0][0], skeleton_edit['keypoints_body'][14][0], skeleton_edit['keypoints_body'][15][0], skeleton_edit['keypoints_body'][16][0], skeleton_edit['keypoints_body'][17][0]) - \
                    min(skeleton_edit['keypoints_body'][0][0], skeleton_edit['keypoints_body'][14][0], skeleton_edit['keypoints_body'][15][0], skeleton_edit['keypoints_body'][16][0], skeleton_edit['keypoints_body'][17][0])
        headw_ratio = headw / headw_edit

        _, _, shoulder = get_length(skeleton, [6,3])
        _, _, shoulder_edit = get_length(skeleton_edit, [6,3])
        shoulder_ratio = shoulder / shoulder_edit

        return max(headw_ratio, shoulder_ratio)
    
    else:
        return 1



def retarget_pose(src_skeleton, dst_skeleton, all_src_skeleton, src_skeleton_edit, dst_skeleton_edit, threshold=0.4):

    if src_skeleton_edit is not None and dst_skeleton_edit is not None:
        use_edit_for_base = True
    else:
        use_edit_for_base = False

    src_skeleton_ori = copy.deepcopy(src_skeleton)
    src_skeleton_ori_h, src_skeleton_ori_w = src_skeleton['height'], src_skeleton['width']

    dst_skeleton_ori_h, dst_skeleton_ori_w = dst_skeleton['height'], dst_skeleton['width']
    if src_skeleton['keypoints_body'][0] != None and src_skeleton['keypoints_body'][10] != None and src_skeleton['keypoints_body'][13] != None and \
        dst_skeleton['keypoints_body'][0] != None and dst_skeleton['keypoints_body'][10] != None and dst_skeleton['keypoints_body'][13] != None and \
            src_skeleton['keypoints_body'][0][2] > 0.5 and src_skeleton['keypoints_body'][10][2] > 0.5 and src_skeleton['keypoints_body'][13][2] > 0.5 and \
        dst_skeleton['keypoints_body'][0][2] > 0.5 and dst_skeleton['keypoints_body'][10][2] > 0.5 and dst_skeleton['keypoints_body'][13][2] > 0.5:

        src_height = src_skeleton['height'] * abs(
            (src_skeleton['keypoints_body'][10][1] + src_skeleton['keypoints_body'][13][1]) / 2 -
            src_skeleton['keypoints_body'][0][1])
        dst_height = dst_skeleton['height'] * abs(
            (dst_skeleton['keypoints_body'][10][1] + dst_skeleton['keypoints_body'][13][1]) / 2 -
            dst_skeleton['keypoints_body'][0][1])
        scale_min = 1.0 * src_height / dst_height
    elif src_skeleton['keypoints_body'][0] != None and src_skeleton['keypoints_body'][8] != None and src_skeleton['keypoints_body'][11] != None and \
        dst_skeleton['keypoints_body'][0] != None and dst_skeleton['keypoints_body'][8] != None and dst_skeleton['keypoints_body'][11] != None and \
            src_skeleton['keypoints_body'][0][2] > 0.5 and src_skeleton['keypoints_body'][8][2] > 0.5 and src_skeleton['keypoints_body'][11][2] > 0.5 and \
        dst_skeleton['keypoints_body'][0][2] > 0.5 and dst_skeleton['keypoints_body'][8][2] > 0.5 and dst_skeleton['keypoints_body'][11][2] > 0.5:

        src_height = src_skeleton['height'] * abs(
            (src_skeleton['keypoints_body'][8][1] + src_skeleton['keypoints_body'][11][1]) / 2 -
            src_skeleton['keypoints_body'][0][1])
        dst_height = dst_skeleton['height'] * abs(
            (dst_skeleton['keypoints_body'][8][1] + dst_skeleton['keypoints_body'][11][1]) / 2 -
            dst_skeleton['keypoints_body'][0][1])
        scale_min = 1.0 * src_height / dst_height
    else:
        scale_min = np.sqrt(src_skeleton['height'] * src_skeleton['width']) / np.sqrt(dst_skeleton['height'] * dst_skeleton['width'])
    
    if use_edit_for_base:
        scale_ratio_flag = False
        if src_skeleton_edit['keypoints_body'][0] != None and src_skeleton_edit['keypoints_body'][10] != None and src_skeleton_edit['keypoints_body'][13] != None and \
            dst_skeleton_edit['keypoints_body'][0] != None and dst_skeleton_edit['keypoints_body'][10] != None and dst_skeleton_edit['keypoints_body'][13] != None and \
                src_skeleton_edit['keypoints_body'][0][2] > 0.5 and src_skeleton_edit['keypoints_body'][10][2] > 0.5 and src_skeleton_edit['keypoints_body'][13][2] > 0.5 and \
            dst_skeleton_edit['keypoints_body'][0][2] > 0.5 and dst_skeleton_edit['keypoints_body'][10][2] > 0.5 and dst_skeleton_edit['keypoints_body'][13][2] > 0.5:

            src_height_edit = src_skeleton_edit['height'] * abs(
                (src_skeleton_edit['keypoints_body'][10][1] + src_skeleton_edit['keypoints_body'][13][1]) / 2 -
                src_skeleton_edit['keypoints_body'][0][1])
            dst_height_edit = dst_skeleton_edit['height'] * abs(
                (dst_skeleton_edit['keypoints_body'][10][1] + dst_skeleton_edit['keypoints_body'][13][1]) / 2 -
                dst_skeleton_edit['keypoints_body'][0][1])
            scale_min_edit = 1.0 * src_height_edit / dst_height_edit
        elif src_skeleton_edit['keypoints_body'][0] != None and src_skeleton_edit['keypoints_body'][8] != None and src_skeleton_edit['keypoints_body'][11] != None and \
            dst_skeleton_edit['keypoints_body'][0] != None and dst_skeleton_edit['keypoints_body'][8] != None and dst_skeleton_edit['keypoints_body'][11] != None and \
                src_skeleton_edit['keypoints_body'][0][2] > 0.5 and src_skeleton_edit['keypoints_body'][8][2] > 0.5 and src_skeleton_edit['keypoints_body'][11][2] > 0.5 and \
            dst_skeleton_edit['keypoints_body'][0][2] > 0.5 and dst_skeleton_edit['keypoints_body'][8][2] > 0.5 and dst_skeleton_edit['keypoints_body'][11][2] > 0.5:

            src_height_edit = src_skeleton_edit['height'] * abs(
                (src_skeleton_edit['keypoints_body'][8][1] + src_skeleton_edit['keypoints_body'][11][1]) / 2 -
                src_skeleton_edit['keypoints_body'][0][1])
            dst_height_edit = dst_skeleton_edit['height'] * abs(
                (dst_skeleton_edit['keypoints_body'][8][1] + dst_skeleton_edit['keypoints_body'][11][1]) / 2 -
                dst_skeleton_edit['keypoints_body'][0][1])
            scale_min_edit = 1.0 * src_height_edit / dst_height_edit
        else:
            scale_min_edit = np.sqrt(src_skeleton_edit['height'] * src_skeleton_edit['width']) / np.sqrt(dst_skeleton_edit['height'] * dst_skeleton_edit['width'])
            scale_ratio_flag = True
        
        # Flux may change the scale, compensate for it here
        ratio_src = calculate_scale_ratio(src_skeleton, src_skeleton_edit, scale_ratio_flag)
        ratio_dst = calculate_scale_ratio(dst_skeleton, dst_skeleton_edit, scale_ratio_flag)

        dst_skeleton_edit['height'] = int(dst_skeleton_edit['height'] * scale_min_edit)
        dst_skeleton_edit['width'] = int(dst_skeleton_edit['width'] * scale_min_edit)
        for idx in range(len(dst_skeleton_edit['keypoints_left_hand'])):
            dst_skeleton_edit['keypoints_left_hand'][idx][0] *= scale_min_edit
            dst_skeleton_edit['keypoints_left_hand'][idx][1] *= scale_min_edit
        for idx in range(len(dst_skeleton_edit['keypoints_right_hand'])):
            dst_skeleton_edit['keypoints_right_hand'][idx][0] *= scale_min_edit
            dst_skeleton_edit['keypoints_right_hand'][idx][1] *= scale_min_edit
    

    dst_skeleton['height'] = int(dst_skeleton['height'] * scale_min)
    dst_skeleton['width'] = int(dst_skeleton['width'] * scale_min)
    for idx in range(len(dst_skeleton['keypoints_left_hand'])):
        dst_skeleton['keypoints_left_hand'][idx][0] *= scale_min
        dst_skeleton['keypoints_left_hand'][idx][1] *= scale_min
    for idx in range(len(dst_skeleton['keypoints_right_hand'])):
        dst_skeleton['keypoints_right_hand'][idx][0] *= scale_min
        dst_skeleton['keypoints_right_hand'][idx][1] *= scale_min


    dst_body_flag = check_full_body(dst_skeleton['keypoints_body'], threshold)
    src_body_flag = check_full_body(src_skeleton_ori['keypoints_body'], threshold)
    body_flag = check_full_body_both(dst_body_flag, src_body_flag)
    #print('body_flag: ', body_flag)

    if use_edit_for_base:
        src_skeleton_edit = fix_lack_keypoints_use_sym(src_skeleton_edit)
        dst_skeleton_edit = fix_lack_keypoints_use_sym(dst_skeleton_edit)
    else:
        src_skeleton = fix_lack_keypoints_use_sym(src_skeleton)
        dst_skeleton = fix_lack_keypoints_use_sym(dst_skeleton)

    none_idx = []
    for idx in range(len(dst_skeleton['keypoints_body'])):
        if dst_skeleton['keypoints_body'][idx] == None or src_skeleton['keypoints_body'][idx] == None:
            src_skeleton['keypoints_body'][idx] = None
            dst_skeleton['keypoints_body'][idx] = None
            none_idx.append(idx)

    # get bone ratio list
    ratio_list, src_length_list, dst_length_list = [], [], []
    for idx, limb in enumerate(limbSeq):
        if use_edit_for_base:
            src_X, src_Y, src_length = get_length(src_skeleton_edit, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton_edit, limb)

            if src_X is None or src_Y is None or dst_X is None or dst_Y is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length * ratio_dst / src_length / ratio_src
        
        else:
            src_X, src_Y, src_length = get_length(src_skeleton, limb)
            dst_X, dst_Y, dst_length = get_length(dst_skeleton, limb)

            if src_X is None or src_Y is None or dst_X is None or dst_Y is None:
                ratio = -1
            else:
                ratio = 1.0 * dst_length / src_length

        ratio_list.append(ratio)
        src_length_list.append(src_length)
        dst_length_list.append(dst_length)
    
    # 对脚部骨骼（index 17, 18）使用固定匹配计算
    # RBigToe (foot_kps[0]) 对应 RAnkle，LBigToe (foot_kps[3]) 对应 LAnkle
    def get_foot_ratio_with_fixed_matching(skeleton, foot_kps):
        """使用固定匹配计算脚部骨骼比例"""
        if foot_kps is None or len(foot_kps) < 6:
            return None, None
        
        body_kps = skeleton['keypoints_body']
        H, W = skeleton['height'], skeleton['width']
        
        # 获取 ankle 位置 (归一化坐标)
        r_ankle = body_kps[10]  # RAnkle
        l_ankle = body_kps[13]  # LAnkle
        
        # foot_kps 顺序: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
        r_toe_kp = foot_kps[0]  # RBigToe
        l_toe_kp = foot_kps[3]  # LBigToe
        
        # 计算右脚长度 (RAnkle -> RBigToe)
        r_foot_length = None
        if r_ankle is not None and r_toe_kp is not None and len(r_toe_kp) >= 3 and r_toe_kp[2] > 0:
            r_ankle_px = [r_ankle[0] * W, r_ankle[1] * H]
            r_toe_px = [r_toe_kp[0] * W, r_toe_kp[1] * H]
            r_foot_length = ((r_toe_px[0] - r_ankle_px[0])**2 + (r_toe_px[1] - r_ankle_px[1])**2)**0.5
        
        # 计算左脚长度 (LAnkle -> LBigToe)
        l_foot_length = None
        if l_ankle is not None and l_toe_kp is not None and len(l_toe_kp) >= 3 and l_toe_kp[2] > 0:
            l_ankle_px = [l_ankle[0] * W, l_ankle[1] * H]
            l_toe_px = [l_toe_kp[0] * W, l_toe_kp[1] * H]
            l_foot_length = ((l_toe_px[0] - l_ankle_px[0])**2 + (l_toe_px[1] - l_ankle_px[1])**2)**0.5
        
        return l_foot_length, r_foot_length  # 返回 [左脚, 右脚]
    
    # 重新计算脚部骨骼比例（使用固定匹配）
    if use_edit_for_base:
        src_foot_kps = src_skeleton_edit.get('keypoints_foot', None)
        dst_foot_kps = dst_skeleton_edit.get('keypoints_foot', None)
        src_l_foot, src_r_foot = get_foot_ratio_with_fixed_matching(src_skeleton_edit, src_foot_kps)
        dst_l_foot, dst_r_foot = get_foot_ratio_with_fixed_matching(dst_skeleton_edit, dst_foot_kps)
    else:
        src_foot_kps = src_skeleton.get('keypoints_foot', None)
        dst_foot_kps = dst_skeleton.get('keypoints_foot', None)
        src_l_foot, src_r_foot = get_foot_ratio_with_fixed_matching(src_skeleton, src_foot_kps)
        dst_l_foot, dst_r_foot = get_foot_ratio_with_fixed_matching(dst_skeleton, dst_foot_kps)
    
    # 更新脚部比例 (index 17 = 左脚, index 18 = 右脚)
    if src_l_foot is not None and dst_l_foot is not None and src_l_foot > 0:
        if use_edit_for_base:
            ratio_list[17] = 1.0 * dst_l_foot * ratio_dst / src_l_foot / ratio_src
        else:
            ratio_list[17] = 1.0 * dst_l_foot / src_l_foot
    
    if src_r_foot is not None and dst_r_foot is not None and src_r_foot > 0:
        if use_edit_for_base:
            ratio_list[18] = 1.0 * dst_r_foot * ratio_dst / src_r_foot / ratio_src
        else:
            ratio_list[18] = 1.0 * dst_r_foot / src_r_foot
    
    for idx, ratio in enumerate(ratio_list):
        if ratio == -1:
            if ratio_list[0] != -1 and ratio_list[1] != -1:
                ratio_list[idx] = (ratio_list[0] + ratio_list[1]) / 2

    # Consider adding constraints when Flux fails to correct head pose, causing neck issues.
    # if ratio_list[12] > (ratio_list[0]+ratio_list[1])/2*1.25:
    #     ratio_list[12] = (ratio_list[0]+ratio_list[1])/2*1.25
    
    ratio_list, src_length_list, dst_length_list = rescale_shorten_skeleton(ratio_list, src_length_list, dst_length_list)

    rescaled_src_skeleton_ori = rescale_skeleton(src_skeleton_ori['height'], src_skeleton_ori['width'],
                                                 src_skeleton_ori['keypoints_body'], ratio_list)

    # get global translation offset_x and offset_y
    if body_flag == 'full_body':
        #print('use foot mark.')
        dst_ground_y = max(dst_skeleton['keypoints_body'][10][1], dst_skeleton['keypoints_body'][13][1]) * dst_skeleton[
            'height']
        # The midpoint between toe and ankle
        if dst_skeleton['keypoints_body'][18] != None and dst_skeleton['keypoints_body'][19] != None:
            right_foot_mid = (dst_skeleton['keypoints_body'][10][1] + dst_skeleton['keypoints_body'][19][1]) / 2
            left_foot_mid = (dst_skeleton['keypoints_body'][13][1] + dst_skeleton['keypoints_body'][18][1]) / 2
            dst_ground_y = max(left_foot_mid, right_foot_mid) * dst_skeleton['height']

        rescaled_src_ground_y = max(rescaled_src_skeleton_ori[10][1], rescaled_src_skeleton_ori[13][1])
        delta_ground_y = rescaled_src_ground_y - dst_ground_y
       
        dst_ground_x = (dst_skeleton['keypoints_body'][8][0] + dst_skeleton['keypoints_body'][11][0]) * dst_skeleton[
            'width'] / 2
        rescaled_src_ground_x = (rescaled_src_skeleton_ori[8][0] + rescaled_src_skeleton_ori[11][0]) / 2
        delta_ground_x = rescaled_src_ground_x - dst_ground_x
        delta_x, delta_y = delta_ground_x, delta_ground_y

    else:
        #print('use neck mark.')
        # use neck keypoint as mark
        src_neck_y = rescaled_src_skeleton_ori[1][1]
        dst_neck_y = dst_skeleton['keypoints_body'][1][1]
        delta_neck_y = src_neck_y - dst_neck_y * dst_skeleton['height']

        src_neck_x = rescaled_src_skeleton_ori[1][0]
        dst_neck_x = dst_skeleton['keypoints_body'][1][0]
        delta_neck_x = src_neck_x - dst_neck_x * dst_skeleton['width']
        delta_x, delta_y = delta_neck_x, delta_neck_y
        rescaled_src_ground_x = src_neck_x

    
    # 计算手部缩放系数
    hand_ratio = calculate_hand_scale_ratio(src_skeleton_ori, dst_skeleton)
    
    dst_shape = (dst_skeleton_ori_w, dst_skeleton_ori_h)
    # dst_shape = (src_skeleton_ori_w, src_skeleton_ori_h)
    output = write_to_poses(all_src_skeleton, none_idx, dst_shape, ratio_list, delta_x, delta_y,
                                rescaled_src_ground_x, body_flag, scale_min, hand_ratio=hand_ratio)
    return output


def get_retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas, tql_edit_pose_meta0, refer_edit_pose_meta):

    for key, value in tpl_pose_meta0.items():
        if type(value) is np.ndarray:
            if key in ['keypoints_left_hand', 'keypoints_right_hand']:
                value = value * np.array([[tpl_pose_meta0["width"], tpl_pose_meta0["height"], 1.0]])
            if not isinstance(value, list):
                value = value.tolist()
        tpl_pose_meta0[key] = value

    for key, value in refer_pose_meta.items():
        if type(value) is np.ndarray:
            if key in ['keypoints_left_hand', 'keypoints_right_hand']:
                value = value * np.array([[refer_pose_meta["width"], refer_pose_meta["height"], 1.0]])
            if not isinstance(value, list):
                value = value.tolist()
        refer_pose_meta[key] = value

    tpl_pose_metas_new = []
    for meta in tpl_pose_metas:
        for key, value in meta.items():
            if type(value) is np.ndarray:
                if key in ['keypoints_left_hand', 'keypoints_right_hand']:
                    value = value * np.array([[meta["width"], meta["height"], 1.0]])
                if not isinstance(value, list):
                    value = value.tolist()
            meta[key] = value
        tpl_pose_metas_new.append(meta)

    if tql_edit_pose_meta0 is not None:
        for key, value in tql_edit_pose_meta0.items():
            if type(value) is np.ndarray:
                if key in ['keypoints_left_hand', 'keypoints_right_hand']:
                    value = value * np.array([[tql_edit_pose_meta0["width"], tql_edit_pose_meta0["height"], 1.0]])
                if not isinstance(value, list):
                    value = value.tolist()
            tql_edit_pose_meta0[key] = value
    
    if refer_edit_pose_meta is not None:
        for key, value in refer_edit_pose_meta.items():
            if type(value) is np.ndarray:
                if key in ['keypoints_left_hand', 'keypoints_right_hand']:
                    value = value * np.array([[refer_edit_pose_meta["width"], refer_edit_pose_meta["height"], 1.0]])
                if not isinstance(value, list):
                    value = value.tolist()
            refer_edit_pose_meta[key] = value

    retarget_tpl_pose_metas = retarget_pose(tpl_pose_meta0, refer_pose_meta, tpl_pose_metas_new, tql_edit_pose_meta0, refer_edit_pose_meta)

    pose_metas = []
    for meta in retarget_tpl_pose_metas:
        pose_meta = AAPoseMeta()
        width, height = meta["width"], meta["height"]
        pose_meta.width = width
        pose_meta.height = height
        pose_meta.kps_body = np.array(meta["keypoints_body"])[:, :2] * (width, height)
        pose_meta.kps_body_p = np.array(meta["keypoints_body"])[:, 2]

        kps_lhand = []
        kps_lhand_p = []
        for each_kps_lhand in meta["keypoints_left_hand"]:
            if each_kps_lhand is not None:
                kps_lhand.append([each_kps_lhand.x, each_kps_lhand.y])
                kps_lhand_p.append(each_kps_lhand.score)
            else:
                kps_lhand.append([None, None])
                kps_lhand_p.append(0.0)

        pose_meta.kps_lhand = np.array(kps_lhand)
        pose_meta.kps_lhand_p = np.array(kps_lhand_p)

        kps_rhand = []
        kps_rhand_p = []
        for each_kps_rhand in meta["keypoints_right_hand"]:
            if each_kps_rhand is not None:
                kps_rhand.append([each_kps_rhand.x, each_kps_rhand.y])
                kps_rhand_p.append(each_kps_rhand.score)
            else:
                kps_rhand.append([None, None])
                kps_rhand_p.append(0.0)

        pose_meta.kps_rhand = np.array(kps_rhand)
        pose_meta.kps_rhand_p = np.array(kps_rhand_p)

        pose_metas.append(pose_meta)

    return pose_metas


def calculate_hand_bone_length(hand_keypoints, width, height):
    """
    计算手部骨骼平均长度
    手部关键点0是手腕，1-20是手指关键点
    计算手腕到食指、中指、无名指指尖的平均距离作为手部尺寸参考
    
    手部关键点布局：
    - 点0: 手腕
    - 点8: 食指指尖
    - 点12: 中指指尖
    - 点16: 无名指指尖
    
    注意：hand_keypoints 应该是嵌套列表: [[x1, y1, score1], [x2, y2, score2], ...]
    坐标已经是归一化的（0-1范围）
    """
    if hand_keypoints is None or len(hand_keypoints) < 21:
        return None
    
    # 将关键点转换为像素坐标
    hand_kps = []
    for i in range(21):
        if i < len(hand_keypoints):
            kp = hand_keypoints[i]
            if kp is not None and len(kp) >= 3:
                x, y, score = kp[0], kp[1], kp[2]
                if score > 0.3:  # 置信度阈值
                    hand_kps.append([x * width, y * height, score])
                else:
                    hand_kps.append(None)
            else:
                hand_kps.append(None)
        else:
            hand_kps.append(None)
    
    # 手腕是第0个点
    if hand_kps[0] is None:
        return None
    
    wrist = hand_kps[0]
    
    # 计算手腕到食指(8)、中指(12)、无名指(16)指尖的距离
    finger_tips = [8, 12, 16]
    distances = []
    
    for tip_idx in finger_tips:
        if tip_idx < len(hand_kps) and hand_kps[tip_idx] is not None:
            dx = hand_kps[tip_idx][0] - wrist[0]
            dy = hand_kps[tip_idx][1] - wrist[1]
            dist = math.sqrt(dx * dx + dy * dy)
            distances.append(dist)
    
    if len(distances) == 0:
        return None
    
    # 返回最小距离作为手部尺寸（避免手指弯曲时的误差）
    return min(distances)


def calculate_hand_scale_ratio(src_skeleton, dst_skeleton):
    """
    根据手部关键点计算缩放系数
    左右手使用相同的缩放比例（取两者均值）
    """
    src_width = src_skeleton['width']
    src_height = src_skeleton['height']
    dst_width = dst_skeleton['width']
    dst_height = dst_skeleton['height']
    
    # 计算左手缩放系数
    src_left_hand_size = calculate_hand_bone_length(
        src_skeleton.get('keypoints_left_hand', []), src_width, src_height)
    dst_left_hand_size = calculate_hand_bone_length(
        dst_skeleton.get('keypoints_left_hand', []), dst_width, dst_height)
    
    if src_left_hand_size is not None and dst_left_hand_size is not None and src_left_hand_size > 0:
        left_ratio = dst_left_hand_size / src_left_hand_size
    else:
        left_ratio = None
    
    # 计算右手缩放系数
    src_right_hand_size = calculate_hand_bone_length(
        src_skeleton.get('keypoints_right_hand', []), src_width, src_height)
    dst_right_hand_size = calculate_hand_bone_length(
        dst_skeleton.get('keypoints_right_hand', []), dst_width, dst_height)
    
    if src_right_hand_size is not None and dst_right_hand_size is not None and src_right_hand_size > 0:
        right_ratio = dst_right_hand_size / src_right_hand_size
    else:
        right_ratio = None
    
    # 左右手使用相同的缩放比例（取均值）
    if left_ratio is not None and right_ratio is not None:
        avg_ratio = (left_ratio + right_ratio) / 2
        return {'left': avg_ratio, 'right': avg_ratio}
    elif left_ratio is not None:
        return {'left': left_ratio, 'right': left_ratio}
    elif right_ratio is not None:
        return {'left': right_ratio, 'right': right_ratio}
    else:
        return {'left': None, 'right': None}


def openpose_to_internal_format(openpose_data):
    """
    将 OpenPose JSON 格式转换为内部使用的格式
    自动过滤面部关键点（face_keypoints_2d）
    
    OpenPose 格式:
    {
        "people": [{
            "pose_keypoints_2d": [x1, y1, c1, x2, y2, c2, ...],  # 18个点 * 3 = 54个值
            "hand_left_keypoints_2d": [x1, y1, c1, ...],         # 21个点 * 3 = 63个值
            "hand_right_keypoints_2d": [x1, y1, c1, ...],        # 21个点 * 3 = 63个值
            "foot_keypoints_2d": [x1, y1, c1, ...],              # 6个点 * 3 = 18个值
            "face_keypoints_2d": [...]                            # 忽略
        }],
        "canvas_width": 1024,
        "canvas_height": 1024
    }
    
    内部格式:
    {
        "width": 1024,
        "height": 1024,
        "keypoints_body": [[x, y, score], ...],  # 20个点（归一化坐标 0-1）
        "keypoints_left_hand": [[x, y, score], ...],  # 21个点（归一化坐标 0-1）
        "keypoints_right_hand": [[x, y, score], ...],  # 21个点（归一化坐标 0-1）
        "keypoints_foot": [[x, y, score], ...]  # 6个点（归一化坐标 0-1）: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
    }
    """
    if not openpose_data or 'people' not in openpose_data or len(openpose_data['people']) == 0:
        return None
    
    person = openpose_data['people'][0]  # 只处理第一个人
    width = openpose_data.get('canvas_width', 1024)
    height = openpose_data.get('canvas_height', 1024)
    
    # 处理身体关键点（18个OpenPose点 + 2个脚趾点 = 20个点）
    body_kps_raw = person.get('pose_keypoints_2d', [])
    foot_kps_raw = person.get('foot_keypoints_2d', [])
    
    # 自动检测坐标是否需要归一化
    # 如果坐标的最大值 > 1.0，说明是像素坐标，需要归一化
    def need_normalize_coords(coords_flat, w, h):
        """检查坐标是否需要归一化"""
        for i in range(0, len(coords_flat), 3):
            if i + 1 < len(coords_flat):
                x, y = coords_flat[i], coords_flat[i + 1]
                if x > 1.0 or y > 1.0:
                    return True
        return False
    
    body_need_norm = need_normalize_coords(body_kps_raw, width, height)
    foot_need_norm = need_normalize_coords(foot_kps_raw, width, height)
    
    keypoints_body = []
    # OpenPose 18个身体关键点
    for i in range(18):
        if i * 3 + 2 < len(body_kps_raw):
            x = body_kps_raw[i * 3]
            y = body_kps_raw[i * 3 + 1]
            score = body_kps_raw[i * 3 + 2]
            # 只有当坐标是像素坐标时才归一化
            if body_need_norm:
                x = x / width
                y = y / height
            if score > 0:
                keypoints_body.append([x, y, score])
            else:
                keypoints_body.append(None)
        else:
            keypoints_body.append(None)
    
    # 添加脚趾关键点到 body（用于 limbSeq 连接）
    # OpenPose foot_keypoints_2d 顺序: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
    # 内部格式顺序: body_kps[18] = LToe (LBigToe), body_kps[19] = RToe (RBigToe)
    if len(foot_kps_raw) >= 18:
        # 左脚趾 (LBigToe, 索引 9, 10, 11) -> body_kps[18]
        left_toe_x = foot_kps_raw[9]
        left_toe_y = foot_kps_raw[10]
        left_toe_score = foot_kps_raw[11]
        if foot_need_norm:
            left_toe_x = left_toe_x / width
            left_toe_y = left_toe_y / height
        if left_toe_score > 0:
            keypoints_body.append([left_toe_x, left_toe_y, left_toe_score])
        else:
            keypoints_body.append(None)
        
        # 右脚趾 (RBigToe, 索引 0, 1, 2) -> body_kps[19]
        right_toe_x = foot_kps_raw[0]
        right_toe_y = foot_kps_raw[1]
        right_toe_score = foot_kps_raw[2]
        if foot_need_norm:
            right_toe_x = right_toe_x / width
            right_toe_y = right_toe_y / height
        if right_toe_score > 0:
            keypoints_body.append([right_toe_x, right_toe_y, right_toe_score])
        else:
            keypoints_body.append(None)
    else:
        keypoints_body.append(None)  # 左脚趾
        keypoints_body.append(None)  # 右脚趾
    
    # 单独存储完整的脚部关键点（6个点）用于输出
    # 顺序: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
    keypoints_foot = []
    if len(foot_kps_raw) >= 18:
        for i in range(6):
            x = foot_kps_raw[i * 3]
            y = foot_kps_raw[i * 3 + 1]
            score = foot_kps_raw[i * 3 + 2]
            if foot_need_norm:
                x = x / width
                y = y / height
            if score > 0:
                keypoints_foot.append([x, y, score])
            else:
                keypoints_foot.append([0, 0, 0])
    else:
        keypoints_foot = [[0, 0, 0]] * 6
    
    # 处理手部关键点 - 需要转换为嵌套列表格式 [[x,y,score], ...]
    # 注意：手部坐标保持像素坐标，不归一化！因为后续 retarget_pose 中会乘以 scale_min 进行缩放
    left_hand_raw = person.get('hand_left_keypoints_2d', [])
    right_hand_raw = person.get('hand_right_keypoints_2d', [])
    
    # 检查手部数据是否为空数组或无效
    # 空数组表示该手没有被检测到，应该标记为完全无效（score=-1）
    left_hand_is_empty = len(left_hand_raw) == 0
    right_hand_is_empty = len(right_hand_raw) == 0
    
    keypoints_left_hand = []
    for i in range(21):
        if left_hand_is_empty:
            # 原始数据为空，使用特殊标记表示无效（score=-1 区别于 score=0）
            keypoints_left_hand.append([0, 0, -1])
        elif i * 3 + 2 < len(left_hand_raw):
            keypoints_left_hand.append([
                left_hand_raw[i * 3],      # 像素坐标，不归一化
                left_hand_raw[i * 3 + 1],  # 像素坐标，不归一化
                left_hand_raw[i * 3 + 2]
            ])
        else:
            keypoints_left_hand.append([0, 0, 0])
    
    keypoints_right_hand = []
    for i in range(21):
        if right_hand_is_empty:
            # 原始数据为空，使用特殊标记表示无效（score=-1 区别于 score=0）
            keypoints_right_hand.append([0, 0, -1])
        elif i * 3 + 2 < len(right_hand_raw):
            keypoints_right_hand.append([
                right_hand_raw[i * 3],      # 像素坐标，不归一化
                right_hand_raw[i * 3 + 1],  # 像素坐标，不归一化
                right_hand_raw[i * 3 + 2]
            ])
        else:
            keypoints_right_hand.append([0, 0, 0])
    
    return {
        'width': width,
        'height': height,
        'keypoints_body': keypoints_body,
        'keypoints_left_hand': keypoints_left_hand,
        'keypoints_right_hand': keypoints_right_hand,
        'keypoints_foot': keypoints_foot  # 新增: 完整的6个脚部关键点
    }


def scale_and_center_pose(openpose_data, target_width, target_height):
    """
    将 OpenPose 数据缩放并居中到目标尺寸
    
    参数:
        openpose_data: OpenPose JSON 数据
        target_width: 目标宽度
        target_height: 目标高度
    
    返回:
        缩放并居中后的 OpenPose JSON 数据
    """
    if not openpose_data or 'people' not in openpose_data or len(openpose_data['people']) == 0:
        return openpose_data
    
    person = openpose_data['people'][0]
    
    # 收集所有有效的身体关键点坐标
    body_kps = person.get('pose_keypoints_2d', [])
    valid_points = []
    for i in range(0, len(body_kps), 3):
        if i+2 >= len(body_kps):
            break
        x, y, score = body_kps[i], body_kps[i+1], body_kps[i+2]
        if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
            valid_points.append([x, y])
    
    # 收集手部关键点
    for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
        hand_kps = person.get(hand_key, [])
        for i in range(0, len(hand_kps), 3):
            if i+2 >= len(hand_kps):
                break
            x, y, score = hand_kps[i], hand_kps[i+1], hand_kps[i+2]
            if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
                valid_points.append([x, y])
    
    # 收集脚部关键点
    foot_kps = person.get('foot_keypoints_2d', [])
    for i in range(0, len(foot_kps), 3):
        if i+2 >= len(foot_kps):
            break
        x, y, score = foot_kps[i], foot_kps[i+1], foot_kps[i+2]
        if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
            valid_points.append([x, y])
    
    if len(valid_points) == 0:
        # 没有有效点，直接返回
        openpose_data['canvas_width'] = target_width
        openpose_data['canvas_height'] = target_height
        return openpose_data
    
    # 计算边界框
    valid_points = np.array(valid_points)
    min_x, min_y = valid_points.min(axis=0)
    max_x, max_y = valid_points.max(axis=0)
    
    # 当前骨骼的宽高
    pose_width = max_x - min_x
    pose_height = max_y - min_y
    
    # 添加边距（保留10%的边距）
    margin_ratio = 0.1
    pose_width_with_margin = pose_width * (1 + 2 * margin_ratio)
    pose_height_with_margin = pose_height * (1 + 2 * margin_ratio)
    
    # 计算缩放比例，保持宽高比
    scale_x = target_width / pose_width_with_margin
    scale_y = target_height / pose_height_with_margin
    scale = min(scale_x, scale_y)
    
    # 计算缩放后的尺寸
    scaled_width = pose_width * scale
    scaled_height = pose_height * scale
    
    # 计算偏移量，使骨骼居中
    offset_x = (target_width - scaled_width) / 2 - min_x * scale
    offset_y = (target_height - scaled_height) / 2 - min_y * scale
    
    # 应用缩放和平移
    def transform_keypoints(kps_array):
        result = []
        for i in range(0, len(kps_array), 3):
            if i+2 >= len(kps_array):
                break
            x, y, score = kps_array[i], kps_array[i+1], kps_array[i+2]
            if x is not None and y is not None and score is not None and score > 0:
                new_x = x * scale + offset_x
                new_y = y * scale + offset_y
                result.extend([new_x, new_y, score])
            else:
                result.extend([0, 0, 0])
        return result
    
    # 转换所有关键点
    person['pose_keypoints_2d'] = transform_keypoints(person.get('pose_keypoints_2d', []))
    person['hand_left_keypoints_2d'] = transform_keypoints(person.get('hand_left_keypoints_2d', []))
    person['hand_right_keypoints_2d'] = transform_keypoints(person.get('hand_right_keypoints_2d', []))
    person['foot_keypoints_2d'] = transform_keypoints(person.get('foot_keypoints_2d', []))
    
    # 更新画布尺寸
    openpose_data['canvas_width'] = target_width
    openpose_data['canvas_height'] = target_height
    
    return openpose_data


def scale_to_match_target_pose(openpose_data, target_pose, target_width, target_height):
    """
    将 OpenPose 数据缩放并定位到与 target_pose 相同的位置
    保持输出骨骼的整体位置和构图与 target_pose 一致
    
    关键：所有计算都在归一化空间（0-1）中进行，最后再映射到目标画布
    
    参数:
        openpose_data: 需要变换的 OpenPose JSON 数据
        target_pose: 目标姿态的 OpenPose JSON 数据（作为位置参考）
        target_width: 目标画布宽度
        target_height: 目标画布高度
    
    返回:
        变换后的 OpenPose JSON 数据
    """
    if not openpose_data or 'people' not in openpose_data or len(openpose_data['people']) == 0:
        openpose_data['canvas_width'] = target_width
        openpose_data['canvas_height'] = target_height
        return openpose_data
    
    if not target_pose or 'people' not in target_pose or len(target_pose['people']) == 0:
        # 如果没有 target_pose，退化为居中模式
        return scale_and_center_pose(openpose_data, target_width, target_height)
    
    person = openpose_data['people'][0]
    target_person = target_pose['people'][0]
    
    # 获取源数据的画布尺寸
    src_canvas_w = openpose_data.get('canvas_width', target_width)
    src_canvas_h = openpose_data.get('canvas_height', target_height)
    
    # 获取 target_pose 的画布尺寸
    target_canvas_w = target_pose.get('canvas_width', target_width)
    target_canvas_h = target_pose.get('canvas_height', target_height)
    
    # 辅助函数：检测坐标是否已经是归一化的（0-1范围）
    def is_normalized_coords(kps_array):
        """检查坐标是否已经是归一化的"""
        for i in range(0, len(kps_array), 3):
            if i+2 >= len(kps_array):
                break
            x, y, score = kps_array[i], kps_array[i+1], kps_array[i+2]
            if x is not None and y is not None and score is not None and score > 0:
                if x > 1.0 or y > 1.0:
                    return False
        return True
    
    # 检测源数据和目标数据的坐标格式
    src_body_kps = person.get('pose_keypoints_2d', [])
    target_body_kps = target_person.get('pose_keypoints_2d', [])
    
    src_is_normalized = is_normalized_coords(src_body_kps)
    target_is_normalized = is_normalized_coords(target_body_kps)
    
    # 辅助函数：收集有效关键点并返回归一化坐标
    def collect_valid_points_normalized(person_data, canvas_w, canvas_h, is_normalized):
        points = []
        body_kps = person_data.get('pose_keypoints_2d', [])
        for i in range(0, len(body_kps), 3):
            if i+2 >= len(body_kps):
                break
            x, y, score = body_kps[i], body_kps[i+1], body_kps[i+2]
            if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
                if is_normalized:
                    points.append([x, y])
                else:
                    points.append([x / canvas_w, y / canvas_h])
        
        for hand_key in ['hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            hand_kps = person_data.get(hand_key, [])
            hand_is_norm = is_normalized_coords(hand_kps) if hand_kps else True
            for i in range(0, len(hand_kps), 3):
                if i+2 >= len(hand_kps):
                    break
                x, y, score = hand_kps[i], hand_kps[i+1], hand_kps[i+2]
                if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
                    if hand_is_norm:
                        points.append([x, y])
                    else:
                        points.append([x / canvas_w, y / canvas_h])
        
        foot_kps = person_data.get('foot_keypoints_2d', [])
        foot_is_norm = is_normalized_coords(foot_kps) if foot_kps else True
        for i in range(0, len(foot_kps), 3):
            if i+2 >= len(foot_kps):
                break
            x, y, score = foot_kps[i], foot_kps[i+1], foot_kps[i+2]
            if x is not None and y is not None and score is not None and score > 0 and x > 0 and y > 0:
                if foot_is_norm:
                    points.append([x, y])
                else:
                    points.append([x / canvas_w, y / canvas_h])
        
        return points
    
    # 收集归一化后的关键点
    src_points_norm = collect_valid_points_normalized(person, src_canvas_w, src_canvas_h, src_is_normalized)
    target_points_norm = collect_valid_points_normalized(target_person, target_canvas_w, target_canvas_h, target_is_normalized)
    
    if len(src_points_norm) == 0 or len(target_points_norm) == 0:
        openpose_data['canvas_width'] = target_width
        openpose_data['canvas_height'] = target_height
        return openpose_data
    
    src_points_norm = np.array(src_points_norm)
    target_points_norm = np.array(target_points_norm)
    
    # 在归一化空间中计算源骨骼的边界框和中心
    src_min_norm = src_points_norm.min(axis=0)
    src_max_norm = src_points_norm.max(axis=0)
    src_center_norm = (src_min_norm + src_max_norm) / 2
    src_size_norm = src_max_norm - src_min_norm
    
    # 在归一化空间中计算目标骨骼的边界框和中心
    target_min_norm = target_points_norm.min(axis=0)
    target_max_norm = target_points_norm.max(axis=0)
    target_center_norm = (target_min_norm + target_max_norm) / 2
    target_size_norm = target_max_norm - target_min_norm
    
    # 在归一化空间中计算缩放比例
    if src_size_norm[0] > 0 and src_size_norm[1] > 0:
        scale_x = target_size_norm[0] / src_size_norm[0]
        scale_y = target_size_norm[1] / src_size_norm[1]
        scale_norm = min(scale_x, scale_y)  # 保持宽高比
    else:
        scale_norm = 1.0
    
    # 变换函数：在归一化空间中进行缩放和偏移，然后映射到目标画布
    def transform_keypoints(kps_array, kps_is_normalized, canvas_w, canvas_h):
        result = []
        for i in range(0, len(kps_array), 3):
            if i+2 >= len(kps_array):
                break
            x, y, score = kps_array[i], kps_array[i+1], kps_array[i+2]
            if x is not None and y is not None and score is not None and score > 0:
                # 1. 归一化到 0-1 范围（如果还不是）
                if kps_is_normalized:
                    x_norm = x
                    y_norm = y
                else:
                    x_norm = x / canvas_w
                    y_norm = y / canvas_h
                
                # 2. 计算相对于源中心的偏移（归一化空间）
                dx = x_norm - src_center_norm[0]
                dy = y_norm - src_center_norm[1]
                
                # 3. 缩放偏移
                dx_scaled = dx * scale_norm
                dy_scaled = dy * scale_norm
                
                # 4. 加上目标中心（归一化空间）
                new_x_norm = target_center_norm[0] + dx_scaled
                new_y_norm = target_center_norm[1] + dy_scaled
                
                # 5. 映射到目标画布尺寸（输出像素坐标）
                new_x = new_x_norm * target_width
                new_y = new_y_norm * target_height
                
                result.extend([new_x, new_y, score])
            else:
                result.extend([0, 0, 0])
        return result
    
    # 转换所有关键点
    person['pose_keypoints_2d'] = transform_keypoints(
        person.get('pose_keypoints_2d', []), src_is_normalized, src_canvas_w, src_canvas_h)
    
    # 手部坐标可能有不同的归一化状态
    hand_left_kps = person.get('hand_left_keypoints_2d', [])
    hand_left_is_norm = is_normalized_coords(hand_left_kps) if hand_left_kps else True
    person['hand_left_keypoints_2d'] = transform_keypoints(
        hand_left_kps, hand_left_is_norm, src_canvas_w, src_canvas_h)
    
    hand_right_kps = person.get('hand_right_keypoints_2d', [])
    hand_right_is_norm = is_normalized_coords(hand_right_kps) if hand_right_kps else True
    person['hand_right_keypoints_2d'] = transform_keypoints(
        hand_right_kps, hand_right_is_norm, src_canvas_w, src_canvas_h)
    
    foot_kps = person.get('foot_keypoints_2d', [])
    foot_is_norm = is_normalized_coords(foot_kps) if foot_kps else True
    person['foot_keypoints_2d'] = transform_keypoints(
        foot_kps, foot_is_norm, src_canvas_w, src_canvas_h)
    
    # 更新画布尺寸
    openpose_data['canvas_width'] = target_width
    openpose_data['canvas_height'] = target_height
    
    return openpose_data


def retarget_pose_main(ref_pose_json, target_pose_json, video_poses_json, target_width=512, target_height=512, threshold=0.4):
    """
    主入口函数，处理 OpenPose JSON 格式的骨骼重映射
    
    参数:
        ref_pose_json: 参考视频某一帧的 OpenPose JSON 数据
        target_pose_json: 目标图像的 OpenPose JSON 数据
        video_poses_json: 参考视频所有帧的 OpenPose JSON 数据列表
        target_width: 目标输出宽度
        target_height: 目标输出高度
        threshold: 关键点置信度阈值
    
    返回:
        重映射后的所有帧 OpenPose JSON 数据列表
    """
    
    # 转换格式
    ref_skeleton = openpose_to_internal_format(ref_pose_json)
    target_skeleton = openpose_to_internal_format(target_pose_json)
    
    if ref_skeleton is None:
        return []
    
    if target_skeleton is None:
        return []
    
    all_skeletons = []
    for i, frame_data in enumerate(video_poses_json):
        skeleton = openpose_to_internal_format(frame_data)
        if skeleton is not None:
            all_skeletons.append(skeleton)
    
    if len(all_skeletons) == 0:
        return []
    
    # 调用原有的 retarget_pose 函数
    retargeted_skeletons = retarget_pose(
        ref_skeleton, 
        target_skeleton, 
        all_skeletons,
        None,  # src_skeleton_edit
        None,  # dst_skeleton_edit
        threshold=threshold
    )
    
    result = []
    for skeleton in retargeted_skeletons:
        openpose_frame = internal_to_openpose_format(skeleton)
        # 缩放并定位到与 target_pose 相同的位置（保持构图一致）
        openpose_frame = scale_to_match_target_pose(openpose_frame, target_pose_json, target_width, target_height)
        result.append(openpose_frame)
    
    return result


def internal_to_openpose_format(skeleton):
    """
    将内部格式转换回 OpenPose JSON 格式
    """
    width = skeleton.get('width', 1024)
    height = skeleton.get('height', 1024)
    
    # 转换身体关键点 - 确保输出18个点（54个值）
    # 注意：内部格式的 keypoints_body 是归一化坐标（0-1范围），需要转换为像素坐标
    body_kps = skeleton.get('keypoints_body', [])
    pose_keypoints_2d = []
    for i in range(18):  # OpenPose 标准18个点
        if i < len(body_kps) and body_kps[i] is not None and len(body_kps[i]) >= 3:
            # 将归一化坐标转换为像素坐标
            x = body_kps[i][0] * width
            y = body_kps[i][1] * height
            score = body_kps[i][2]
            pose_keypoints_2d.extend([x, y, score])
        else:
            pose_keypoints_2d.extend([0, 0, 0])
    
    # 转换脚部关键点 - OpenPose 格式: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
    # 优先使用 keypoints_foot（完整的6个脚部关键点）
    keypoints_foot = skeleton.get('keypoints_foot', None)
    foot_keypoints_2d = []
    
    if keypoints_foot is not None and len(keypoints_foot) >= 6:
        # 使用完整的脚部关键点
        # keypoints_foot 顺序: [RBigToe, RSmallToe, RHeel, LBigToe, LSmallToe, LHeel]
        for i in range(6):
            if keypoints_foot[i] is not None and len(keypoints_foot[i]) >= 3 and keypoints_foot[i][2] > 0:
                x = keypoints_foot[i][0] * width
                y = keypoints_foot[i][1] * height
                score = keypoints_foot[i][2]
                foot_keypoints_2d.extend([x, y, score])
            else:
                foot_keypoints_2d.extend([0, 0, 0])
    elif len(body_kps) >= 20:
        # 降级方案：只有 BigToe，使用 body_kps[18] (LToe) 和 body_kps[19] (RToe)
        # 右脚趾 (body_kps[19] = RToe = RBigToe)
        if body_kps[19] is not None and len(body_kps[19]) >= 3:
            right_toe = body_kps[19]
            foot_keypoints_2d.extend([right_toe[0] * width, right_toe[1] * height, right_toe[2]])
        else:
            foot_keypoints_2d.extend([0, 0, 0])
        
        # 右脚其他点（用0填充: RSmallToe, RHeel）
        foot_keypoints_2d.extend([0] * 6)
        
        # 左脚趾 (body_kps[18] = LToe = LBigToe)
        if body_kps[18] is not None and len(body_kps[18]) >= 3:
            left_toe = body_kps[18]
            foot_keypoints_2d.extend([left_toe[0] * width, left_toe[1] * height, left_toe[2]])
        else:
            foot_keypoints_2d.extend([0, 0, 0])
        
        # 左脚其他点（用0填充: LSmallToe, LHeel）
        foot_keypoints_2d.extend([0] * 6)
    else:
        # 如果没有足够的身体关键点，脚部关键点全部设为0
        foot_keypoints_2d = [0] * 18
    
    # 转换手部关键点
    left_hand = skeleton.get('keypoints_left_hand', [])
    right_hand = skeleton.get('keypoints_right_hand', [])
    
    # 处理左手 - Keypoint 中的 x, y 已经是像素坐标（经过 get_handpose_meta 处理）
    hand_left_keypoints_2d = []
    if isinstance(left_hand, list) and len(left_hand) > 0:
        # 检查列表中是否有任何有效的非 None 元素
        first_valid = None
        for item in left_hand:
            if item is not None:
                first_valid = item
                break
        
        if first_valid is not None and hasattr(first_valid, 'x'):  # Keypoint NamedTuple
            for kp in left_hand:
                if kp is not None and kp.score > 0:
                    # Keypoint 中的坐标是像素坐标，直接使用
                    hand_left_keypoints_2d.extend([kp.x, kp.y, kp.score])
                else:
                    hand_left_keypoints_2d.extend([0, 0, 0])
        elif first_valid is not None and isinstance(first_valid, (list, tuple)):  # 嵌套列表格式 [[x,y,score], ...]
            for kp in left_hand:
                if kp is not None and len(kp) >= 3 and kp[2] > 0:  # score > 0
                    hand_left_keypoints_2d.extend([kp[0], kp[1], kp[2]])
                else:
                    hand_left_keypoints_2d.extend([0, 0, 0])
        elif first_valid is None:
            # 全是 None，输出全0
            hand_left_keypoints_2d = [0, 0, 0] * 21
        else:  # 已经是 flat list 格式 [x,y,score,x,y,score,...] 或其他格式
            if isinstance(left_hand, list):
                # 过滤掉任何 None 值
                for item in left_hand:
                    if item is not None:
                        hand_left_keypoints_2d.append(item)
                    else:
                        hand_left_keypoints_2d.append(0)
    
    # 处理右手 - Keypoint 中的 x, y 已经是像素坐标（经过 get_handpose_meta 处理）
    hand_right_keypoints_2d = []
    if isinstance(right_hand, list) and len(right_hand) > 0:
        # 检查列表中是否有任何有效的非 None 元素
        first_valid = None
        for item in right_hand:
            if item is not None:
                first_valid = item
                break
        
        if first_valid is not None and hasattr(first_valid, 'x'):  # Keypoint NamedTuple
            for kp in right_hand:
                if kp is not None and kp.score > 0:
                    # Keypoint 中的坐标是像素坐标，直接使用
                    hand_right_keypoints_2d.extend([kp.x, kp.y, kp.score])
                else:
                    hand_right_keypoints_2d.extend([0, 0, 0])
        elif first_valid is not None and isinstance(first_valid, (list, tuple)):  # 嵌套列表格式 [[x,y,score], ...]
            for kp in right_hand:
                if kp is not None and len(kp) >= 3 and kp[2] > 0:  # score > 0
                    hand_right_keypoints_2d.extend([kp[0], kp[1], kp[2]])
                else:
                    hand_right_keypoints_2d.extend([0, 0, 0])
        elif first_valid is None:
            # 全是 None，输出全0
            hand_right_keypoints_2d = [0, 0, 0] * 21
        else:  # 已经是 flat list 格式 [x,y,score,x,y,score,...] 或其他格式
            if isinstance(right_hand, list):
                # 过滤掉任何 None 值
                for item in right_hand:
                    if item is not None:
                        hand_right_keypoints_2d.append(item)
                    else:
                        hand_right_keypoints_2d.append(0)
    
    # 填充到63个值（21个点 * 3）
    while len(hand_left_keypoints_2d) < 63:
        hand_left_keypoints_2d.extend([0, 0, 0])
    while len(hand_right_keypoints_2d) < 63:
        hand_right_keypoints_2d.extend([0, 0, 0])
    
    return {
        "people": [{
            "pose_keypoints_2d": pose_keypoints_2d,
            "foot_keypoints_2d": foot_keypoints_2d,
            "face_keypoints_2d": [],  # 始终为空，不包含面部关键点
            "hand_right_keypoints_2d": hand_right_keypoints_2d,
            "hand_left_keypoints_2d": hand_left_keypoints_2d
        }],
        "canvas_width": int(width),
        "canvas_height": int(height)
    }
