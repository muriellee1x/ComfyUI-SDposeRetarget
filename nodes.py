"""
SDpose Retarget Node
用于将SDPose-OOD格式的骨骼动画重映射到目标骨骼
"""

import json
import numpy as np
import cv2
import math
import torch
from .retarget_pose import retarget_pose_main
from .pose_draw import batch_draw_pose

class SDposeRetarget:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_pose": ("POSE_KEYPOINT", {
                    "tooltip": "参考视频某一帧的pose data (OpenPose JSON格式)"
                }),
                "target_pose": ("POSE_KEYPOINT", {
                    "tooltip": "目标图像的pose data (OpenPose JSON格式)"
                }),
                "video_poses": ("POSE_KEYPOINT", {
                    "tooltip": "参考视频所有帧的pose data (OpenPose JSON格式)"
                }),
            },
             "optional": {
                "threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "关键点置信度阈值"
                }),
            }
        }
    
    RETURN_TYPES = ("POSE_KEYPOINT", "STRING",)
    RETURN_NAMES = ("pose_keypoint", "pose_keypoint_json",)
    FUNCTION = "retarget"
    CATEGORY = "SDposeRetarget"
    DESCRIPTION = "将视频骨骼动画重映射到目标骨骼。输入为OpenPose格式的Pose Data。自动计算身体和手部的缩放系数。"
    
    def retarget(self, ref_pose, target_pose, video_poses, threshold=0.4):
        # Helper to ensure input is a list of dicts or dict
        def parse_input(data):
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON input")
                    return []
            return data

        ref_data = parse_input(ref_pose)
        target_data = parse_input(target_pose)
        video_data = parse_input(video_poses)

        # Call the main retargeting logic
        retargeted_result = retarget_pose_main(ref_data, target_data, video_data, threshold=threshold)
        
        return (retargeted_result, json.dumps(retargeted_result))

class SDPoseDraw:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT", {
                    "tooltip": "SDpose Retarget输出的Pose Data"
                }),
            },
             "optional": {
                "threshold": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "绘制时的置信度阈值"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw"
    CATEGORY = "SDposeRetarget"
    DESCRIPTION = "绘制OpenPose骨骼图。支持Body(17点)、Hand和Feet。"
    
    def draw(self, pose_keypoint, threshold=0.4):
        # Ensure input is parsed
        if isinstance(pose_keypoint, str):
            try:
                pose_keypoint = json.loads(pose_keypoint)
            except json.JSONDecodeError:
                # Return empty image if parse fails
                return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

        images = batch_draw_pose(pose_keypoint, threshold=threshold)
        return (images,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SDposeRetarget": SDposeRetarget,
    "SDPoseDraw": SDPoseDraw,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDposeRetarget": "SDpose Retarget",
    "SDPoseDraw": "SDPose Draw OpenPose",
}
