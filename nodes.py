"""
SDpose Retarget Node
用于将SDPose-OOD格式的骨骼动画重映射到目标骨骼
"""

import json
import numpy as np
import cv2
import math
import torch


def validate_openpose_format(data, name="data"):
    """验证 OpenPose 格式数据的有效性"""
    if not isinstance(data, dict):
        return False
    
    if 'people' not in data:
        return False
    
    if not isinstance(data['people'], list) or len(data['people']) == 0:
        return False
    
    person = data['people'][0]
    required_fields = ['pose_keypoints_2d']
    for field in required_fields:
        if field not in person:
            return False
    
    return True
try:
    from .retarget_pose import retarget_pose_main
except ImportError:
    print("警告：无法导入 retarget_pose_main，请检查依赖")
    retarget_pose_main = None
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
                "target_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "目标输出宽度"
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "目标输出高度"
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
    DESCRIPTION = "将视频骨骼动画重映射到目标骨骼。输入为OpenPose格式的Pose Data。自动计算身体和手部的缩放系数，并缩放到目标尺寸。"
    
    def retarget(self, ref_pose, target_pose, video_poses, target_width, target_height, threshold=0.4):
        if retarget_pose_main is None:
            raise RuntimeError("retarget_pose_main 函数未正确导入，请检查依赖安装")
        
        # Helper to ensure input is parsed correctly
        def parse_input(data, param_name):
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return None
            return data

        ref_data = parse_input(ref_pose, "ref_pose")
        target_data = parse_input(target_pose, "target_pose")
        video_data = parse_input(video_poses, "video_poses")

        # 验证输入数据
        if not ref_data:
            raise ValueError("参考帧 pose data 为空或格式错误")
        if not target_data:
            raise ValueError("目标 pose data 为空或格式错误")
        if not video_data:
            raise ValueError("视频 pose data 为空或格式错误")
        
        # 处理 ref_pose - 如果是列表，取第一帧
        if isinstance(ref_data, list):
            if len(ref_data) == 0:
                raise ValueError("ref_pose 列表为空")
            ref_data = ref_data[0]
        elif not isinstance(ref_data, dict):
            raise ValueError(f"ref_pose 格式错误，期望字典或列表，得到: {type(ref_data)}")
        
        # 处理 target_pose - 如果是列表，取第一帧
        if isinstance(target_data, list):
            if len(target_data) == 0:
                raise ValueError("target_pose 列表为空")
            target_data = target_data[0]
        elif not isinstance(target_data, dict):
            raise ValueError(f"target_pose 格式错误，期望字典或列表，得到: {type(target_data)}")
        
        # 处理 video_poses - 确保是列表格式
        if isinstance(video_data, dict):
            video_data = [video_data]
        elif not isinstance(video_data, list):
            raise ValueError(f"video_poses 格式错误，期望列表或字典，得到: {type(video_data)}")
        
        # 验证 OpenPose 格式
        validate_openpose_format(ref_data, "ref_pose")
        validate_openpose_format(target_data, "target_pose")

        # Call the main retargeting logic
        try:
            retargeted_result = retarget_pose_main(ref_data, target_data, video_data, 
                                                   target_width=target_width, 
                                                   target_height=target_height,
                                                   threshold=threshold)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        
        if not retargeted_result:
            return ([], "[]")
        
        return (retargeted_result, json.dumps(retargeted_result, ensure_ascii=False))

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
                "bone_thickness": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "骨骼线条粗细"
                }),
                "align_to_bottom": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "将所有帧的骨骼最低点对齐到画布底部"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw"
    CATEGORY = "SDposeRetarget"
    DESCRIPTION = "绘制OpenPose骨骼图。支持Body(17点)、Hand和Feet。可调整骨骼粗细，支持底部对齐。"
    
    def draw(self, pose_keypoint, threshold=0.4, bone_thickness=4, align_to_bottom=False):
        # Ensure input is parsed
        if isinstance(pose_keypoint, str):
            try:
                pose_keypoint = json.loads(pose_keypoint)
            except json.JSONDecodeError:
                # Return empty image if parse fails
                return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

        images = batch_draw_pose(pose_keypoint, threshold=threshold, stickwidth=bone_thickness, align_to_bottom=align_to_bottom)
        return (images,)

class LoadVideoFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "输入图像序列"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1, "tooltip": "要提取的帧索引（从0开始）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING",)
    RETURN_NAMES = ("image", "total_frames", "info",)
    FUNCTION = "load_frame"
    CATEGORY = "SDposeRetarget"
    DESCRIPTION = "从图像序列中选择并输出指定索引的帧。输入为图像序列，输出为单张图像。"

    def load_frame(self, images, frame_index):
        # images shape: (B, H, W, C) 其中B是总帧数
        B, H, W, C = images.shape
        total_frames = B
        
        # 检查帧索引是否有效
        if frame_index >= total_frames:
            frame_index = total_frames - 1
        
        if frame_index < 0:
            frame_index = 0
        
        # 提取指定索引的帧
        selected_frame = images[frame_index:frame_index+1]  # 保持batch维度
        
        # 生成信息字符串
        info = (f"图像序列信息:\n"
               f"总帧数: {total_frames}\n"
               f"分辨率: {W}x{H}\n"
               f"当前帧: {frame_index}\n"
               f"输出图像形状: {selected_frame.shape}")
        
        return (selected_frame, total_frames, info)

class ImageToContiguous:
    """
    将图像转换为内存连续的格式
    解决某些节点（如SDPose）在处理生成图片时因内存不连续导致的OpenCV报错
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "输入图像"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "to_contiguous"
    CATEGORY = "SDposeRetarget"
    DESCRIPTION = "将图像转换为内存连续格式。用于解决生成图片连接到某些节点（如SDPose estimation）时的OpenCV报错问题。"

    def to_contiguous(self, images):
        # 方法1：使用 torch.contiguous()
        if not images.is_contiguous():
            images = images.contiguous()
        
        # 方法2：额外保险 - 转numpy再转回来确保C-contiguous
        # 这可以解决一些更深层的内存布局问题
        images_np = images.cpu().numpy()
        images_np = np.ascontiguousarray(images_np)
        images = torch.from_numpy(images_np).to(images.device)
        
        return (images,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "SDposeRetarget": SDposeRetarget,
    "SDPoseDraw": SDPoseDraw,
    "LoadVideoFrame": LoadVideoFrame,
    "ImageToContiguous": ImageToContiguous,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDposeRetarget": "SDpose Retarget",
    "SDPoseDraw": "SDPose Draw OpenPose",
    "LoadVideoFrame": "Load Video Frame",
    "ImageToContiguous": "Image To Contiguous (Fix OpenCV)",
}
