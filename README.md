# SDPose Retarget for ComfyUI

This custom node performs pose retargeting for SDPose-OOD data. It takes a single-frame reference pose, a target pose, and a video pose sequence as input. The node calculates scaling factors for the body and hands independently and retargets the entire video motion to match the target character's skeleton. It also includes a `SDPose Draw` node to visualize the retargeted OpenPose JSON data as image sequences.

本 ComfyUI 节点用于 SDPose-OOD 数据的骨骼重映射（Retargeting）。接收参考姿态单帧、目标姿态和视频姿态序列作为输入，独立计算身体和手部的缩放系数，将参考视频的动作迁移到目标角色的骨骼上。同时包含 `SDPose Draw` 节点，可将重映射后的 OpenPose JSON 数据可视化为骨骼图像序列。