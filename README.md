# SDPose Retarget for ComfyUI

基于 SDpose-OOD 模型的骨骼动画重映射节点，支持全身/半身视频动画迁移到任意姿势的角色参考图。

A skeleton animation retargeting node based on SDpose-OOD model, supporting full/half-body video animation transfer to character reference images with arbitrary poses.

---

## 核心节点 / Core Nodes

### 1. SDpose Retarget
**核心重映射节点** - 将参考视频的动作迁移到目标角色。

| 输入 Input | 说明 Description |
|------------|------------------|
| `ref_pose` | 参考视频关键帧的姿态数据 / Reference video keyframe pose |
| `target_pose` | 目标角色的姿态数据 / Target character pose |
| `video_poses` | 参考视频全部帧的姿态序列 / All frames of reference video |
| `target_width/height` | 输出画布尺寸 / Output canvas size |

**处理逻辑 / Logic:**
- 自动计算身体、手部、脚部的独立缩放系数
- 智能匹配骨骼位置，保持目标角色构图
- 自动过滤面部关键点，仅保留有效骨骼数据

### 2. SDPose Draw OpenPose
**骨骼绘制节点** - 将姿态数据渲染为可视化骨骼图。

| 参数 Parameter | 说明 Description |
|----------------|------------------|
| `bone_thickness` | 骨骼线条粗细 / Bone line thickness |
| `align_to_bottom` | 底部对齐（稳定脚底位置）/ Align to bottom (stabilize feet) |

### 3. Load Video Frame
**帧提取节点** - 从图像序列中提取指定帧。

---

## 输入格式 / Input Format

支持标准 OpenPose JSON 格式，包含 `pose_keypoints_2d`、`hand_left/right_keypoints_2d`、`foot_keypoints_2d`。

Supports standard OpenPose JSON format with body, hand, and foot keypoints.

---

## 工作流程 / Workflow

```
参考视频 → DWPose检测 → SDpose Retarget → SDPose Draw → ControlNet
Reference Video → DWPose → SDpose Retarget → SDPose Draw → ControlNet
```
