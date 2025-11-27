# SDPose Retarget for ComfyUI

基于 SDpose-OOD 模型的骨骼动画重映射节点，支持全身/半身视频动画迁移到任意姿势的角色参考图。

A skeleton animation retargeting node based on SDpose-OOD model, supporting full/half-body video animation transfer to character reference images with arbitrary poses.

---

## 节点说明 / Node Documentation

| 序号 | 节点 | 说明 |
|:---:|------|------|
| 1 | [SDpose Retarget](#1-sdpose-retarget) | 核心重映射节点，将参考视频的动作迁移到目标角色骨骼上 |
| 2 | [SDPose Draw OpenPose](#2-sdpose-draw-openpose) | 骨骼绘制节点，将姿态数据渲染为可视化骨骼图 |
| 3 | [Load Video Frame](#3-load-video-frame) | 帧提取节点，从图像序列中提取指定帧 |
| 4 | [Image to Contiguous](#4-image-to-contiguous-fix-opencv) | 图像内存修复节点，将图像内存转为连续内存格式 |

---

### 1. SDpose Retarget

**核心重映射节点，将参考视频的动作迁移到目标角色骨骼上。**

#### 输入 / Input

| 参数 | 说明 |
|------|------|
| `ref_pose` | 参考视频中角色的姿态数据（只需要一帧，最好是 Tpose 数据） |
| `target_pose` | 目标角色的姿态数据 |
| `video_poses` | 参考视频的全部帧的姿态序列 |
| `target_width/height` | 输出画布尺寸 |
| `threshold` | 骨骼置信度过滤 |

#### 输出 / Output

| 输出 | 说明 |
|------|------|
| `pose_keypoint` | 重映射后的骨骼数据 |

#### 处理逻辑 / Processing Logic

1. 自动计算身体、手部、脚部的独立缩放系数
2. 自动匹配原始目标角色的构图
3. 过滤面部关键点，仅计算其他骨骼

---

### 2. SDPose Draw OpenPose

**骨骼绘制节点，将姿态数据渲染为可视化骨骼图。**

#### 输入 / Input

| 参数 | 说明 |
|------|------|
| `pose_keypoint` | 传入骨骼数据 |
| `threshold` | 骨骼置信度过滤 |
| `bone_thickness` | 绘制骨骼的线条粗细 |
| `align_to_bottom` | 将骨骼对齐画面底部（适用于半身视频） |

---

### 3. Load Video Frame

**帧提取节点，从图像序列中提取指定帧。**

#### 输入 / Input

| 参数 | 说明 |
|------|------|
| `frame_index` | 希望提取的帧数索引（从 0 开始） |

---

### 4. Image to Contiguous (Fix OpenCV)

**图像内存修复节点，将图像内存转为连续内存格式。**

此节点主要是为了解决 SDPose Estimation 节点的 OpenCV 报错问题，接在 SDPose Estimation 节点之前。

#### 常见错误 / Common Error

```
OpenCV error: (-5:Bad argument) in function 'fillConvexPoly'
Layout of the output array img is incompatible with cv::Mat
```

#### 使用方法 / Usage

```
[生成图片节点] → [Image To Contiguous] → [SDPose Estimation]
```

---

## 输入格式 / Input Format

支持标准 OpenPose JSON 格式，包含 `pose_keypoints_2d`、`hand_left/right_keypoints_2d`、`foot_keypoints_2d`。

Supports standard OpenPose JSON format with body, hand, and foot keypoints.
