# MotionFix 数据格式文档

## 目录
- [MotionFix 数据格式文档](#motionfix-数据格式文档)
  - [目录](#目录)
  - [1. 原始加载数据 `dataset_dict_raw`](#1-原始加载数据-dataset_dict_raw)
  - [2. 转换后的数据 `data_dict`](#2-转换后的数据-data_dict)
  - [3. 数据集初始化参数 `data`](#3-数据集初始化参数-data)
  - [3+. SMPL 参数与特征关系](#3-smpl-参数与特征关系)
  - [4. 统计数据 `self.stats`](#4-统计数据-selfstats)
  - [5. 单个样本数据 (`__getitem__()` 返回)](#5-单个样本数据-__getitem__-返回)
  - [6. 批次数据 (DataLoader 输出)](#6-批次数据-dataloader-输出)
  - [7. 模型输入数据 (`train_diffusion_forward` 接收)](#7-模型输入数据-train_diffusion_forward-接收)
  - [8. 特征维度列表 `self.nfeats`](#8-特征维度列表-selfnfeats)
  - [9. ROTS 数据分解](#9-rots-数据分解)
  - [10. JOINT\_POSITIONS 数据分解](#10-joint_positions-数据分解)
  - [11. 数据流示意图](#11-数据流示意图)
  - [12. 数据格式总结表](#12-数据格式总结表)
    - [原始数据格式](#原始数据格式)
    - [提取的特征格式](#提取的特征格式)
    - [批次数据格式](#批次数据格式)
    - [模型输入数据格式（train\_diffusion\_forward）](#模型输入数据格式train_diffusion_forward)
    - [其他数据结构](#其他数据结构)
  - [附录：常用配置](#附录常用配置)
    - [典型的 load\_feats 配置](#典型的-load_feats-配置)
    - [旋转表示转换](#旋转表示转换)
- [1. 总览：项目数据流（全程大图）](#1-总览项目数据流全程大图)
- [2. 详解每个阶段的数据**格式与变化**](#2-详解每个阶段的数据格式与变化)
  - [★ ① 原始文件/磁盘](#--原始文件磁盘)
  - [★ ② 转换为PyTorch张量](#--转换为pytorch张量)
  - [★ ③ 数据集划分（MotionFixDataset）](#--数据集划分motionfixdataset)
    - [采样（FrameSampler，当前未启用）](#采样framesampler当前未启用)
    - [单条样本输出](#单条样本输出)
  - [★ ④ 批次构建/归一化/合并](#--批次构建归一化合并)
    - [collate\_fn](#collate_fn)
    - [allsplit\_step + norm\_and\_cat (重点特征归一化+拼接)](#allsplit_step--norm_and_cat-重点特征归一化拼接)
  - [★ ⑤ 训练/推理阶段（扩散空间）](#--训练推理阶段扩散空间)
  - [★ ⑥ 动作重建与特征还原（diffout2motion）](#--动作重建与特征还原diffout2motion)
- [3. 各阶段对应数据结构对照表](#3-各阶段对应数据结构对照表)
- [4. 典型输入输出 **格式变化路径梳理**](#4-典型输入输出-格式变化路径梳理)
- [5. 项目代码部分主要函数与流程配合解释](#5-项目代码部分主要函数与流程配合解释)
- [6. 相关文档链接组合](#6-相关文档链接组合)
- [7. 直观流程小结（精华版）](#7-直观流程小结精华版)

---

## 1. 原始加载数据 `dataset_dict_raw`

从 `joblib.load()` 加载的字典结构（numpy arrays）：

```python
dataset_dict_raw = {
    'sample_id': {
        'motion_source': {
            'rots': np.ndarray,              # [T1, 66] - 22个关节的axis-angle旋转
            'trans': np.ndarray,             # [T1, 3] - 骨盆全局平移(x,y,z)
            'joint_positions': np.ndarray    # [T1, 22, 3] - 22个关节的3D位置
        },
        'motion_target': {
            'rots': np.ndarray,              # [T2, 66]
            'trans': np.ndarray,             # [T2, 3]
            'joint_positions': np.ndarray    # [T2, 22, 3]
        },
        'text': str                          # 文本描述
    },
    # ... 更多样本
}
```

**说明**：
- `T1`, `T2` 表示时间步（帧数），source和target可以有不同的长度
- `rots`: 22关节 × 3维axis-angle = 66维
- `trans`: 骨盆的(x, y, z)全局位置
- `joint_positions`: 所有关节的3D坐标

---

## 2. 转换后的数据 `data_dict`

经过 `cast_dict_to_tensors()` 后转为PyTorch tensors：

```python
data_dict = {
    'sample_id': {
        'motion_source': {
            'rots': torch.Tensor,              # [T1, 66]
            'trans': torch.Tensor,             # [T1, 3]
            'joint_positions': torch.Tensor    # [T1, 22, 3]
        },
        'motion_target': {
            'rots': torch.Tensor,              # [T2, 66]
            'trans': torch.Tensor,             # [T2, 3]
            'joint_positions': torch.Tensor    # [T2, 22, 3]
        },
        'text': str,
        'id': str,                             # 新增：样本ID
        'split': int                           # 新增：0=train, 1=val, 2=test
    },
    # ... 更多样本
}
```

---

## 3. 数据集初始化参数 `data`

传入 `MotionFixDataset.__init__()` 的列表格式：

```python
data = [
    {  # 第1个样本
        'motion_source': {
            'rots': torch.Tensor,              # [T1, 66]
            'trans': torch.Tensor,             # [T1, 3]
            'joint_positions': torch.Tensor    # [T1, 22, 3]
        },
        'motion_target': {
            'rots': torch.Tensor,              # [T2, 66]
            'trans': torch.Tensor,             # [T2, 3]
            'joint_positions': torch.Tensor    # [T2, 22, 3]
        },
        'text': str,
        'id': str,
        'split': int
    },
    # ... 更多样本
]
```

**说明**：
- 这是从 `data_dict` 的值（values）构成的列表
- 每个数据集（train/val/test）使用不同的样本子集

---

## 3+. SMPL 参数与特征关系

SMPL 参数和 MotionFix 里的“初始特征 / `__getitem__` 特征”其实都是对同一段人体动作的不同表述，只是信息粒度和处理阶段不同：

- **SMPL 参数 (`rots`, `trans`, `joint_positions`)**  
  - 这是标准 SMPL-H 模型输出的“骨架 + 网格”参数：`rots` 是 22 个关节的 axis-angle 旋转（骨盆 + 21 个身体关节），`trans` 是骨盆的全局平移，`joint_positions` 是把 `rots/trans` 喂给 SMPL 之后得到的 22×3 关节坐标。  
  - 它们是磁盘上 `motionfix.pth.tar` 里最原始的动作描述，能直接驱动 SMPL 模型重建 3D 人体。

- **初始 MotionFix 数据特征**  
  - 数据集加载时先把 `rots/trans/joints` 转成 tensor，然后存成 `data_dict` → `self.data`（参考 `data_view.md` 第 1~3 节）。这些还是“SMPL 语义”的参数，只是换成 PyTorch 格式、加了 `id/split` 等元数据。

- **`__getitem__` 之后的特征**  
  - 训练前还要把原始 SMPL 参数派生出更适合扩散模型的特征：如 `body_pose`（21 个关节的 6D 旋转）、`body_transl_delta_pelv`（骨盆坐标速度）、`body_orient_xy`、`body_joints_local_wo_z_rot` 等。  
  - 这些特征都是从 `rots`、`trans`、`joint_positions` 计算出来的派生量，便于归一化、拼接、对齐 source/target。  
  - `MotionFixDataset.__getitem__()` 会根据 `load_feats` 取出这些派生特征，附带文本、长度等，供 DataLoader → 模型使用。

**关联总结**  
1. **同一动作，不同表达**：SMPL 参数是最原始的动作表示；`__getitem__` 特征是对这些参数做了补充（旋转格式转换、速度/相对位姿、去 z 旋等），仍然描述同一动作。  
2. **SMPL 是“几何核心”**：无论是初始数据还是 `__getitem__`，最终都能通过 SMPL 前向得到真实 3D 骨架。训练阶段模型预测的也是这些派生特征，最后再通过 `diffout2motion`+SMPL 还原回可渲染的姿态。  
3. **数据流**：`SMPL 参数 → 特征派生 → 归一化拼接 → 模型 → 反归一化 → SMPL 重建`，`data_view.md` 从 raw 到模型输入/输出的每一步都基于这套 SMPL 语义。

因此三者本质上是一条链：SMPL 原始参数（磁盘）→ 转换成初始数据结构 → 在 `__getitem__` 里拆分/转换成更适合模型的特征。这些特征都仍然是“动作描述”，只是经过定制处理以满足模型训练和评估的需求。

---

## 4. 统计数据 `self.stats`

从 `.npy` 文件加载的归一化统计信息：

```python
self.stats = {
    'feature_name': {
        'mean': torch.Tensor,    # [D] - 每个特征维度的均值
        'std': torch.Tensor,     # [D] - 标准差
        'max': torch.Tensor,     # [D] - 最大值
        'min': torch.Tensor      # [D] - 最小值
    },
    # 所有特征的统计量
}
```

**常见特征**：
- `body_pose`: [126] - 21关节×6D旋转
- `body_transl_delta_pelv`: [3] - xyz速度
- `body_orient_xy`: [6] - 去z旋转的方向
- `z_orient_delta`: [6] - z轴旋转变化
- `body_joints_local_wo_z_rot`: [66] - 局部关节位置

---

## 5. 单个样本数据 (`__getitem__()` 返回)

从数据集获取单个样本后的完整格式：

```python
item = dataset[idx]  # 返回 DotDict

item = {
    # ========== Source 特征（输入运动）==========
    'body_pose_source': torch.Tensor,                      # [T1, 126]
    'body_transl_delta_pelv_source': torch.Tensor,         # [T1, 3]
    'body_orient_xy_source': torch.Tensor,                 # [T1, 6]
    'z_orient_delta_source': torch.Tensor,                 # [T1, 6]
    'body_joints_local_wo_z_rot_source': torch.Tensor,     # [T1, 66]
    
    # ========== Target 特征（目标运动）==========
    'body_pose_target': torch.Tensor,                      # [T2, 126]
    'body_transl_delta_pelv_target': torch.Tensor,         # [T2, 3]
    'body_orient_xy_target': torch.Tensor,                 # [T2, 6]
    'z_orient_delta_target': torch.Tensor,                 # [T2, 6]
    'body_joints_local_wo_z_rot_target': torch.Tensor,     # [T2, 66]
    
    # ========== 元数据 ==========
    'framerate': torch.Tensor,        # [1] - 帧率（通常30）
    'dataset_name': str,              # 数据集名称
    'length_source': int,             # 源序列长度
    'length_target': int,             # 目标序列长度
    'text': str,                      # 文本描述
    'split': int,                     # 数据划分
    'id': str                         # 样本ID
}
```

**特征维度说明**：
- `body_pose`: 21关节 × 6D旋转 = 126维
- `body_transl_delta_pelv`: xyz速度 = 3维
- `body_orient_xy`: 去z旋转的骨盆方向（6D表示）= 6维
- `z_orient_delta`: z轴旋转变化（6D表示）= 6维
- `body_joints_local_wo_z_rot`: 22关节 × 3D = 66维

---

## 6. 批次数据 (DataLoader 输出)

经过 `collate_fn` 处理后的批次数据（batch_size=B）：

```python
batch = {
    # ========== Source 特征（pad到最长序列）==========
    'body_pose_source': torch.Tensor,                      # [B, T_max_src, 126]
    'body_transl_delta_pelv_source': torch.Tensor,         # [B, T_max_src, 3]
    'body_orient_xy_source': torch.Tensor,                 # [B, T_max_src, 6]
    'z_orient_delta_source': torch.Tensor,                 # [B, T_max_src, 6]
    'body_joints_local_wo_z_rot_source': torch.Tensor,     # [B, T_max_src, 66]
    
    # ========== Target 特征（pad到最长序列）==========
    'body_pose_target': torch.Tensor,                      # [B, T_max_tgt, 126]
    'body_transl_delta_pelv_target': torch.Tensor,         # [B, T_max_tgt, 3]
    'body_orient_xy_target': torch.Tensor,                 # [B, T_max_tgt, 6]
    'z_orient_delta_target': torch.Tensor,                 # [B, T_max_tgt, 6]
    'body_joints_local_wo_z_rot_target': torch.Tensor,     # [B, T_max_tgt, 66]
    
    # ========== 长度与元数据 ==========
    'length_source': List[int],        # 批内每个样本的原始长度
    'length_target': List[int],
    'text': List[str],
    'framerate': List[torch.Tensor],   # 每项是 [1] tensor，例如 tensor([30])
    'split': List[int],
    'id': List[str],
    'dataset_name': List[str],         # 若混合多数据集时存在
    'task': List[str],                 # 每个样本的任务类型：'edit' 或 't2m'
    # 其他未出现在 feats 中的字段同样保持 List[...] 格式
}
```

**说明**：
- `T_max_src/tgt`: 批次中最长序列的长度
- `collate_batch_last_padding` 仅对 `load_feats` 中声明的特征执行“复制最后一帧”的 replicate padding，并将这些特征堆叠成 `[B, T_max, D]`
- `mask_source/mask_target` **不会**在 `collate_fn` 中生成；模型侧通过 `BaseModel.prepare_mot_masks()` 根据 `length_*` 列表构造 padding 掩码
- `task` 字段由 `collate_batch_last_padding()` 写入：含 source+target 的编辑样本标记为 `'edit'`，仅 target 的 text-to-motion 样本标记为 `'t2m'`
- 当 batch 仅包含 target 特征时（`task='t2m'`），collate 只会堆 target 张量；若混入如 HML3D 这类 target-only 数据集但模型仍期望 source，占位的 `source_*` 张量会在 collate 中被创建为零张量，以保证下游逻辑统一
- 元数据保持 Python list，Lightning 模块中按需转换
- 参考实现：`src/model/base.py::prepare_mot_masks`

---

## 7. 模型输入数据 (`train_diffusion_forward` 接收)

经过 `allsplit_step` → `norm_and_cat` 处理后的数据格式，传入 `train_diffusion_forward` 方法：

```python
batch = {
    # ========== 经过 norm_and_cat 处理后的统一特征（序列优先格式）==========
    # 注意：格式已从 [B, T_max, D] 转换为 [T_max, B, total_feat_dim]
    'source_motion': torch.Tensor,      # [T_max_src, B, total_feat_dim]
    'target_motion': torch.Tensor,      # [T_max_tgt, B, total_feat_dim]
    
    # ========== 长度信息 ==========
    'length_source': List[int],         # Lightning 模块中保持 list
    'length_target': List[int],
    
    # ========== 其他元数据（保持不变）==========
    'text': List[str],                  # 文本描述列表
    'id': List[str],                    # 样本ID列表
    'split': List[int],
    'framerate': List[torch.Tensor],    # 每个元素是 [1] tensor，例如 tensor([30])
    'dataset_name': List[str],         # 数据集名称列表
    # ... 其他元数据
}
```

**关键变化**：

1. **格式转换**：
   - DataLoader 输出：`[B, T_max, D]`（批次优先）
   - `norm_and_cat` 后：`[T_max, B, total_feat_dim]`（序列优先）
   - 转换方法：`rearrange(t, 'b s ... -> s b ...')`

2. **特征归一化**：
   - 所有特征都经过 `norm_inputs()` 归一化
   - 使用 `self.stats` 中的统计量（mean/std 或 min/max）

3. **特征拼接**：
   - 多个独立特征（如 `body_pose`, `body_transl_delta_pelv` 等）被拼接成一个统一特征向量
   - 总维度 = 所有 `input_feats` 维度之和（如 207 维）
4. **掩码构造在模型侧进行**：
   - `BaseModel.prepare_mot_masks(length_source, length_target)` 根据长度列表生成 `[B, T_max]` bool mask
   - 因此 DataLoader/batch 字典无需 `mask_source/mask_target` 字段

**示例**：

假设 `input_feats = ['body_pose', 'body_transl_delta_pelv', 'body_orient_xy', 'z_orient_delta', 'body_joints_local_wo_z_rot']`，总计 207 维：

```python
# DataLoader 输出（批次优先）
batch = {
    'body_pose_source': [32, 50, 126],           # 32个样本，50帧，126维
    'body_transl_delta_pelv_source': [32, 50, 3],
    'body_orient_xy_source': [32, 50, 6],
    'z_orient_delta_source': [32, 50, 6],
    'body_joints_local_wo_z_rot_source': [32, 50, 66],
    # ... target 特征类似
}

# norm_and_cat 处理后（序列优先，归一化，拼接）
batch = {
    'source_motion': [50, 32, 207],  # 50帧，32个样本，207维（拼接后的总特征）
    'target_motion': [60, 32, 207],  # 60帧，32个样本，207维
    'length_source': [50, 40, 45, ...],  # 32个样本的实际长度
    'length_target': [60, 55, 50, ...],
    'text': ['walk forward', 'sit down', ...],
    # ... 其他元数据
}
```

**数据流**：

```
DataLoader 输出（第6节格式）
    ↓
allsplit_step 接收
    ↓
norm_and_cat() 处理：
    - 格式转换：[B, T, D] → [T, B, D]
    - 归一化：使用 stats 归一化
    - 拼接：多个特征 → 统一特征向量
    ↓
batch['source_motion'] = [T_src, B, total_feat_dim]
batch['target_motion'] = [T_tgt, B, total_feat_dim]
    ↓
train_diffusion_forward 接收
```

**注意事项**：
- `source_motion` 和 `target_motion` 使用**序列优先格式** `[T, B, D]`
- 特征已**归一化**，可直接用于模型训练
- 多个特征已**拼接**成一个统一向量，不再有独立的特征键
- 元数据（text, length, id 等）保持原样

---

## 8. 特征维度列表 `self.nfeats`

记录每个加载特征的维度：

```python
self.nfeats = [D1, D2, D3, ...]  # List[int]
```

**示例配置**（对应 `load_feats`）：
```python
load_feats = [
    'body_pose',                    # → nfeats[0] = 126
    'body_transl_delta_pelv',       # → nfeats[1] = 3
    'body_orient_xy',               # → nfeats[2] = 6
    'z_orient_delta',               # → nfeats[3] = 6
    'body_joints_local_wo_z_rot'    # → nfeats[4] = 66
]

self.nfeats = [126, 3, 6, 6, 66]  # 总计207维
```

---

## 9. ROTS 数据分解

`rots` 的66维组成（axis-angle表示）：

```python
rots = [66]  # 单帧

# 维度分解：
rots[0:3]     # 骨盆方向 (pelvis orientation) - 3D axis-angle
rots[3:6]     # 左髋旋转 (left_hip) - 3D axis-angle
rots[6:9]     # 右髋旋转 (right_hip) - 3D axis-angle
rots[9:12]    # 脊柱1旋转 (spine1) - 3D axis-angle
# ... 共22个关节
rots[63:66]   # 第22个关节旋转 - 3D axis-angle
```

**关节总数**：22个
**每个关节**：3维 axis-angle 旋转
**总维度**：22 × 3 = 66

```
# ============ 从 rots [66] 提取的特征 ============

# 1. 骨盆旋转 (rots[0:3]) → 多个特征
rots[0:3]  # axis-angle, 3维
    ↓ transform_body_pose("aa->6d")
    ├─→ body_orient_xy [6]        # 骨盆全局旋转(6D表示)
    └─→ z_orient_delta [6]        # z轴旋转变化(6D表示)
                                   # 小计: 6 + 6 = 12维 (从3维扩展)

# 2. 身体姿态 (rots[3:66]) → body_pose
rots[3:66]  # 21个关节 × 3维axis-angle = 63维
    ↓ transform_body_pose("aa->6d")
    └─→ body_pose [126]           # 21个关节 × 6维6D = 126维
                                   # (从63维扩展到126维)

# rots衍生特征总计: 12 + 126 = 138维 (原始66维 → 138维)


# ============ 从 trans [3] 提取的特征 ============

trans [3]  # xyz位置
    ↓ 计算速度 (trans[i] - trans[i-1])
    ↓ 转换到骨盆坐标系
    └─→ body_transl_delta_pelv [3]  # xyz速度
                                     # 保持3维


# ============ 从 joint_positions [66] 提取的特征 ============

joint_positions [22, 3] = [66]  # 22个关节xyz坐标
    ↓ 转换到局部坐标系（去除z旋转）
    └─→ body_joints_local_wo_z_rot [66]  # 保持66维
```
---

## 10. JOINT_POSITIONS 数据分解

22个关节的3D位置：

```python
joint_positions = [22, 3]  # 单帧

# 关节索引和名称（按SMPL顺序）：
joint_positions[0]   # 骨盆 (pelvis)
joint_positions[1]   # 左髋 (left_hip)
joint_positions[2]   # 右髋 (right_hip)
joint_positions[3]   # 脊柱1 (spine1)
joint_positions[4]   # 左膝 (left_knee)
joint_positions[5]   # 右膝 (right_knee)
joint_positions[6]   # 脊柱2 (spine2)
joint_positions[7]   # 左踝 (left_ankle)
joint_positions[8]   # 右踝 (right_ankle)
joint_positions[9]   # 脊柱3 (spine3)
joint_positions[10]  # 左脚 (left_foot)
joint_positions[11]  # 右脚 (right_foot)
joint_positions[12]  # 颈部 (neck)
joint_positions[13]  # 左锁骨 (left_collar)
joint_positions[14]  # 右锁骨 (right_collar)
joint_positions[15]  # 头部 (head)
joint_positions[16]  # 左肩 (left_shoulder)
joint_positions[17]  # 右肩 (right_shoulder)
joint_positions[18]  # 左肘 (left_elbow)
joint_positions[19]  # 右肘 (right_elbow)
joint_positions[20]  # 左腕 (left_wrist)
joint_positions[21]  # 右腕 (right_wrist)
```

**每个关节**：[x, y, z] 3D坐标
**坐标系**：世界坐标系，单位为米

---

## 11. 数据流示意图

```
📁 磁盘文件 (motionfix.pth.tar)
    ↓
    joblib.load()
    ↓
📊 dataset_dict_raw (dict, numpy arrays)
    {sample_id: {motion_source, motion_target, text}}
    ↓
    cast_dict_to_tensors()
    ↓
🔢 data_dict (dict, torch tensors)
    添加 'id' 和 'split' 字段
    ↓
    按 train/val/test 划分
    ↓
📋 data (list of dicts)
    [sample1, sample2, ...]
    ↓
    MotionFixDataset.__init__()
    ↓
🗂️ self.data (dataset 内部存储)
    ↓
    __getitem__(idx)
    提取特征 + 旋转表示转换
    ↓
📦 单个样本 (DotDict)
    {body_pose_source, body_pose_target, ...}
    - source特征: [T1, D]
    - target特征: [T2, D]
    - 元数据: text, length, id
    ↓
    DataLoader + collate_fn
    padding + batching
    ↓
🎁 批次数据 (dict of tensors)
    {feature_name: [B, T_max, D], mask: [B, T_max], ...}
    ↓
    allsplit_step + norm_and_cat
    格式转换 + 归一化 + 特征拼接
    ↓
📥 train_diffusion_forward 输入
    {source_motion: [T_src, B, total_feat_dim],
     target_motion: [T_tgt, B, total_feat_dim], ...}
    ↓
    送入模型训练
    ↓
🧠 Model (MotionFix)
```

---

## 12. 数据格式总结表

### 原始数据格式

| 数据名称 | 类型 | Shape | 说明 |
|---------|------|-------|------|
| `rots` | Tensor | `[T, 66]` | 22关节×3D axis-angle旋转 |
| `trans` | Tensor | `[T, 3]` | 骨盆全局位置(x,y,z) |
| `joint_positions` | Tensor | `[T, 22, 3]` | 22关节3D位置 |

### 提取的特征格式

| 特征名称 | Shape | 说明 |
|---------|-------|------|
| `body_pose` | `[T, 126]` | 21关节×6D旋转（去除骨盆） |
| `body_transl` | `[T, 3]` | 骨盆全局位置 |
| `body_transl_z` | `[T, 1]` | 仅z轴高度 |
| `body_transl_delta` | `[T, 3]` | 全局速度 |
| `body_transl_delta_pelv` | `[T, 3]` | 骨盆坐标系下的速度 |
| `body_transl_delta_pelv_xy` | `[T, 3]` | 去除z旋转的xy平面速度 |
| `body_transl_delta_pelv_xy_wo_z` | `[T, 2]` | 仅xy速度 |
| `body_orient` | `[T, 6]` | 骨盆全局旋转（6D） |
| `body_orient_xy` | `[T, 6]` | 去除z轴旋转的方向 |
| `body_orient_delta` | `[T, 6]` | 方向变化率 |
| `z_orient_delta` | `[T, 6]` | 仅z轴旋转变化 |
| `body_pose_delta` | `[T, 126]` | 姿态变化率 |
| `body_joints` | `[T, 66]` | 关节全局位置（22×3） |
| `body_joints_rel` | `[T, 66]` | 相对骨盆的关节位置 |
| `body_joints_local_wo_z_rot` | `[T, 66]` | 去除z旋转的局部位置 |
| `body_joints_vel` | `[T, 66]` | 关节速度 |
| `joint_global_oris` | `[T, 189]` | 21关节全局方向（21×9） |
| `joint_ang_vel` | `[T, 126]` | 关节角速度（21×6） |
| `wrists_ang_vel` | `[T, 12]` | 手腕角速度（2×6） |
| `wrists_ang_vel_euler` | `[T, 6]` | 手腕角速度欧拉角（2×3） |

### 批次数据格式

| 数据名称 | Shape | 说明 |
|---------|-------|------|
| 特征张量 | `[B, T_max, D]` | padding后的特征（DataLoader输出） |
| `length_source/target` | `List[int]` | 每个样本的实际长度 |
| `text` | `List[str]` | 文本描述列表 |
| `framerate` | `List[torch.Tensor]` | 每项形如 `tensor([30])` |
| `split` | `List[int]` | 数据划分标记 |
| `id` | `List[str]` | 样本ID |
| （mask） | *模型端生成* | 由 `BaseModel.prepare_mot_masks()` 根据长度列表构造 |

### 模型输入数据格式（train_diffusion_forward）

| 数据名称 | Shape | 说明 |
|---------|-------|------|
| `source_motion` | `[T_max_src, B, total_feat_dim]` | 序列优先格式，归一化并拼接后的源动作 |
| `target_motion` | `[T_max_tgt, B, total_feat_dim]` | 序列优先格式，归一化并拼接后的目标动作 |
| `length_source` | `List[int]` | 源序列实际长度（稍后转换为 mask） |
| `length_target` | `List[int]` | 目标序列实际长度 |
| `text` | `List[str]` | 文本描述列表 |

### 其他数据结构

| 数据名称 | 类型 | 说明 |
|---------|------|------|
| `self.stats` | dict | 嵌套dict，包含mean/std/max/min |
| `self.nfeats` | List[int] | 特征维度列表 |
| `self.body_chain` | Tensor | `[52]` SMPL运动学链 |
| `self.joint_idx` | dict | 关节名称→索引映射 |

---

## 附录：常用配置

### 典型的 load_feats 配置

```python
# 配置1：基础特征（207维）
load_feats = [
    'body_pose',                      # 126维
    'body_transl_delta_pelv',         # 3维
    'body_orient_xy',                 # 6维
    'z_orient_delta',                 # 6维
    'body_joints_local_wo_z_rot'      # 66维
]

# 配置2：包含角速度
load_feats = [
    'body_pose',                      # 126维
    'body_transl_delta_pelv',         # 3维
    'joint_ang_vel',                  # 126维
    'body_joints_local_wo_z_rot'      # 66维
]
```

### 旋转表示转换

- **axis-angle (aa)**: 3维，编码旋转轴和角度
- **rotation matrix (rot)**: 3×3矩阵
- **6D rotation (6d)**: 6维，旋转矩阵的前两列
- **euler angles**: 3维，欧拉角表示

转换关系：`aa → rot → 6d`

---


# 1. 总览：项目数据流（全程大图）

```text
磁盘(.pth/.tar/.npz)
    ↓ joblib.load/load
原始数据(dataset_dict_raw [dict, numpy])
    ↓ cast_dict_to_tensors
PyTorch tensor数据(data_dict)
    ↓ 划分、标准化
数据集对象(MotionFixDataset)
    ↓ （可选）FrameSampler【当前代码未接入】
    ↓ __getitem__ 特征提取/转换
单条样本(DotDict: 含特征与元数据)
    ↓ collate_fn
batch数据(dict of [B,T,D] + 元数据列表)
    ↓ allsplit_step + norm_and_cat
拼接&归一化&格式转换(batch['source_motion'], 'target_motion' 等 [T,B,D])
    ↓ train_diffusion_forward
模型训练（扩散空间）
    ↓ 推理/采样
生成结果(diffout, [B,T,D])
    ↓ diffout2motion等（反归一化、积分）
还原真实动作特征 [B,T,D]（first_pose_feats等）
    ↓ 可视化/SMPL重建/评测
```

---

# 2. 详解每个阶段的数据**格式与变化**

## ★ ① 原始文件/磁盘

- **格式**：dict（嵌套numpy张量）
  - 详见 `data_view.md`“1. 原始加载数据”
- **特征**：每个样本有 source/target rots, trans, joint_positions, text

## ★ ② 转换为PyTorch张量

- **via** `cast_dict_to_tensors`
- **格式**：嵌套 torch.Tensor，字段同上，但全部转为 torch 格式

## ★ ③ 数据集划分（MotionFixDataset）

- 分成 train/val/test，多份 MotionFixDataset
- 在 `__getitem__` 时，自动做特征拆分与转换，并可选转多种角度

### 采样（FrameSampler，当前未启用）

- `configs/train.yaml` 中仍保留 `sampler: variable_conseq` 等占位配置，但 `MotionFixDataModule` 与 `MotionFixDataset` 目前不会实例化 `FrameSampler`
- 所有序列均以磁盘原长直接进入 `__getitem__`，长度控制完全依赖 `length_source/length_target` 与后续 mask
- 若需要重新启用裁剪，需要在数据模块中显式调用 `src/data/sampling/base.py` 的 `FrameSampler` 并更新本文档

### 单条样本输出
见 `data_view.md`“5. 单个样本数据”

- Example（target/source分开）：
  - body_pose_source       [T1,126]
  - body_transl_delta_pelv [T1,3]
  - ...（共多组 feature）

---

## ★ ④ 批次构建/归一化/合并

### collate_fn
- 合并多条样本成 batch（padding 补全），输出 `[B,T_max,D]` 格式
- 仅返回特征张量堆叠结果与元数据列表；mask 需在模型端利用长度信息生成

### allsplit_step + norm_and_cat (重点特征归一化+拼接)
- 拼接所有特征纬度，转为 `[T,B,total_feat_dim]` 格式  
- **归一化**：通过 self.stats，所有特征统计均值/方差带入归一化
- 只留下拼接后的 `'source_motion'`, `'target_motion'`, 长度列表等

**此时数据已为神经网络（扩散模型）可用的输入状态**

详见：
- `data_view.md`“7. 模型输入数据”
- `feature_view.md`“input_feats”定义（如 207 维五特征组合）

---

## ★ ⑤ 训练/推理阶段（扩散空间）

- 接收 [T, B, D] 格式，采样/预测时间步，加噪声，喂入网络
- **动作特征始终保持归一化、拼接格式，全部在 diffusion latent space**
- 网络预测的输出 diffout，shape 同输入

---

## ★ ⑥ 动作重建与特征还原（diffout2motion）

- 把扩散空间（归一化拼接特征）预测还原/解码为**原始物理空间**动作序列
  - 反归一化（unnormalize）
  - 特征拆分（uncat_inputs）、再重拼（cat_inputs）
  - “速度/增量/变化量”特征通过积累积分还原为绝对值
      - 例如从`body_transl_delta_pelv`积分得到全局位置

重建（`diffout2motion` → SMPL）只需要两类信息：

- **首帧基准** `first_pose_feats = ['body_transl', 'body_orient', 'body_pose']`
  - `body_transl`：骨盆绝对位置（3）
  - `body_orient`：骨盆全局旋转 6D（6）
  - `body_pose`：21 个身体关节 6D 旋转（126）
  - 这三项提供“绝对起点”，见 `data_view.md` 第 7 节的首帧说明。

- **序列特征**（模型输出后反归一化得到，默认为五项）
  - `body_pose`：全程 6D 关节姿态
  - `body_transl_delta_pelv`：骨盆坐标系速度，积分 + 首帧平移 → 绝对轨迹
  - `body_orient_xy`：去 z 旋的骨盆方向
  - `z_orient_delta`：z 轴旋转增量，叠加在首帧方向上
  - `body_joints_local_wo_z_rot`：去 z 旋的局部关节坐标，用于补全局部形变

`diffout2motion` 会把这五项拆回各自维度、做反归一化后，与首帧三项一起完成：
1. 积分 `body_transl_delta_pelv` → `body_transl`
2. 组合 `body_orient_xy` + `z_orient_delta` → `body_orient`
3. 直接复用/反归一化 `body_pose`、`body_joints_local_wo_z_rot`
最终得到 `full_motion_unnorm`（包含绝对平移、方向、姿态等全部重建特征），可直接喂进 SMPL。
见：

SMPL 本体确实只需要两块输入：  
- `rots`（22×3 axis-angle）→ 控制骨骼姿态  
- `trans`（3）→ 控制骨盆的全局平移  

MotionFix 在 “重建” 环节之所以看起来有更多特征，是因为我们必须把扩散空间的派生/归一化量还原回这两个 SMPL 输入：  
- `body_pose`、`body_orient`、`body_joints_local_wo_z_rot` 等都是对 `rots` 的拆分/转换（比如把 axis-angle 拆成骨盆 + 身体部分、转 6D 表示、去 z 旋等），便于训练和数值稳定。还原时要先把这些派生量重新组合成完整的 22×3 旋转，再喂给 SMPL。  
- `body_transl_delta_pelv` 是速度量，需要积分才能恢复绝对的 `trans`。  

所以流程是：“首帧基准 + 序列增量特征” → 重建 `body_orient/body_pose/body_transl` → 拼成标准的 `rots/trans` → 交给 SMPL。最终仍然只把这两种输入交给 SMPL，只是中间多了派生特征的还原步骤。
```python
# base_diffusion.py diffout2motion 
full_motion_unnorm = ...  # [B, T, D], 含绝对位置、旋转等
```

- 若用于 SMPL 重建，还会 reshape/pick 135维 `first_pose_feats`
- 见 `feature_view.md`“first_pose_feats（重建用）”

---

# 3. 各阶段对应数据结构对照表

| 阶段/函数           | 结构说明                         | 核心字段/shape               |
|---------------------|----------------------------------|------------------------------|
| 原始数据            | dataset_dict_raw                  | 'rots': [T,66] 'trans': [T,3]|
| cast_dict_to_tensors| data_dict                         | torch.Tensor，同上           |
| 帧采样（可选）       | FrameSampler                      | *当前未接入；保留配置占位*    |
| 数据集单样本        | __getitem__                       | body_pose_source:[T1,126]... |
| 批次数据            | collate_fn                        | [B,T,D] 特征 + List 元数据   |
| 统一输入            | norm_and_cat                      | source_motion: [T,B,207]     |
| 模型预测/采样       | train_diffusion_forward, denoiser | [T,B,207]                    |
| 输出/还原           | diffout2motion                    | full_motion_unnorm:[B,T,D]   |
| SMPL输入            | first_pose_feats                  | [B,T,135]                    |

---

# 4. 典型输入输出 **格式变化路径梳理**

1. **raw**: [T,66]/[T,3]  → (**可选** FrameSampler 裁剪；当前版本直接使用完整序列)  
2. **单条样本**: [T_s, D] / [T_t, D] → **collate_fn padding 成 [B,T_max,D]**  
3. **[B,T,D]**: batch生成 → **norm_and_cat 归一化 + 转为 [T,B,D]**  
4. **[T,B,D]**: 模型/扩散空间输入  → **模型内部运算**  
5. **[T,B,D]**: 模型输出  → **转回 [B,T,D]，拆分归一化 + 累加还原**  
6. **[B,T,135]**: (first_pose_feats) 重建绝对位置姿态  → SMPL驱动/渲染

---

# 5. 项目代码部分主要函数与流程配合解释

- **特征归一化与合并**：`norm_and_cat`/`cat_inputs`/`norm_inputs`
- **模型 forward**：通常接收 `[T,B,D]`, 网络内部可再处理/残差/投影
- **采样/生成**后：`diffout2motion` 做反向流程，解码动作
- **训练过程**主 loss 使用 diff 条件/掩码/特征重组算 loss
- **推理 eval/inference/render**：模型输出 motions 通过 `diffout2motion` 还原可视化

---

# 6. 相关文档链接组合

- **[数据加载与格式流转](motionfix/src/data/dataHelp/data_view.md)**
  - 1-6节数据结构定义
  - 7节及数据流图精确说明如何归一化、拼接、格式转换
  - 模型输入与输出格式
- **[数据模块与Loader分层](motionfix/src/data/dataHelp/DataModuleHelp.md)**
  - DataModule/Dataset/DataLoader的组织、责任区分和调用链
- **[特征视角说明](motionfix/src/data/dataHelp/feature_view.md)**
  - 各特征的来龙去脉，拼接方法
  - delta特征和first_pose_feats（重建特征）的互补与还原逻辑

---

# 7. 直观流程小结（精华版）

- **训练/推理正向**：  
  [Raw数据] → [特征归一化/拼接/标准输入] → [模型输入] → [模型输出 diffout（归一化拼接特征）]
- **动作还原逆向**：  
  [diffout] → [反归一化+特征拆分] → [“delta”特征积分还原绝对状态] → [first_pose_feats] → [下游可用/渲染]

---

如需可视化/代码级trace/具体某环节插入自定义处理的建议，请进一步告知！

**文档版本**: 1.0  
**最后更新**: 2025-11

