## 📊 MotionFix 数据集特征提取函数完整列表

### **一、平移特征（Translation Features）** - 6个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 1 | `body_transl` | `[T, 3]` | `data['trans']` | 骨盆全局位置(x,y,z) | 轨迹分析、位置跟踪 |
| 2 | `body_transl_z` | `[T, 1]` | `joint_positions[:, 0, 2:]` | 仅骨盆高度(z轴) | 高度检测、跳跃识别 |
| 3 | `body_transl_delta` | `[T, 3]` | `data['trans']` | 全局速度 (世界坐标系) | 整体运动速度分析 |
| 4 | `body_transl_delta_pelv` | `[T, 3]` | `data['trans']` | 骨盆坐标系下的速度<br>（包含完整旋转） | 精确运动分析 |
| 5 | `body_transl_delta_pelv_xy` | `[T, 3]` | `data['trans']` | 去除z轴旋转的速度<br>（保留高度变化） | 地面运动分析<br>跳跃/蹲下检测 |
| 6 | `body_transl_delta_pelv_xy_wo_z` | `[T, 2]` | `joint_positions[:, 0, :]` | 2D水平速度<br>（去除高度） | 路径规划<br>步态分析 |

**关键区别说明**：
- `delta_pelv`：完整骨盆坐标系（xyz所有旋转）
- `delta_pelv_xy`：只考虑z轴旋转 + 保留z维度
- `delta_pelv_xy_wo_z`：只考虑z轴旋转 + 去除z维度

---

### **二、方向特征（Orientation Features）** - 4个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 7 | `body_orient` | `[T, 6]` | `data['rots'][:3]` | 骨盆完整全局旋转<br>（6D表示） | 完整姿态重建 |
| 8 | `body_orient_xy` | `[T, 6]` | `data['rots'][:3]` | 去除z旋转的骨盆方向<br>（前倾/侧倾） | 姿态分析<br>方向不变特征 |
| 9 | `body_orient_delta` | `[T, 6]` | `data['rots'][:3]` | 骨盆方向变化率<br>（旋转速度） | 转身检测<br>动态分析 |
| 10 | `z_orient_delta` | `[T, 6]` | `data['rots'][:3]` | 仅z轴旋转变化率<br>（转身速度） | 转身动作识别<br>朝向变化 |

**关键点**：
- 所有方向特征使用**6D旋转表示**（更适合神经网络）
- `body_orient_xy` + `z_orient_delta` = 完整的方向信息分解

---

### **三、姿态特征（Pose Features）** - 2个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 11 | `body_pose` | `[T, 126]` | `data['rots'][3:66]` | 21个关节的局部旋转<br>（21×6D=126） | 姿态识别<br>动作分类 |
| 12 | `body_pose_delta` | `[T, 126]` | `data['rots'][3:66]` | 姿态变化率<br>（关节旋转速度） | 动作速度分析<br>动态特征 |

**说明**：
- 去除骨盆（关节0），只包含21个身体关节
- 从axis-angle（3D）转换为6D旋转表示

---

### **四、关节位置特征（Joint Position Features）** - 4个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 13 | `body_joints` | `[T, 66]` | `joint_positions` | 22个关节全局3D位置<br>（22×3=66） | 空间位置分析 |
| 14 | `body_joints_rel` | `[T, 66]` | `joint_positions` | 相对骨盆的关节位置<br>（完整旋转变换） | 局部姿态分析 |
| 15 | `body_joints_local_wo_z_rot` | `[T, 66]` | `joint_positions` | 去除z旋转的局部位置<br>（标准化姿态） | **常用**<br>姿态标准化<br>方向不变特征 |
| 16 | `body_joints_vel` | `[T, 66]` | `joint_positions` | 关节位置速度<br>（帧间差分） | 运动速度分析<br>动态检测 |

**关键区别**：
- `body_joints`：世界坐标系
- `body_joints_rel`：骨盆坐标系（完整旋转）
- `body_joints_local_wo_z_rot`：去z旋转的骨盆坐标系（最常用）

---

### **五、角速度特征（Angular Velocity Features）** - 4个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 17 | `joint_global_oris` | `[T, 189]` | `data['rots']` | 21个关节的全局旋转<br>（21×9=189，rotmat） | 全局方向分析<br>正向运动学 |
| 18 | `joint_ang_vel` | `[T, 126]` | `data['rots'][3:66]` | 21个关节的角速度<br>（21×6D=126） | 旋转速度分析<br>动态特征 |
| 19 | `wrists_ang_vel` | `[T, 12]` | `data['rots'][60:66]` | 双手腕角速度<br>（2×6D=12） | 手部动作识别<br>精细动作 |
| 20 | `wrists_ang_vel_euler` | `[T, 6]` | `data['rots'][60:66]` | 双手腕角速度<br>（欧拉角，2×3=6） | 可解释的手部速度 |

**计算方法**：
- 通过相邻帧旋转矩阵的相对旋转计算
- `ΔR = R_t @ R_{t-1}^T`

---

### **六、元数据（Meta Data）** - 2个

| # | 特征名称 | 维度 | 数据源 | 主要作用 | 应用场景 |
|---|---------|------|--------|---------|---------|
| 21 | `framerate` | `[1]` | 固定值 | 帧率（30fps） | 时序计算 |
| 22 | `dataset_name` | `str` | 固定值 | 数据集名称<br>("motionfix") | 数据管理 |

---

## 📈 特征分类总结

### **按功能分类**

```
空间特征（Where）：
├─ 平移：body_transl, body_transl_z
└─ 关节位置：body_joints, body_joints_rel, body_joints_local_wo_z_rot

姿态特征（What）：
├─ 方向：body_orient, body_orient_xy
└─ 姿态：body_pose

动态特征（How Fast）：
├─ 平移速度：body_transl_delta系列
├─ 旋转速度：body_orient_delta, z_orient_delta, body_pose_delta
├─ 关节速度：body_joints_vel
└─ 角速度：joint_ang_vel, wrists_ang_vel
```

### **按数据源分类**

```
data['rots'] [T, 66]:
├─ rots[0:3]    → 骨盆旋转特征（4个）
└─ rots[3:66]   → 身体姿态特征（6个）

data['trans'] [T, 3]:
└─ 平移特征（4个）

data['joint_positions'] [T, 22, 3]:
└─ 关节位置特征（6个）
```

### **常用特征组合**

```python
# 配置1：基础207维特征（最常用）
load_feats = [
    'body_pose',                      # 126维 - 姿态
    'body_transl_delta_pelv',         # 3维 - 速度
    'body_orient_xy',                 # 6维 - 方向（去z旋转）
    'z_orient_delta',                 # 6维 - 转身
    'body_joints_local_wo_z_rot'      # 66维 - 关节位置
]
# 总计：207维

# 配置2：完整特征（包含角速度）
load_feats = [
    'body_pose',                      # 126维
    'body_transl_delta_pelv',         # 3维
    'body_orient_xy',                 # 6维
    'z_orient_delta',                 # 6维
    'body_joints_local_wo_z_rot',     # 66维
    'joint_ang_vel'                   # 126维 - 角速度
]
# 总计：333维

# 配置3：2D运动（路径规划）
load_feats = [
    'body_pose',                      # 126维
    'body_transl_delta_pelv_xy_wo_z', # 2维 - 2D速度
    'body_orient_xy',                 # 6维
    'z_orient_delta',                 # 6维
    'body_joints_local_wo_z_rot'      # 66维
]
# 总计：206维
```

---

## 🎯 特征选择建议

| 任务类型 | 推荐特征 | 理由 |
|---------|---------|------|
| **动作识别** | body_pose + body_joints_local_wo_z_rot | 完整姿态信息 |
| **运动生成** | 全部空间+姿态特征 | 需要完整重建 |
| **路径规划** | body_transl_delta_pelv_xy_wo_z | 2D轨迹足够 |
| **跳跃检测** | body_transl_delta_pelv_xy (保留z) | 需要高度信息 |
| **转身识别** | z_orient_delta | 专注朝向变化 |
| **手部动作** | wrists_ang_vel | 专注手部 |
| **快速动作** | 所有delta/vel特征 | 强调动态 |

这个特征系统设计非常全面，提供了多个粒度和视角的运动表示！🎯✨


# first_pose_feats 与 input_feats 的关系：


## 两种特征的不同用途

### 1. `input_feats` (207维，5个特征) - 用于模型训练/推理

```python
# 实际使用的特征（训练时）
input_feats = [
    'body_pose',                      # 126维 - 姿态
    'body_transl_delta_pelv',         # 3维 - 速度（delta）
    'body_orient_xy',                 # 6维 - 方向（去z旋转）
    'z_orient_delta',                 # 6维 - 转身速度（delta）
    'body_joints_local_wo_z_rot'      # 66维 - 关节位置
]
# 总计：207维
```

特点：
- 包含 delta 特征（速度、变化量）
- 包含去 z 旋转的特征（更适合学习）
- 用于神经网络训练和推理

### 2. `first_pose_feats` (135维，3个特征) - 用于 SMPL 重建

```python
# SMPL重建所需的基础参数
first_pose_feats = [
    'body_transl',    # 3维 - 绝对位置
    'body_orient',    # 6维 - 完整旋转（包含z）
    'body_pose'       # 126维 - 姿态
]
# 总计：135维
```

特点：
- 都是绝对状态（不是 delta）
- 是 SMPL 模型的标准输入参数
- 用于重建人体网格

## 它们的关系

### 在 `diffout2motion` 中的转换

查看 `diffout2motion` 方法（第880-1003行），它负责从 `input_feats` 的 delta 特征重建 `first_pose_feats` 的绝对状态：

```python
# 1. 模型输出的是 input_feats (207维，包含delta)
feats_unnorm = ...  # 从 input_feats 反归一化

# 2. 从 delta 特征积分得到绝对状态
# 例如：从 body_transl_delta_pelv 积分得到 body_transl
pelvis_delta = feats_unnorm[..., :3]  # body_transl_delta_pelv
full_trans = torch.cumsum(trans_vel_pelv, dim=1) + first_trans  # 积分

# 3. 从 body_orient_xy + z_orient_delta 重建 body_orient
xy_orient = feats_unnorm[..., 3:9]  # body_orient_xy
z_orient_delta = feats_unnorm[..., 9:15]  # z_orient_delta
# 积分得到完整的 body_orient

# 4. 最终拼接成 first_pose_feats 格式用于SMPL重建
full_motion_unnorm = torch.cat([full_trans, full_rots], dim=-1)
# [body_transl, body_orient, body_pose] = first_pose_feats
```

### 在 `unnorm_state` 和 `norm_state` 中的使用

```305:324:src/model/base.py
    def unnorm_state(self, state_norm: Tensor) -> Tensor:
        # unnorm state
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(state_norm,
                                                 self.first_pose_feats_dims),
                               self.first_pose_feats))[0]
        
    def unnorm_delta(self, delta_norm: Tensor) -> Tensor:
        # unnorm delta
        return self.cat_inputs(
            self.unnorm_inputs(self.uncat_inputs(delta_norm,
                                                 self.input_feats_dims),
                               self.input_feats))[0]

    def norm_state(self, state:Tensor) -> Tensor:
        # normalise state
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(state, 
                                               self.first_pose_feats_dims),
                             self.first_pose_feats))[0]

    def norm_delta(self, delta:Tensor) -> Tensor:
        # normalise delta
        return self.cat_inputs(
            self.norm_inputs(self.uncat_inputs(delta, self.input_feats_dims),
                             self.input_feats))[0]
```

可以看到：
- `norm_state` / `unnorm_state` 使用 `first_pose_feats`（状态）
- `norm_delta` / `unnorm_delta` 使用 `input_feats`（delta）

## 完整流程

```
训练阶段：
┌─────────────────────────────────────┐
│ input_feats (207维)                 │
│ - body_pose                         │
│ - body_transl_delta_pelv (delta)    │
│ - body_orient_xy (去z旋转)          │
│ - z_orient_delta (delta)            │
│ - body_joints_local_wo_z_rot        │
└─────────────────────────────────────┘
           ↓ 模型学习
           ↓
┌─────────────────────────────────────┐
│ 模型输出: input_feats (207维)       │
└─────────────────────────────────────┘
           ↓ diffout2motion()
           ↓ 从delta积分得到绝对状态
┌─────────────────────────────────────┐
│ first_pose_feats (135维)            │
│ - body_transl (从delta积分得到)      │
│ - body_orient (从xy+z_delta重建)     │
│ - body_pose                         │
└─────────────────────────────────────┘
           ↓ SMPL前向传播
           ↓
┌─────────────────────────────────────┐
│ 人体网格顶点 [B, 6890, 3]           │
└─────────────────────────────────────┘
```

## 总结

- 模型训练/推理使用 `input_feats` (207维，5个特征)，包含 delta 和去 z 旋转的特征。
- `first_pose_feats` (135维，3个特征) 用于 SMPL 重建，是绝对状态。
- 转换：`diffout2motion` 从 `input_feats` 的 delta 积分/重建得到 `first_pose_feats` 的绝对状态，再用于 SMPL 重建。

因此，`first_pose_feats` 不是用于训练，而是用于从模型输出重建 SMPL 姿态的中间表示。