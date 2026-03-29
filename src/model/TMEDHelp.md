# TMED的完整流程
## 场景设定

假设我们要编辑一个动作：
- 源动作 (M_S)：一个人慢慢走路，20 帧，每帧 263 维特征
- 编辑文本 (L)："walk faster"
- 目标动作 (M_T)：生成一个更快的走路动作，25 帧

---

## 第一阶段：模型初始化 (`__init__`)

```python
# 创建模型实例
model = TMED_denoiser(
    nfeats=263,              # 每帧动作特征维度 (SMPL+H)
    latent_dim=256,         # 潜在空间维度
    text_encoded_dim=768,    # CLIP 文本编码维度
    motion_condition="source",  # 使用源动作条件
    use_sep=True            # 使用 [SEP] 分隔符
)
```

### 初始化了什么？

```python
# 1. 动作编码器（两个独立的线性层）
pose_proj_in_source = Linear(263 → 256)  # 编码源动作
pose_proj_in_target = Linear(263 → 256)  # 编码目标动作
pose_proj_out = Linear(256 → 263)         # 解码回动作空间

# 2. 文本投影层（CLIP 768维 → 256维）
emb_proj = Linear(768 → 256)

# 3. 时间步嵌入器
embed_timestep = TimestepEmbedderMDM(256)

# 4. 位置编码器
query_pos = PositionalEncoding(256)

# 5. [SEP] 分隔符（可学习参数）
sep_token = Parameter([1, 256])  # 随机初始化

# 6. Transformer 编码器（9层）
encoder = TransformerEncoder(9 layers, 4 heads, 1024 hidden)
```

---

## 第二阶段：前向传播 (`forward`)

### 输入准备

```python
# 假设我们在扩散过程的第 t=150 步（共 300 步）
noised_motion = [2, 25, 263]      # 批次=2, 25帧, 263维（带噪声的目标动作）
timestep = [150, 150]            # 两个样本都在第150步
text_embeds = [2, 77, 768]       # CLIP 编码的文本（77个token）
motion_embeds = [2, 20, 263]     # 源动作（20帧）
in_motion_mask = [2, 25]         # 目标动作掩码（25帧都有效）
condition_mask = [2, 97]         # 条件掩码（77文本 + 20源动作）
```

### 步骤 0：维度转换

```python
# 第 157-158 行
bs = 2
noised_motion = noised_motion.permute(1, 0, 2)
# [2, 25, 263] → [25, 2, 263]  (序列优先格式)
```

### 步骤 1：时间步嵌入

```python
# 第 170-171 行
timesteps = [150, 150]  # 广播到序列长度
time_emb = embed_timestep(timesteps)
# 输出: [1, 2, 256]  (1个时间token, 2个批次, 256维)
```

时间步嵌入告诉模型当前处于扩散过程的哪个阶段（150/300 = 50%）。

### 步骤 2：文本条件嵌入

```python
# 第 176-194 行
text_embeds = text_embeds.permute(1, 0, 2)  # [77, 2, 768]
text_emb_latent = emb_proj(text_embeds)     # [77, 2, 256]
emb_latent = torch.cat((time_emb, text_emb_latent), 0)
# 拼接后: [78, 2, 256]  (1个时间 + 77个文本token)
```

### 步骤 3：源动作条件嵌入

```python
# 第 198-207 行
motion_embeds_proj = pose_proj_in_source(motion_embeds)
# [20, 2, 263] → [20, 2, 256]
```

### 步骤 4：目标动作编码

```python
# 第 215 行
proj_noised_motion = pose_proj_in_target(noised_motion)
# [25, 2, 263] → [25, 2, 256]
```

### 步骤 5：构建完整输入序列

```python
# 第 224-229 行
sep_token_batch = sep_token.repeat(2, 1)  # [2, 256]
xseq = torch.cat((
    emb_latent,              # [78, 2, 256]  时间+文本
    motion_embeds_proj,      # [20, 2, 256]  源动作
    sep_token_batch[None],   # [1, 2, 256]   [SEP]
    proj_noised_motion       # [25, 2, 256]  目标动作
), axis=0)
# 最终序列: [124, 2, 256]  (78+20+1+25)
```

序列结构：
```
位置:  0     1-77   78-97   98     99-123
内容: [时间] [文本] [源动作] [SEP] [目标动作]
维度:  1     77     20      1      25
```

### 步骤 6：添加位置编码

```python
# 第 237 行
xseq = query_pos(xseq)  # 为每个位置添加位置信息
```

### 步骤 7：构建注意力掩码

```python
# 第 259-265 行
time_token_mask = [True, True]           # 时间token有效
text_mask = condition_mask[:, :77]        # 文本掩码 [2, 77]
motion_mask = condition_mask[:, 77:]     # 源动作掩码 [2, 20]
sep_token_mask = [True, True]             # SEP有效
target_mask = in_motion_mask              # 目标动作掩码 [2, 25]

aug_mask = torch.cat((
    time_token_mask,    # [2, 1]
    text_mask,          # [2, 77]
    motion_mask,        # [2, 20]
    sep_token_mask,     # [2, 1]
    target_mask         # [2, 25]
), 1)
# 最终: [2, 124]
```

### 步骤 8：Transformer 编码

```python
# 第 276 行
tokens = encoder(xseq, src_key_padding_mask=~aug_mask)
# 输入: [124, 2, 256]
# 输出: [124, 2, 256]
```

Transformer 通过自注意力处理整个序列：
- 目标动作的每个帧可以关注文本、源动作和时间信息
- 学习如何根据文本和源动作去噪

### 步骤 9：提取去噪后的动作

```python
# 第 280-289 行
denoised_motion_proj = tokens[78:]        # 跳过时间+文本 [46, 2, 256]
denoised_motion_proj = denoised_motion_proj[21:]  # 跳过源动作+SEP [25, 2, 256]
```

### 步骤 10：投影回动作空间

```python
# 第 296 行
denoised_motion = pose_proj_out(denoised_motion_proj)
# [25, 2, 256] → [25, 2, 263]
```

### 步骤 11：填充区域置零

```python
# 第 314 行
denoised_motion[~motion_in_mask.T] = 0  # 确保填充区域为0
```

### 步骤 12：转回批次优先格式

```python
# 第 318 行
denoised_motion = denoised_motion.permute(1, 0, 2)
# [25, 2, 263] → [2, 25, 263]
return denoised_motion
```

---

## 第三阶段：带引导的前向传播 (`forward_with_guidance`)

推理时使用无分类器引导（CFG）增强条件控制。

### 3-way 引导（有源动作）

```python
# 第 432-446 行
# 1. 复制输入为3份
third = noised_motion[:len(noised_motion)//3]  # [1, 25, 263]
combined = torch.cat([third, third, third], dim=0)  # [3, 25, 263]

# 2. 同时计算三种预测
model_out = forward(combined, ...)
# 得到: [3, 25, 263]

# 3. 分离三种预测
uncond_eps = model_out[0]        # 无条件预测（无文本无源动作）
cond_eps_motion = model_out[1]    # 仅源动作条件
cond_eps_text_n_motion = model_out[2]  # 文本+源动作条件
```

### 应用引导公式

```python
# 第 475-476 行（论文公式 5）
guidance_motion = 2.0          # 源动作引导强度
guidance_text_n_motion = 2.0   # 文本引导强度

third_eps = uncond_eps + \
            guidance_motion * (cond_eps_motion - uncond_eps) + \
            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
```

公式含义：
- `cond_eps_motion - uncond_eps`：源动作带来的变化
- `cond_eps_text_n_motion - cond_eps_motion`：文本带来的额外变化
- 引导尺度放大这些差异，使生成更符合条件

---

## 完整流程图

```
输入:
├─ noised_motion: [2, 25, 263]  (带噪声的目标动作)
├─ text_embeds: [2, 77, 768]    (CLIP编码的"walk faster")
├─ motion_embeds: [2, 20, 263]  (源动作: 慢走)
└─ timestep: 150

↓ 编码阶段
├─ 时间嵌入: [1, 2, 256]
├─ 文本投影: [77, 2, 256]
├─ 源动作投影: [20, 2, 256]
└─ 目标动作投影: [25, 2, 256]

↓ 序列构建
[时间|文本|源动作|SEP|目标动作]
[1|77|20|1|25] = 124 tokens

↓ Transformer 编码
[124, 2, 256] → [124, 2, 256]

↓ 提取目标动作部分
[25, 2, 256]

↓ 投影回动作空间
[25, 2, 263]

↓ 输出
denoised_motion: [2, 25, 263]  (去噪后的动作预测)
```

---

## 关键设计点

1. 序列拼接：将时间、文本、源动作、目标动作拼接成一个序列，让 Transformer 同时处理所有信息
2. 掩码机制：区分有效数据和填充，避免关注无效位置
3. 无分类器引导：通过对比条件和无条件预测，增强条件控制
4. 位置编码：帮助模型理解序列的时间顺序

这就是 `TMED_denoiser` 的完整工作流程。每一步都围绕“如何根据文本和源动作，对带噪声的目标动作进行去噪”这个核心任务。

# Clssifier free guide
## 关键点：同一个 `forward`，三种条件

在 `generate_motion()` → `_diffusion_reverse()` → `forward_with_guidance()` 的调用链里，模型会一次性喂入一个**三倍长的批次**，而这个批次里的三个“段”分别对应：

| 段编号 | 文本条件                 | 源动作条件             | 目标 | 用于计算 |
|--------|--------------------------|------------------------|------|----------|
| 0      | 文本被置空/掩蔽          | 源动作被清零           | `ε_uncond` | 无条件预测 |
| 1      | 文本被置空/掩蔽          | 源动作是真实的         | `ε_motion` | 仅源动作条件 |
| 2      | 文本是真实的             | 源动作是真实的         | `ε_text+motion` | 文本+源动作条件 |

### 如何实现这三种“不同”？

1. **`text_embeds`** 在进入 `forward_with_guidance` 前已经扩展成三段：
   - 在 `generate_motion()` 里：
     ```python
     texts_cond = ['']*bsz + texts_cond          # 先加一段空文本
     texts_cond = ['']*bsz + texts_cond          # 如果有 motion 条件再加一段空文本
     text_emb, text_mask = self.text_encoder(texts_cond)
     ```
     结果：第 0 段、1 段的文本全是 `''`（CLIP 输出全零并配合 mask），第 2 段是原始文本。

2. **`motion_embeds`** 在 `_diffusion_reverse()` 中被拼成：
   ```python
   torch.cat([zeros_like(motion_embeds), motion_embeds, motion_embeds], dim=1)
   ```
   - 第 0 段：全零 → 不提供源动作
   - 第 1 段、2 段：真实源动作

3. **`condition_mask`** 也被拼接成三段，确保：
   - 第 0 段：文本和动作都标记为“无效”
   - 第 1 段：文本无效，动作有效
   - 第 2 段：文本有效，动作有效

   这一步是在 `_diffusion_reverse()` 里完成的：
   ```python
   motion_masks = torch.cat([nomotion_mask, cond_motion_masks, cond_motion_masks], dim=0)
   aug_mask = torch.cat([text_masks, motion_masks], dim=1)
   ```
   对应关系和上面三段一致。

当这些重新排列好的张量传到 `forward_with_guidance` 里：

```python
third = noised_motion[: len(noised_motion) // 3]
combined = torch.cat([third, third, third], dim=0)
model_out = self.forward(combined, timestep,
                         in_motion_mask=...,
                         text_embeds=text_embeds,        # 已经 3*bsz
                         condition_mask=aug_mask,        # 已经 3*bsz
                         motion_embeds=motion_embeds,    # 已经 3*bsz
                         ...)
uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out, len(model_out) // 3, dim=0)
```

虽然 `noised_motion` 三段内容一样，但**不同段**搭配了不同的 `text_embeds` / `motion_embeds` / `condition_mask`。Transformer 在前向时会看到“同一帧噪声”在不同条件下的三种可能，于是就得到三种预测。

最后再用 CFG 公式组合：

```python
third_eps = uncond_eps \
            + guidance_motion       * (cond_eps_motion - uncond_eps) \
            + guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
```

这一步就是论文里公式 (5) 的实现：无条件、只有源动作、文本+源动作的预测一次性算出来，再按引导系数组合。

---

## 小结

- **复制三份 `noised_motion`** 是为了在同一批次内对同一个噪声状态分别跑“无条件 / 单条件 / 双条件”。
- 使它们真正“不同”的，是 `text_embeds` / `motion_embeds` / `condition_mask` 被预先拼成了三段。
- 这种“拼批”方式可以只调用一次 `forward`，就得到 CFG 需要的三套输出，既省时间又方便做并行。

因此，“输入复制三份”只是为了对齐批次，真正控制条件的是额外的嵌入和掩码。

## 实际流程分析

### 1. 输入准备阶段（第372-374行）

```372:374:/media/data1/wurenhao/projects/motionfix/src/model/base_diffusion.py
                                motion_embeds=torch.cat([torch.zeros_like(motion_embeds),
                                                        motion_embeds,
                                                        motion_embeds], 1),
```

`motion_embeds` 被拼接成3份：
- 第1份：`torch.zeros_like(motion_embeds)` - 全零，用于无条件预测
- 第2份：`motion_embeds` - 真实动作，用于仅动作条件预测
- 第3份：`motion_embeds` - 真实动作，用于文本+动作条件预测

### 2. 在 `forward_with_guidance` 中（第432-446行）

```432:446:/media/data1/wurenhao/projects/motionfix/src/model/tmed_denoiser.py
            # 复制输入以同时计算三种预测: 无条件、仅动作条件、文本+动作条件
            third = noised_motion[: len(noised_motion) // 3]
            combined = torch.cat([third, third, third], dim=0)
            
            # 前向传播: 同时得到三种条件的预测
            model_out = self.forward(combined, timestep,
                                     in_motion_mask=in_motion_mask,
                                     text_embeds=text_embeds,
                                     condition_mask=condition_mask, 
                                     motion_embeds=motion_embeds,
                                     lengths=lengths)
            
            # 分离三种预测
            uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out,
                                                                            len(model_out) // 3,
                                                                            dim=0)
```

这里计算了3种不同条件的预测：
- `uncond_eps`：无条件预测（使用第1份全零的 motion_embeds）
- `cond_eps_motion`：仅动作条件预测（使用第2份真实 motion_embeds）
- `cond_eps_text_n_motion`：文本+动作条件预测（使用第3份真实 motion_embeds）

### 3. 融合阶段（第475-476行）

```475:476:/media/data1/wurenhao/projects/motionfix/src/model/tmed_denoiser.py
                third_eps = uncond_eps + guidance_motion * (cond_eps_motion - uncond_eps) + \
                            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
```

将3种预测融合成一个结果 `third_eps`。

### 4. 输出复制（第483行）

```483:483:/media/data1/wurenhao/projects/motionfix/src/model/tmed_denoiser.py
            eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
```

融合结果被复制成3份（相同的）。

### 5. 最终处理（第412行）

```412:412:/media/data1/wurenhao/projects/motionfix/src/model/base_diffusion.py
            _, _, samples = samples.chunk(3, dim=0)  # 移除无条件（null class）采样，仅保留最后一组真实条件结果
```

## 注释为什么不准确？

注释说“移除无条件采样，保留真实条件”，但实际是：
- 采样循环中每一步都计算了3种不同条件的预测（包括无条件）
- 这些预测被融合成一个结果
- 融合后的结果被复制成3份（相同的）
- 最终只保留一份，因为其他份都是相同的


