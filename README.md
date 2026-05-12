# L1-Robust KV Cache

> 用 ℓ₁ 子空间嵌入理论为 LLM 推理选择最优 KV 缓存保留策略。
> —— 基于 David P. Woodruff (2014) *"Sketching as a Tool for Numerical Linear Algebra"* §3.5。

---

## 目录

1. [背景：LLM 推理为什么需要 KV 缓存？](#1-背景llm-推理为什么需要-kv-缓存)
2. [问题：KV 缓存满了怎么办？](#2-问题kv-缓存满了怎么办)
3. [现有方案的局限](#3-现有方案的局限)
4. [我们的方法：ℓ₁ 杠杆分数](#4-我们的方法ℓ₁-杠杆分数)
5. [数学原理](#5-数学原理)
6. [算法流程](#6-算法流程)
7. [实验结果摘要](#7-实验结果摘要)
8. [项目结构](#8-项目结构)
9. [快速开始](#9-快速开始)
10. [引用](#10-引用)

---

## 1. 背景：LLM 推理为什么需要 KV 缓存？

大语言模型（GPT、Llama、Qwen）是 **decoder-only Transformer**。它们生成文本是逐 token 的——每次产出一个新 token，然后把它拼到输入里，再产下一个。

在 **Self-Attention** 层中：

```
Q（新 token 的 query）：  "我想找什么信息"
K（所有旧 token 的 key）：  "我身上有什么标签"
V（所有旧 token 的 value）："我有什么内容可以贡献"

输出 = softmax(Q·K^T / √d) · V
```

每生成一个新 token，它需要**对之前所有 token 的 K 和 V 做 attention**。如果不缓存，每一步都要把前 1000 个 token 全部重算一遍 K 和 V——计算量 O(N²)，一秒只能产一个字。

**KV 缓存**就是把已经算过的 K 和 V 存起来。新 token 只算自己的 Q、K、V，然后从缓存里读旧 token 的 K 和 V——计算量降到 O(N)，速度快了上千倍。

```
past_key_values = (
    (K_layer0, V_layer0),    # 第 0 层的缓存
    (K_layer1, V_layer1),    # 第 1 层的缓存
    ...                        # 总共 L 层
)
```

---

## 2. 问题：KV 缓存满了怎么办？

KV 缓存占的显存随序列长度**线性增长**。对于 Llama-2-7B，**每 1000 个 token 占约 0.5 GB 显存**。上下文窗口越长（32K、128K），缓存越大，最终超出 GPU 显存上限。

**必须丢掉一些旧的 token。** 但丢谁？

这就是 KV 缓存淘汰问题。在固定预算（比如只保留 128 个 token）下，**选哪些 token 保留，能让模型在后续生成中依然保持高质量？**

---

## 3. 现有方案的局限

| 方法 | 选 token 标准 | 局限 |
|------|------|------|
| **StreamingLLM** (2023) | 前 4 个 + 最近 124 个（位置启发式） | 不知道中间哪些 token 重要 |
| **H2O** (NeurIPS 2023) | 累积 attention score（过去被关注越多的留） | token 刚出现时累积分数为 0，可能来不及累积就被淘汰 |
| **SnapKV** (2024) | 注意力模式匹配 | 对检索型任务帮助有限 |

**所有现有方法都在问同一个问题："哪些 token 对下一个预测最重要？"** 但它们都依赖 attention 信号或位置启发式——没有一个用到了 V 矩阵本身的**数学结构**。

---

## 4. 我们的方法：ℓ₁ 杠杆分数

我们换个问题问：

> "哪些 token 的 V 向量，在 ℓ₁ 范数下，**不能被其他 token 的 V 向量线性近似**？"

如果能被其他 token 拼出来，那它就不重要——丢了不心疼。如果不能被近似，那它在 V 空间结构中**不可替代**——必须保留。

这个"不可被近似程度"在数学上称为 **ℓ₁ 杠杆分数（leverage score）**：

$$\ell_i = \|(V \cdot R^{-1})_{i,*}\|_1$$

- V：缓存中所有 token 的 value 向量组成的矩阵 [n, d]
- R：来自对 V 的 ℓ₁ 子空间嵌入做 QR 分解得到的良条件基
- ℓ_i 越高 → 第 i 个 token 在 V 列空间中的贡献越不可替代

**我们的 ℓ₁ 杠杆分数不依赖 attention 信号——只看 V 矩阵本身的几何结构。**

---

## 5. 数学原理

### 5.1 为什么不用 ℓ₂？

ℓ₂ 杠杆分数（CurDKV 使用）对异常值**平方放大**——一个 attention sink token 的 ℓ₂ 分数会大到挤占所有其他 token 的预算。ℓ₁ 线性看待每个值——异常值和普通值被平等对待，预算分配更均衡。

### 5.2 ℓ₁ 子空间嵌入（Woodruff 2014 §3.5, 构造 B）

关键数学工具是 Woodruff 的 **构造 B**：

$$S = \Phi \cdot D$$

- $\Phi$：CountSketch 矩阵——稀疏随机投影，每列一个 ±1
- $D$：对角矩阵，$D_{i,i} = 1/E_i$，其中 $E_i \sim \text{Exp}(1)$

**核心直觉**：CountSketch 本身是 ℓ₂ 子空间嵌入。乘以指数分布权重 $1/E_i$ 后，它变成了 ℓ₁ 子空间嵌入。这构造有理论保证——$r = O(d^2/\varepsilon^2)$ 行，与序列长度 n 无关。

### 5.3 管线

```
V [n, d]  →  除以 Exp(1)  →  QR 分解  →  R⁻¹ [d, d]  →  V·R⁻¹  →  ℓ₁ 范数  →  杠杆分数 [n]
```

---

## 6. 算法流程

我们的 **L1RobustKVCache** 淘汰策略：

```
当缓存溢出时：
  1. 提取 V 矩阵（跨 head 平均）
  2. 计算 ℓ₁ 杠杆分数（每 recompute_interval=32 步重算一次）
  3. 用 Q·K^T 注意力权重加权 ℓ₁ 分数（query-aware scoring）
  4. 按混合预算选 token 保留：
     ├── sink tokens（前 4 个，注意力汇聚点）
     ├── recency block（最近 N 个）
     ├── ℓ₁ top-k（从远历史中挑结构独特的 token）
     └── 最后一个 token（始终保留）
```

sink + recency + ℓ₁ + last = budget。recency 兜底局部上下文，ℓ₁ 从远处淘结构上不可替代的关键 token。

### 位置编码修正（pos_shift）

淘汰中间的 token 后，缓存的物理位置和逻辑位置脱节了——RoPE 需要修正。我们在 `pos_shift/` 中替换了 HuggingFace 原生的 `LlamaAttention.forward`，令 Q 用 shifted 位置编码、K 用物理缓存位置编码，保证淘汰后模型仍然正确。

---

## 7. 实验结果摘要

### 7.1 PPL（语言建模保真度）——持平

在 wikitext-103 长文本上，ℓ₁ 混合策略（recent_keep=80）PPL 为 8.98，追平 StreamingLLM（8.37，差距 7%）。

### 7.2 HotpotQA（多跳推理）——碾压

在需要跨文档检索的 HopotQA 困难样本上：

```
recency_only:  PPL=1002  （完全找不到答案）
StreamingLLM:  PPL=847   （有 sink 但不够）
ℓ₁(RK=17):     PPL=20    （42× 提升）
```

### 7.3 隐形针测试（长程检索）——中等距离优势显著

在 200 token 距离上，ℓ₁ 混合策略 PPL 比 StreamingLLM 低 21%。

### 7.4 跨架构验证

在 GPT-NeoX (pythia-410M)、Llama (TinyLlama-1.1B)、Llama-1 (huggyllama-7B) 三个架构上方向一致。

---

## 8. 项目结构

```
l1-robust-kv-cache/
├── benchmark.py              # 主测试脚本（7 种数据源，4 种对比模式）
├── data_sources.py           # 测试文本构造器（HotpotQA, 针测试, PPL 等）
├── cache_baselines.py        # 滑动窗口基线 + 缓存格式转换工具
├── l1_llm/                   # ★ 核心算法
│   ├── l1_sketch.py          #   CountSketch → Exp(1) → QR → ℓ₁ 杠杆分数
│   ├── kv_cache.py           #   L1RobustKVCache 淘汰策略
│   └── pos_shift/            #   位置编码修正（适配 4 种架构）
│       ├── modify_llama.py   #     Llama 架构
│       ├── modify_qwen2.py   #     Qwen2 架构
│       ├── modify_gpt_neox.py#     Pythia/GPT-NeoX 架构
│       └── modify_falcon.py  #     Falcon 架构
├── plain_llm/                # PlainKVCache（不淘汰，基线）
│   └── kv_cache.py
└── streaming_llm/            # StartRecentKVCache（StreamingLLM 基线）
    └── kv_cache.py
```

---

## 9. 快速开始

```bash
# 安装依赖
pip install torch transformers datasets

# 跑 HotpotQA 多跳推理测试（网格搜索最优 recency_keep）
python benchmark.py \
    --model PY007/TinyLlama-1.1B-step-50K-105b \
    --device cuda \
    --text_source hotpotqa \
    --split validation --qa_sample_idx 0 \
    --comparison_mode grid \
    --mixed_recent_keeps "16,17,18,19,20" \
    --max_steps 4096 --cache_size 128 --start_size 4

# 跑隐形针测试（三组对比）
python benchmark.py \
    --model EleutherAI/pythia-410m-deduped \
    --device cpu \
    --text_source needle \
    --comparison_mode needle \
    --needle_pos 400 --max_steps 1400 \
    --cache_size 128 --start_size 4 --mixed_recent_keep 32

# 跑标准 PPL 长文测试
python benchmark.py \
    --model PY007/TinyLlama-1.1B-step-50K-105b \
    --device cuda \
    --text_source long \
    --split test --long_target_words 5000 \
    --comparison_mode three \
    --max_steps 4096 --cache_size 128 --start_size 4 --mixed_recent_keep 80
```

---

## 10. 引用

本项目的理论基础来自：

> David P. Woodruff. *"Sketching as a Tool for Numerical Linear Algebra"*. Foundations and Trends in Theoretical Computer Science, 2014.

KV 缓存淘汰基线：

> Guangxuan Xiao et al. *"Efficient Streaming Language Models with Attention Sinks"*. arXiv:2309.17453, 2023.

> Zhenyu Zhang et al. *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"*. NeurIPS, 2023.
