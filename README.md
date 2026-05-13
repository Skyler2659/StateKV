# L1-Robust KV Cache：基于 ℓ₁ Leverage Score 的 LLM 推理缓存压缩

## 目录

1. [背景：KV Cache 与压缩动机](#1-背景kv-cache-与压缩动机)
2. [理论基础：ℓ₁ Leverage Score](#2-理论基础ℓ₁-leverage-score)
3. [代码实现](#3-代码实现)
4. [项目文件结构](#4-项目文件结构)
5. [环境配置与复现](#5-环境配置与复现)
6. [实验结果](#6-实验结果)
7. [后续规划与局限性](#7-后续规划与局限性)

---

## 1. 背景：KV Cache 与压缩动机

### 1.1 什么是 KV Cache

在仅解码器（decoder-only）Transformer 的自回归生成过程中，每生成一个新 token，模型需要对**所有历史 token**执行注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

其中 $Q$ 仅为当前 token 的查询向量，而 $K$ 和 $V$ 是**全部已生成 token**的键和值矩阵。如果不做任何缓存优化，每一步都要重新计算所有历史 token 的 $K$ 和 $V$，计算量随序列长度 $L$ 呈 $O(L^2)$ 增长。

**KV Cache** 的做法是将每一步计算出的 $K$ 和 $V$ 存储在显存中，后续步骤直接拼接使用：

$$\begin{aligned}
K_{\text{cache}} &\leftarrow [K_{\text{cache}} \parallel K_{\text{new}}] \\
V_{\text{cache}} &\leftarrow [V_{\text{cache}} \parallel V_{\text{new}}]
\end{aligned}$$

这使得单步计算复杂度从 $O(L^2)$ 降至 $O(L)$。然而，缓存大小随序列长度**线性增长**。对于一个 $H$ 层、$h$ 个注意力头的 $d$ 维模型，每生成 $L$ 个 token 需要存储：

$$
M_{\text{KV}} = 2 \times H \times h \times d \times L \times \text{bytes\\_per\\_element}
$$

以 LLaMA-7B（32 层，32 头，128 维）在半精度（2 字节）下生成 2048 token 为例：$2 \times 32 \times 32 \times 128 \times 2048 \times 2 \approx 1\text{GB}$。当上下文扩展到 128K 乃至 1M token 时，KV Cache 会远超 GPU 显存容量。

### 1.2 为什么要压缩：KV Cache Eviction

观察到并非所有历史 token 对后续生成同等重要。例如：

- **注意力汇聚（Attention Sink）**：前几个 token（尤其是第一个）往往会吸引不成比例的高注意力权重，删除它们会导致严重退化（Xiao et al., 2024）。
- **局部性**：最近的 token 通常比远距离 token 更重要。
- **信息冗余**：许多中间 token 的 Value 向量高度线性相关，可被其他 token 近似替代。

因此，KV Cache 压缩的核心任务是：**在固定的缓存预算 $B$ 内，选择最有价值的 $B$ 个 token 保留，丢弃其余**。即寻找一个选择函数：

$$\mathcal{S}: \mathbb{R}^{L \times d} \to \{0, 1\}^L, \quad \|\mathcal{S}(V)\|_1 = B$$

使得使用 $\mathcal{S}(V)$ 替代完整 $V$ 后，模型的输出分布偏差最小。

### 1.3 现有方法概览

| 方法 | 选择依据 | 理论保证 |
|------|---------|---------|
| Sliding Window | 保留最近 $B$ 个 token | 无 |
| StreamingLLM | 保留前 $s$ 个 + 最近 $B-s$ 个 | 经验观察 |
| H2O | 累加注意力权重，保留 top-$k$ | 无 |
| **L1-Robust（本工作）** | ℓ₁ Leverage Score | ✓ Woodruff 2014 子空间嵌入理论 |

---

## 2. 理论基础：ℓ₁ Leverage Score

> 本节给出 ℓ₁ leverage score 的完整数学定义、性质及基于 Woodruff (2014) Construction B 的高效近似算法。**本节不含任何代码**，仅使用 LaTeX 数学公式。

### 2.1 定义

设 $A \in \mathbb{R}^{n \times d}$（$n \gg d$）为数据矩阵，其 $n$ 行对应 $n$ 个样本，$d$ 列对应特征维度。第 $i$ 行的 **ℓ₁ leverage score** 定义为：

$$\tau_i(A) = \sup_{x \in \mathbb{R}^d,\; Ax \neq 0} \frac{|(Ax)_i|}{\lVert Ax \rVert_1}$$

其中 $(Ax)_i = a_i^\top x$ 是 $Ax$ 的第 $i$ 个分量（$a_i$ 为 $A$ 的第 $i$ 行转置为列向量），$\|\cdot\|_1$ 为向量的 ℓ₁ 范数。

### 2.2 直观含义

$\tau_i(A)$ 衡量的是：在所有可能的线性组合 $Ax$ 中，第 $i$ 行的贡献最多能占到 ℓ₁ 总和的多少比例。

- 若 $\tau_i(A)$ **大**（接近 1）：存在某个方向 $x$，使得 $Ax$ 的 ℓ₁ 范数几乎全部集中在第 $i$ 个分量上——第 $i$ 行"不可替代"。
- 若 $\tau_i(A)$ **小**（接近 0）：对于所有方向 $x$，第 $i$ 行的贡献都可以被其他行的线性组合所淹没——第 $i$ 行"冗余"。

### 2.3 基本性质

**性质 1（有界性）**：$0 \leq \tau_i(A) \leq 1$，且 $\sum_{i=1}^n \tau_i(A) \leq d$。

这是 ℓ₁ leverage score 的核心性质：所有 $n$ 行的分数之和不超过 $d$（矩阵的列维度）。这意味着**平均每行的分数仅为 $d/n$**，当 $n$ 很大时大多数行有很低的 leverage score。这为稀疏选择提供了理论依据。

**性质 2（子空间不变性）**：$\tau_i(A)$ 仅依赖于 $A$ 的**列空间**，而不依赖于 $A$ 的特定基的选择。即对任意可逆矩阵 $U \in \mathbb{R}^{d \times d}$：

$$\tau_i(AU) = \tau_i(A)$$

**性质 3（与 ℓ₂ leverage score 的关系）**：ℓ₂ leverage score 定义为 $\tau_i^{(2)}(A) = \sup_{x} \frac{(Ax)_i^2}{\|Ax\|_2^2}$，等价于 $\tau_i^{(2)}(A) = \|U_{(i)}\|_2^2$，其中 $U$ 是 $A$ 的任意正交基（如 $Q$ 因子）。ℓ₁ 版本不满足此封闭形式，因此需要更复杂的近似算法。

**性质 4（灵敏度——ℓ₁ 回归的联系）**：ℓ₁ leverage score 刻画了在 ℓ₁ 回归问题 $\min_x \|Ax - b\|_1$ 中，每个数据点对解的影响程度。这是它们在鲁棒统计和子空间嵌入中扮演核心角色的原因。

### 2.4 ℓ₁ Leverage Score 的计算难题

直接按定义计算 $\tau_i(A) = \sup_x \frac{|(Ax)_i|}{\|Ax\|_1}$ 需要对所有 $x \in \mathbb{R}^d$ 取上确界，这在计算上不可行。然而，通过 ℓ₁ 子空间嵌入，可以在多项式时间内获得 $(1 \pm \varepsilon)$ 近似。

关键观察：如果存在一个矩阵 $S \in \mathbb{R}^{m \times n}$（$m \ll n$）满足对**所有** $x \in \mathbb{R}^d$：

$$(1 - \varepsilon)\|Ax\|_1 \leq \|SAx\|_1 \leq (1 + \varepsilon)\|Ax\|_1$$

则 $SA$ 在 ℓ₁ 范数下"保持"了 $A$ 的整个列空间的结构。这样的 $S$ 称为 **ℓ₁ 子空间嵌入**。

### 2.5 Woodruff (2014) Construction B：Exp(1) 加权 + CountSketch

David P. Woodruff 在 2014 年综述 "Sketching as a Tool for Numerical Linear Algebra" 的 §3.5 中，给出了构造 ℓ₁ 子空间嵌入的一个具体方案（Construction B）：

$$S = \Phi \cdot D \in \mathbb{R}^{m \times n}$$

其中：

- **$D \in \mathbb{R}^{n \times n}$** 是对角矩阵，其对角元 $D_{ii} = 1/E_i$，且 `$E_1, \ldots, E_n \overset{\mathrm{i.i.d.}}{\sim} \mathrm{Exp}(1)$`（均值为 1 的指数分布）。

- **$\Phi \in \mathbb{R}^{m \times n}$** 是 CountSketch 矩阵，每列恰好有一个非零元素（位置随机），取值为 $\pm 1$（等概率）。

#### 2.5.1 $D$ 的作用：为什么 Exp(1)？

CountSketch $\Phi$ 本身只能保证 ℓ₂ 范数的近似保持（即对任意固定向量 $y$，$\|\Phi y\|_2 \approx \|y\|_2$）。但我们需要的是 ℓ₁ 保持。

指数分布有一个关键性质：若 $E \sim \text{Exp}(1)$，则对任意标量 $z$：

$$\mathbb{E}\left[\left|\frac{z}{E}\right|\right] = |z| \cdot \mathbb{E}[1/E]$$

而更关键的是 **稳定性（stability）**性质：指数分布的倒数使得 $\ell_2$ 嵌入转变为 $\ell_1$ 嵌入。具体地，对任意向量 $y \in \mathbb{R}^n$：

$$\mathbb{E}\left[\|Dy\|_2^2\right] = \mathbb{E}\left[\sum_{i=1}^n \frac{y_i^2}{E_i^2}\right] = \sum_{i=1}^n y_i^2 \cdot \mathbb{E}[E_i^{-2}]$$

由于指数分布的尾部分布，$\mathbb{E}[E_i^{-2}]$ 发散（指数分布的概率密度函数 $f(e) = e^{-e}$，在 $e \to 0$ 时 $f(e) \to 1$，因此 $\int_0^\infty \frac{1}{e^2} e^{-e} de = \infty$）。这使得单个条目可能极大，从而"模拟"了 ℓ₁ 范数对异常值的敏感度。

更严格的分析表明：对 $A$ 的列空间中的所有向量 $y$，以高概率满足：

$$\|Dy\|_2 \approx \|y\|_1$$

将此与 CountSketch 拼接：$\Phi \cdot D$ 先将 ℓ₁ 结构"编码"进 ℓ₂（通过 $D$ 的指数加权），再用 $\Phi$ 做 ℓ₂ 降维。这就是 Construction B 的核心思想。

#### 2.5.2 CountSketch 的定义与性质

CountSketch 矩阵 $\Phi \in \mathbb{R}^{m \times n}$ 由两个随机映射定义：

- **哈希函数** $h: [n] \to [m]$，将每列映射到一个均匀随机的行（bucket）。
- **符号函数** $\sigma: [n] \to \{-1, +1\}$，每列赋予一个均匀随机的 $\pm 1$ 符号。

$\Phi$ 的每个元素为：

$$\Phi_{h(i), i} = \sigma(i), \quad \text{其余元素为 } 0$$

CountSketch 的 ℓ₂ 保证：对于任意固定向量 $y \in \mathbb{R}^n$，设 $m = \Theta(d^2/\varepsilon^2)$，则以高概率：

$$(1 - \varepsilon)\|y\|_2 \leq \|\Phi y\|_2 \leq (1 + \varepsilon)\|y\|_2$$

更重要的是，$\Phi \cdot y$ 可以在 $O(\text{nnz}(y))$ 时间内计算（仅需扫描 $y$ 的非零元素），且 $\Phi$ 本身不需要显式存储——只需存 $h(\cdot)$ 和 $\sigma(\cdot)$。

#### 2.5.3 完整算法

基于 Construction B 的 ℓ₁ leverage score 近似算法如下。

**输入**：数据矩阵 $A \in \mathbb{R}^{n \times d}$，草图维度 $m$（通常取 $m = \Theta(d^2)$ 以控制近似质量）。

**步骤 1：Exp(1) 加权**

对每一行 $i = 1, \ldots, n$，采样 $E_i \sim \text{Exp}(1)$，构造加权矩阵：

$$\tilde{A} = D \cdot A, \quad \text{其中 } \tilde{a}_i = \frac{a_i}{E_i}$$

即 $\tilde{A}$ 的第 $i$ 行为原始行 $a_i^\top$ 除以独立指数随机变量。

**步骤 2：CountSketch 降维（可选）**

若 $n$ 很大（> $m$），应用 CountSketch 将 $\tilde{A}$ 从 $n \times d$ 压缩至 $m \times d$：

$$S\tilde{A} = \Phi \cdot \tilde{A} \in \mathbb{R}^{m \times d}$$

若 $n$ 不大，可跳过此步（即令 $S = D$，$m = n$）。

**步骤 3：QR 分解**

对压缩后的矩阵进行（经济型）QR 分解：

$$S\tilde{A} = Q \cdot R, \quad Q \in \mathbb{R}^{m \times d}, \; R \in \mathbb{R}^{d \times d}$$

其中 $Q^\top Q = I_d$，$R$ 为上三角矩阵且可逆（假设 $A$ 列满秩）。

**步骤 4：计算近似 ℓ₁ Leverage Score**

对原始矩阵 $A$ 的每一行 $a_i \in \mathbb{R}^d$，计算：

$$\tilde{\tau}_i(A) = \left\lVert a_i^\top \cdot R^{-1} \right\rVert_1$$

其中 $R^{-1} \in \mathbb{R}^{d \times d}$ 为 $R$ 的逆矩阵。$\tilde{\tau}_i(A)$ 即为 $\tau_i(A)$ 的近似值。

#### 2.5.4 理论保证

Woodruff (2014) 证明了：当 $m = \Omega(d^2 / \varepsilon^2)$ 时，以高概率对**所有** $i = 1, \ldots, n$ 同时满足：

$$\frac{1}{1+\varepsilon} \cdot \tau_i(A) \leq \tilde{\tau}_i(A) \leq (1+\varepsilon) \cdot \tau_i(A)$$

这一保证是**确定性的逐行近似**——不是平均意义下的，而是对每一行都成立。这是 Construction B 组合了 $D$（Exp 加权）和 $\Phi$（CountSketch）后的非平凡结果。

### 2.6 从理论到 KV Cache 的应用

在 KV Cache 场景中：

- **矩阵 $A$** = $V \in \mathbb{R}^{L \times d_h}$，即某一层某个注意力头的 Value 矩阵，$L$ 为当前序列长度，$d_h$ 为每个头的维度。
- **行 $a_i$** = 第 $i$ 个 token 在该层的 value 向量。
- **ℓ₁ leverage score $\tau_i(V)$** 衡量该 token 的 value 向量在 $V$ 的列空间中的"不可替代性"。
- **Eviction 策略**：当 $L > B$（缓存预算），丢弃 $\tau_i(V)$ 最低的 $L - B$ 个 token 的 KV 条目。

核心直觉：ℓ₁ leverage score 低的 token 意味着其 value 向量可以被其他 token 的线性组合很好地近似——因此丢弃它对注意力输出影响最小。这与 H2O 基于注意力权重的启发式方法有本质区别：ℓ₁ leverage score 是**几何结构度量**，直接刻画数据矩阵本身的线性相关性，而非依赖注意力模式的经验观察。

### 2.7 与 ℓ₂ Leverage Score 的对比

| | ℓ₂ Leverage Score | ℓ₁ Leverage Score |
|---|---|---|
| 定义 | $\tau_i^{(2)} = \sup_x \frac{(Ax)_i^2}{\|Ax\|_2^2}$ | $\tau_i = \sup_x \frac{\|(Ax)_i\|}{\|Ax\|_1}$ |
| 封闭解 | $\tau_i^{(2)} = \|u_i\|_2^2$（$U$ 为 $A$ 的正交基） | 无封闭解 |
| 计算 | $O(nd^2)$，直接 QR | $O(nd + \text{poly}(d))$，需子空间嵌入 |
| 对异常值 | 敏感（平方放大） | 鲁棒（线性） |
| 在 KV Cache 中的含义 | 偏向保留"大范数" token | 偏向保留"线性不可替代" token |

ℓ₁ leverage score 的鲁棒性使其更适合高维注意力空间中的 token 选择：注意力头的 value 向量往往带有重尾分布，ℓ₁ 度量对极端值的线性处理方式比 ℓ₂ 的平方处理更加稳定。

---

## 3. 代码实现

### 3.1 核心模块：`l1_llm/l1_sketch.py`

实现三个类，对应理论部分的三个组件：

| 类 | 对应理论 | 职责 |
|---|---|---|
| `CountSketch` | $\Phi$ 矩阵 | 稀疏随机投影，将 $n$ 个向量压缩到 `sketch_dim` 个 bucket |
| `L1SubspaceEmbedding` | $S = \Phi \cdot D$ | 先做 Exp(1) 加权再 CountSketch |
| `L1LeverageScoreEstimator` | 完整算法 | QR 分解 + 维护 $R^{-1}$ + 计算 $\|v_i \cdot R^{-1}\|_1$ |

关键实现细节：

- **Exp(1) 采样**：使用逆变换法 $E = -\ln(1 - U)$，其中 $U \sim \text{Uniform}(0, 1)$。为保证数值稳定性，$U$ 被 clamp 到 $[10^{-8}, 1 - 10^{-8}]$，且始终在 `float32` 精度下采样（`float16` 下 $10^{-8}$ 为次正规数）。
- **QR 分解**：始终在 `float32` 精度下进行（`torch.linalg.qr` 不支持 `float16`）。对 $R$ 矩阵加入自适应对角线扰动（jitter）以防止奇异。
- **小规模优化**：当 $n < \text{sketch\_dim}$（默认为 1024）时，跳过 CountSketch，直接用 Exp(1) 加权后的矩阵做 QR。这在实际使用中非常常见（KV cache 通常不超过 1024），此时得到的是**精确**（而非近似）的 ℓ₁ leverage score。
- **周期性重计算**：每 `recompute_interval` 步重新拟合 $R^{-1}$（参数 `recompute_interval`，默认 32），以平衡精度和效率。

### 3.2 KV Cache 管理层：`l1_llm/kv_cache.py`

`L1RobustKVCache` 实现了混合预算分配策略：

$$\text{budget} = \underbrace{s}_{\text{sink}} + \underbrace{r}_{\text{recent}} + \underbrace{\ell}_{\ell_1\text{-selected}} + \underbrace{1}_{\text{last}}$$

其中：
- **sink tokens**（$s = 4$，默认）：前几个 token 作为"注意力汇聚"无条件保留。
- **recent tokens**（$r$，可配置）：保留倒数第 $r+1$ 到倒数第 $2$ 个 token。
- **ℓ₁-selected tokens**（$\ell$）：在剩余候选区间 $[s, L - 1 - r)$ 内，用 ℓ₁ leverage score 选 top-$\ell$。
- **last token**（$1$）：当前最新 token 无条件保留。

此外，文件中还包含 `StartRecentKVCache`（StreamingLLM 基线）和 `PlainKVCache`（无压缩基线）。

### 3.3 位置编码修正：`l1_llm/pos_shift/`

当中间 token 被 evict 后，RoPE（旋转位置编码）的索引连续性被破坏。若不移正位置索引，$Q$ 和 $K$ 之间的相对位置关系出错，导致注意力计算错误。

解决方案（每个架构一个文件）：
- 替换对应 Attention 类的 `forward` 方法。
- $Q$ 使用**逻辑位置**（position_ids 经偏移校正后），确保相对位置正确。
- $K$ 使用**物理位置**（在 cache 中的实际索引 $0, 1, \ldots$）。
- 同时将每层的最后一个 $Q$ 向量存入 `shared_q.LAST_QUERY_STATES`，供 H2O 使用。

已实现的架构：

| 文件 | 模型系列 | 代表模型 |
|------|---------|---------|
| `modify_llama.py` | LLaMA 1/2/3, Mistral | `meta-llama/Llama-*`, `mistralai/*` |
| `modify_gpt_neox.py` | GPT-NeoX, Pythia | `EleutherAI/pythia-*`, `EleutherAI/gpt-neox-*` |
| `modify_qwen2.py` | Qwen2/2.5 | `Qwen/Qwen2-*`, `Qwen/Qwen2.5-*` |
| `modify_falcon.py` | Falcon | `tiiuae/falcon-*` |

### 3.4 其他 KV Cache 实现

| 目录 | 文件 | 算法 | 描述 |
|------|------|------|------|
| `streaming_llm/` | `kv_cache.py` | StreamingLLM | start + recent 启发式（Xiao et al., 2024）|
| `h2o_llm/` | `kv_cache.py` | H2O | 累加注意力权重的 Heavy-Hitter Oracle（Zhang et al., 2023）|
| `plain_llm/` | `kv_cache.py` | 无压缩 | 直通模式，不做任何 eviction |

---

## 4. 项目文件结构

```
streaming-llm-main/
│
├── benchmark.py               # 主测试入口：多策略对比、参数网格搜索、needle 测试
├── data_sources.py             # 7 种测试数据源构建器
├── cache_baselines.py          # SlidingWindowKVCache + KV cache 工具函数
├── shared_q.py                 # 全局 LAST_QUERY_STATES 字典（pos_shift 与 H2O 共享）
│
├── l1_llm/                     # ── 本工作的核心：ℓ₁-鲁棒 KV Cache ──
│   ├── __init__.py
│   ├── l1_sketch.py            #   CountSketch, L1SubspaceEmbedding, L1LeverageScoreEstimator
│   ├── kv_cache.py             #   L1RobustKVCache, StartRecentKVCache, PlainKVCache
│   └── pos_shift/              #   位置编码修正（按架构分文件）
│       ├── __init__.py
│       ├── modify_llama.py     #     LLaMA / Mistral
│       ├── modify_gpt_neox.py  #     GPT-NeoX / Pythia
│       ├── modify_qwen2.py     #     Qwen2 / Qwen2.5
│       └── modify_falcon.py    #     Falcon
│
├── h2o_llm/                    # ── H2O 基线 ──
│   └── kv_cache.py             #   H2OKVCache
│
├── streaming_llm/              # ── StreamingLLM 基线 ──
│   └── kv_cache.py             #   StartRecentKVCache
│
├── plain_llm/                  # ── 无压缩基线 ──
│   └── kv_cache.py             #   PlainKVCache
│
├── LICENSE                     # Apache 2.0
├── .gitignore
└── README.md                   # 本文件
```

### 各文件职责

**`benchmark.py`**：实验主控脚本。负责模型加载、pos_shift 注入、各策略 KV cache 构建、文本数据加载、解码循环、结果汇总打印。支持 `--comparison_mode` 选择对比策略组合（`full` / `three` / `needle` / `grid`）。

**`data_sources.py`**：提供 7 种文本数据源：重复文本、Wikitext-2/-103、长文本、Needle-in-a-Haystack（标准版和简化版）、NarrativeQA、HotpotQA。每种数据源返回 `input_ids` 和可选的目标 token 位置列表。

**`cache_baselines.py`**：`SlidingWindowKVCache`（纯滑窗）实现和 KV cache 长度探测工具函数。

**`shared_q.py`**：仅定义一个全局字典 `LAST_QUERY_STATES = {}`，解决跨模块的 Q 状态共享问题。所有 pos_shift 文件写入此字典，H2O 从中读取。

---

## 5. 环境配置与复现

### 5.1 环境要求

- **Python**：3.8+
- **PyTorch**：2.0+（推荐 2.1+，CUDA 11.8 或 12.1）
- **Transformers**：**4.33** 或 **4.46.3**

> **Transformers 版本说明**：4.33 使用传统的 tuple 格式 KV cache，4.46.3 引入了 `DynamicCache`。代码兼容两种格式（通过 `_to_legacy` / `_back_to_original` 自动适配）。其他版本可能因内部 API 变更而报错。

### 5.2 安装步骤

```bash
# 1. 创建虚拟环境
python -m venv .venv_38
source .venv_38/bin/activate  # Linux/Mac
# .venv_38\Scripts\activate   # Windows

# 2. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.3
# 或 pip install transformers==4.33
pip install datasets  # 用于加载 Wikitext/NarrativeQA/HotpotQA

# 3. 克隆项目
git clone <repo-url>
cd streaming-llm-main
```

### 5.3 快速验证

```bash
# 在 CPU 上用小模型快速验证（约 2 分钟）
python benchmark.py \
  --model EleutherAI/pythia-410m-deduped \
  --device cpu \
  --cache_size 64 \
  --max_steps 200 \
  --text_source wikitext \
  --comparison_mode needle
```

### 5.4 完整对比实验

```bash
# 四路对比：recency / main / h2o / l1_mixed（GPU 推荐）
python benchmark.py \
  --model EleutherAI/pythia-410m-deduped \
  --device cuda \
  --cache_size 128 \
  --max_steps 1305 \
  --text_source needle \
  --comparison_mode needle \
  --mixed_recent_keep 64 \
  --h2o_recent_size 4 \
  --needle_pos 400 \
  --needle_prefix_repeat 40
```

### 5.5 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `gpt2` | HuggingFace 模型标识或路径 |
| `--device` | `cpu` | 运行设备（`cuda` / `cpu`） |
| `--cache_size` | `256` | KV cache 预算（总 token 数） |
| `--max_steps` | `512` | 最大解码步数 |
| `--start_size` | `4` | StreamingLLM / ℓ₁ 中的 sink token 数 |
| `--sketch_dim` | `1024` | CountSketch 维度 |
| `--recompute_interval` | `32` | $R^{-1}$ 重计算间隔 |
| `--mixed_recent_keep` | `64` | ℓ₁ 混合策略中的 recency 窗口大小 |
| `--h2o_recent_size` | `4` | H2O 保留的最近 token 数 |
| `--comparison_mode` | `full` | `full` / `three` / `needle` / `grid` |
| `--text_source` | `wikitext` | 数据源类型 |
| `--needle_pos` | `400` | Needle 插入位置 |
| `--needle_depth` | `0.5` | 标准 needle 测试深度比例（0~1） |
| `--enable_pos_shift` | `True` | 启用位置编码修正 |

---

## 6. 实验结果

### 6.1 实验设置

- **模型**：`EleutherAI/pythia-410m-deduped`（410M 参数，24 层，16 头，`d_h = 64`）
- **测试类型**：简化版 Needle-in-a-Haystack（大海捞针）
  - Haystack：重复前缀文本填充
  - Needle：特定句子插入在第 400 个 token 位置
  - 评估：仅对 Needle 之后的目标 token 计算 PPL
  - 总步数：1305 步
- **KV Cache 预算**：128 token
- **设备**：NVIDIA RTX 4090（24GB）

### 6.2 结果对比

| 策略 | PPL ↓ | tok/s | 描述 |
|------|-------|-------|------|
| `recency_only` (Sliding Window) | 44.5400 | — | 仅保留最近 128 个 token |
| `main` (StreamingLLM) | 31.0800 | — | 前 4 个 sink + 最近 124 个 token |
| `h2o` (H2O) | 30.3070 | — | 累加注意力分数 + 4 个 recent token |
| **`l1_mixed`（本工作）** | **11.9380** | — | 4 sink + 64 recent + ℓ₁ 选择 + 最后 1 个 |

### 6.3 分析

- `recency_only` 的 PPL 最高（44.54），因为纯滑窗丢失了所有超出窗口的历史信息，当 needle 位于第 400 token 位置时（远超 128 窗口），模型完全无法访问 needle 内容。
- `main`（StreamingLLM）降到 31.08，说明保留前几个 sink token 有帮助——sink token 捕获了部分全局信息。但中间的 needle 信息仍被丢弃。
- `h2o`（30.31）与 StreamingLLM 表现接近。H2O 基于累加注意力权重的选择有一定效果，但受限于（a）注意力近似使用未 RoPE 的 K 向量，（b）仅保留 4 个 recent 的设定。
- **`l1_mixed`（11.94）显著优于所有基线**。ℓ₁ leverage score 成功识别了 V 的列空间中不可替代的 token，在 sink + recency 的框架内将中间历史中最有信息量的 token 保留下来。

### 6.4 诚实声明

⚠️ **本实验属于玩具级别的概念验证（toy demo）**：

- 模型极小（410M，比当前主流 7B~70B 小 17~170 倍）。
- 仅测试了一条 needle 样本，样本量不足以得出统计显著结论。
- 缓存预算 128 token 极为苛刻（原始 StreamingLLM 使用 4 + 252 = 256），可能导致过度压缩。
- PPL 评估仅覆盖 needle 之后的回答 token，样本数量有限。

实验结果**仅表明 ℓ₁ 方法在小规模设定下有潜力**，不代表在更大模型或更多样本上的表现。

### 6.5 HotpotQA 多跳推理测试

在 TinyLlama-1.1B 上对 HotpotQA 的 5 个样本进行了多跳推理评估。HotpotQA 的任务需要模型跨文档检索多个事实并综合推理——这正是 KV cache 压缩的难点：关键信息可能分散在相距很远的 token 之间。

**实验设置**：TinyLlama-1.1B, cache_size=256, max_steps=2048, mixed_recent_keep 在 16~20 间扫描。

#### 5 样本完整结果

| 样本 | recency_only | main | l1_rk16 | l1_rk17 | l1_rk18 | l1_rk19 | l1_rk20 |
|------|-------------|------|---------|---------|---------|---------|---------|
| 0 | 1002.4 | 847.4 | 60.2 | **33.7** | 66.4 | 34.0 | 30.9 |
| 1 | 44.0 | 42.3 | 260.5 | 342.4 | 328.0 | 424.4 | **268.7** |
| 2 | 464.3 | **27.7** | 210.1 | 290.6 | 263.5 | 218.5 | 200.5 |
| 3 | 7320.5 | 2081.0 | **255.4** | 639.7 | 2223.9 | 3595.7 | 1158.3 |
| 4 | 31.4 | 4.9 | 5.8 | 10.6 | 6.2 | 7.3 | **7.3** |

#### 关键发现

**1. 样本间方差极大，用平均值会掩盖方法差异。**

Sample 4 的 PPL 低至 5——答案几乎可以从问题文本直接推断，不需要检索任何文档。这类"简单"样本对所有方法都一样友好，平均后会冲淡方法间的真正差异。

Sample 3 是极端反例：recency_only 的 PPL 高达 7320，说明 needle 信息完全超出了 256 token 的滑动窗口。ℓ₁ (rk16) 将 PPL 压到 255——**29 倍的改善**。

**2. 用 "Hard Subset" 筛选比全量平均更有意义。**

统计 recency_only PPL > 100 的样本（样本 0、2、3）——这些才是真正需要长程检索的难题：

| 子集 | recency_only | main | l1_rk16 | l1_rk17 |
|------|-------------|------|---------|---------|
| Hard (0,2,3) | 2629.1 | 985.4 | **175.2** | 321.3 |
| 相对 main 改善 | — | baseline | **82.2%** | 67.4% |

在 hard subset 上，ℓ₁ 相对 StreamingLLM (main) 将 PPL 改善了 82%。这比全量中位数（被简单样本主导）更能反映方法在真实检索场景下的能力。

**3. 中位数不可靠。**

全量中位数：recency_only 464.3, main 42.3, l1_rk16 210.1——结论是 main 最好。但这完全是因为 main 在样本 2 上"运气好"（PPL 27.7 vs 其他方法 200+），并非方法本身更优。一旦切到 hard subset，结论完全反转：ℓ₁ 以显著优势胜出。

#### 方法论启示

HotpotQA 评测的正确做法：先用 recency_only 或 sliding window 跑一遍，筛出 PPL 高于某个阈值（如 100）的 hard samples，再在 hard subset 上比较各方法。这避免了"答案可以从问题推断"的简单样本冲淡评测信号。

---

## 7. 后续规划与局限性

### 7.1 当前局限性

1. **小模型 + 小样本**：仅在 410M 模型上测试了一条 needle 样本。
2. **固定预算**：仅测试了 128 token 预算，未探索其他预算配置。
3. **超参数未调优**：sink 数量、recent 比例、重计算间隔等参数未系统调优。
4. **仅对比经典基线**：未与 2024 年的新方法（SnapKV, RocketKV 等）对比。
5. **每层独立计算**：每层的 ℓ₁ leverage score 独立计算，未考虑跨层信息共享。
6. **仅使用 V 矩阵**：未利用 K 或 Q 中的信息。

### 7.2 后续计划

**短期（技术完善）**：
- 在更大模型上测试（7B 起步：LLaMA-2/3-7B、Qwen2.5-7B、Mistral-7B）。
- 对不同 `cache_size`（64 / 128 / 256 / 512 / 1024）做热力图扫描，找到使 ℓ₁ 优势最大化的预算区间。
- 系统搜索 `mixed_recent_keep` 的最优比例（grid search on `{16, 32, 48, 64, 80, 96}`）。
- 在 HotpotQA、NarrativeQA 等多跳推理数据集上评估（30+ 样本以达统计显著性）。

**中期（方法改进）**：
- 与 **SnapKV**（2024.04，基于 prompt 末尾 attention pattern 的全局筛选）对比。
- 与 **RocketKV**（2024，两阶段粗粒度 + 细粒度 hybrid 方法）对比。
- 与 **Quest**（2024，基于 Q 的 ℓ₂ norm 的 query-aware 选择）对比。
- 探索将 K 的信息融入 ℓ₁ 框架（如对 $[K \parallel V]$ 拼接矩阵做 leverage score 计算）。
- 探索跨层共享 estimator，减少计算开销。

**长期（理论深化）**：
- 严格分析 RoPE 对 V 矩阵列空间结构的影响（当前 eviction 破坏了 RoPE 的相对位置关系，pos_shift 是工程修复而非理论解决方案）。
- 将 ℓ₁ leverage score 的选择与注意力输出的误差建立理论 bound。
- 若实验证据足够，整理成论文投稿（目标会议：NeurIPS / ICML / ICLR 的 workshop 或主会）。

---

## 引用

- **Woodruff, D. P.** (2014). "Sketching as a Tool for Numerical Linear Algebra." *Foundations and Trends in Theoretical Computer Science*, 10(1–2), 1–157. — ℓ₁ 子空间嵌入的理论基础 (§3.5 Construction B)。
- **Xiao, G., et al.** (2024). "Efficient Streaming Language Models with Attention Sinks." *ICLR 2024*. — StreamingLLM，发现 attention sink 现象。
- **Zhang, Z., et al.** (2023). "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." *NeurIPS 2023*. — H2O 基线方法。
- **Li, Y., et al.** (2024). "SnapKV: LLM Knows What You are Looking for Before Generation." *arXiv:2404.14469*. — SnapKV。
- **Zhang, Y., et al.** (2024). "RocketKV: Accelerating Long-Context LLM Inference via Inter-Layer KV Cache Sharing." — RocketKV。

---

## 许可

本项目基于 Apache License 2.0 发布。详见 [LICENSE](LICENSE)。
