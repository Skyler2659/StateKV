# StateKV Cheap-R2 最终合并分析与定型报告（finalization v1）

日期： 2026-08-26。Branch: `codex/statekv-counterfactual-utility`。
范围： 方法与数据全部冻结后的纯合并分析。不重跑主实验、不调参；唯一新增计算是
3 条 multikey 样本的 LAQ 臂 diagnostic sink 补跑（只读仪表，逐位确定性已验证，
见 §6），用于 token-level case study。

- 模型： Qwen3-8B-4bit (mlx-community, revision 545dc42)，MLX 后端。
- 协议： strict pure eviction（所有压缩臂 `strict_pure_eviction=True`，
  `recoverable_cold_tokens=0`，`peak_active_cache_tokens ≤ budget`，全部核验通过，
  无缺样本、无 shard 重叠）；sink=4 / recent=32；64 decode cycles。
- 方法定型： **Cheap-R2 = one-shot query-onset H=32 target-model full-KV rollout
  + strict physical eviction**（cycle 0 计算一次 future utility，之后不再刷新）。
- 数据： fresh panel — multikey n=50 (280–329) @128/256/512；hotpotqa / 2wikimqa /
  gov_report n=20 @256（gov 含 c0c/lc top-up，从 n=10 补到 n=20）。
- 统计： 全部 paired 分析为样本级 bootstrap，20000 次重抽样，seed 20260820，
  报 95% percentile CI。

## 0. 完整性审计

| 数据集 | 臂 × budget | 期望 n | 实际 n | 违例 |
|---|---|---|---|---|
| multikey fresh (s0/s1) | Full/QK/SnapKV/H2O/R2-f16 × {128,256,512} | 50 | 50 | 0 |
| multikey cycle0 (c0a/c0b) | R2-cycle0 × {128,256,512} | 50 | 50 | 0 |
| multikey LAQ (la/lb) | LAQ/LAQ++ × {128,256,512} | 50 | 50 | 0 |
| longbench fresh (s0/s0b/s1) | Full/QK/SnapKV/H2O/R2-f16 × 256 | 20/任务 | 20 | 0 |
| longbench cycle0 (c0a/c0b/c0c) | R2-cycle0 × 256 | 20/任务 | 20 | 0 |
| longbench LAQ (la/lb/lc) | LAQ/LAQ++ × 256 | 20/任务 | 20 | 0 |

shard 合并均先 concat 再 `drop_duplicates(sample_id, policy, budget)`
（cycle0_longbench 去重 20 行、laq_longbench 去重 30 行，为 top-up 前旧 tag 的重叠）。
FULL_CACHE_REFERENCE 在三个 longbench run 目录中各自重跑，逐样本核对分数完全一致
（0 处不一致），主表取去重后单份。paired 样本交集在所有对比中均为满 n。

## 1. Phase 1 结论： cycle0-only ≈ f16 → 方法定型 one-shot

R2-cycle0（仅 query onset 刷新一次）vs R2-f16（每 16 步刷新），paired：

| 任务@budget | n | W/T/L | Δ (cycle0−f16) | 95% CI |
|---|---|---|---|---|
| multikey@128 | 50 | 1/49/0 | +0.5 | [0.0, +1.5] |
| multikey@256 | 50 | 2/46/2 | +0.5 | [−2.0, +3.5] |
| multikey@512 | 50 | 0/49/1 | −0.5 | [−1.5, 0.0] |
| hotpotqa@256 | 20 | 0/20/0 | 0.0 | [0, 0] |
| 2wikimqa@256 | 20 | 0/20/0 | 0.0 | [0, 0] |
| gov_report@256 | 20 | 8/3/9 | −0.07 | [−0.26, +0.10] |

成本（multikey@256）： cycle0 75.5s = **1.41×QK**，f16 125.2s = 2.34×QK。
结论： 后续刷新几乎不贡献分数（机制上稳态 current↔future top-B overlap 0.996，
见 §6 机制报告引用），**方法定型为 one-shot query-onset H32 lookahead**，
比 f16 再省 40% 墙钟。

## 2. LongBench boundary validation（one-shot 简化后结论不变）

| 任务@256 | Full | QK | LAQ | LAQ++ | R2-cycle0 | R2-f16 | cycle0 vs f16 |
|---|---|---|---|---|---|---|---|
| hotpotqa | 40.0 | 35.0 | 30.0 | 25.0 | 35.0 | 35.0 | 完全相同 |
| 2wikimqa | 45.0 | 55.0 | 45.0 | 40.0 | 45.0 | 45.0 | 完全相同 |
| gov_report | 6.25 | 6.49 | 6.24 | 6.01 | 6.03 | 6.10 | −0.07 [−0.26,+0.10] |

hotpotqa/2wikimqa 上 cycle0 与 f16 逐样本完全相同（20/20 tie）；
gov_report 差异在噪声内。**one-shot 简化不改变任何 LongBench 结论。**

## 3. LAQ / LAQ++ protocol 摘要

完整 protocol 见 `docs/evidence/laq_implementation_protocol.md`。要点：

- 论文： Lookahead Q-Cache (EMNLP 2025, arXiv:2505.20334)；参考实现只读参考
  （tmp/laq_reference/，未复用代码）。
- 算法： 先用 SnapKV（obs window 32）把 full prefill 压缩到同一目标 budget B，
  在**压缩后的 degraded cache** 上贪心生成 8 个 pseudo-response token
  （horizon 固定为论文默认 8），取其 post-RoPE query 组成 Q-Cache；用 Q-Cache
  对**原始 full prefill keys** 打 q·k 分，one-shot top-B re-evict。LAQ++ 额外加
  最后 8 个 prompt 位置的 query（共 16 个 observation query）。
- Faithful 部分： one-shot 决策点（cycle 0 = prefill eviction）、lookahead
  budget = 目标 budget（照 repo `run_LAQ.sh`）、对原始 full keys 打分、全 32 个
  query head、无信息优势（只见 prompt 和自己生成的 8 个 pseudo token）。
- 框架强制的 deviation: shared cross-layer top-core（而非参考实现的
  per-head/per-layer top-k，与 R2 走同一个 `rank_and_margin`）；paper 版 raw
  pre-softmax q·k（参考实现实际用 softmax+maxpool7）；SnapKV pooling kernel 用
  框架冻结值 63；lookahead 不早停（恰好 8 token）；GQA 按 kv-head 映射（代数等价）。
- 公平性 invariant 与 R2 完全一致： 同一 sink/recent 保护、同一选择器、strict
  physical eviction、lookahead 全部成本计入 `causal_teacher_time_s` / wall time。

## 4. 主表（mean official_score, @256；multikey 另给 @128/@512）

| 任务@budget | Full | QK | SnapKV | H2O | LAQ | LAQ++ | **R2-cycle0** | R2-f16 |
|---|---|---|---|---|---|---|---|---|
| multikey@128 | 82.0 | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | **48.5** | 48.0 |
| multikey@256 | 82.0 | 21.0 | 0.0 | 5.0 | 29.5 | 21.0 | **70.0** | 69.5 |
| multikey@512 | 82.0 | 25.5 | 55.0 | 16.0 | **84.5** | 68.5 | 81.0 | 81.5 |
| hotpotqa@256 | 40.0 | 35.0 | 35.0 | 35.0 | 30.0 | 25.0 | 35.0 | 35.0 |
| 2wikimqa@256 | 45.0 | 55.0 | 50.0 | 45.0 | 45.0 | 40.0 | 45.0 | 45.0 |
| gov_report@256 | 6.25 | 6.49 | 6.09 | 6.23 | 6.24 | 6.01 | 6.03 | 6.10 |

n: multikey 50/臂，longbench 20/臂。multikey 分数为 needle retrieval accuracy×100；
gov_report 为 ROUGE-L 类官方分。

## 5. Paired 分析（R2-cycle0 − 对手， bootstrap 20000×, seed 20260820）

### 5.1 vs LAQ（核心对比）

| 任务@budget | n | W/T/L | Δ | 95% CI |
|---|---|---|---|---|
| multikey@128 | 50 | 49/1/0 | **+48.5** | [+41.0, +56.0] |
| multikey@256 | 50 | 33/14/3 | **+40.5** | [+29.5, +51.0] |
| multikey@512 | 50 | 6/35/9 | −3.5 | [−11.5, +4.0] |
| hotpotqa@256 | 20 | 2/17/1 | +5.0 | [−10.0, +20.0] |
| 2wikimqa@256 | 20 | 1/18/1 | 0.0 | [−15.0, +15.0] |
| gov_report@256 | 20 | 7/1/12 | −0.21 | [−0.57, +0.13] |

### 5.2 vs LAQ++

| 任务@budget | n | W/T/L | Δ | 95% CI |
|---|---|---|---|---|
| multikey@128 | 50 | 49/1/0 | +48.5 | [+41.0, +56.5] |
| multikey@256 | 50 | 35/15/0 | +49.0 | [+39.0, +59.0] |
| multikey@512 | 50 | 21/21/8 | +12.5 | [+4.5, +20.5] |
| hotpotqa@256 | 20 | 2/18/0 | +10.0 | [0.0, +25.0] |
| 2wikimqa@256 | 20 | 2/17/1 | +5.0 | [−10.0, +20.0] |
| gov_report@256 | 20 | 11/1/8 | +0.02 | [−0.43, +0.47] |

### 5.3 vs QK

| 任务@budget | n | W/T/L | Δ | 95% CI |
|---|---|---|---|---|
| multikey@128 | 50 | 48/2/0 | +47.5 | [+40.0, +55.5] |
| multikey@256 | 50 | 36/14/0 | +49.0 | [+39.0, +58.5] |
| multikey@512 | 50 | 41/8/1 | +55.5 | [+46.5, +63.5] |
| hotpotqa@256 | 20 | 0/20/0 | 0.0 | [0, 0] |
| 2wikimqa@256 | 20 | 1/16/3 | −10.0 | [−30.0, +10.0] |
| gov_report@256 | 20 | 5/0/15 | **−0.46** | **[−0.75, −0.16]** |

### 5.4 gov_report non-inferiority 判定（n=20，含 top-up）

R2-cycle0 vs QK: Δ=−0.46, 95% CI [−0.75, −0.16]，**CI 上界 < 0 ——
non-inferiority 不成立**。vs Full: Δ=−0.22 [−0.54, +0.07]（不显著）；
vs LAQ: Δ=−0.21 [−0.57, +0.13]（不显著）。解读： 在 gov_report 这个控制任务上，
R2-cycle0 相对最强 baseline（QK）存在小幅但统计上可分辨的劣势（约 −7% 相对分）;
相对 Full cache 与 LAQ 则无显著差异。**这是必须写入论文的限制，不能包装。**

## 6. Quality–cost Pareto (@256)

横轴 = mean wall_time / mean wall_time(QK)；纵轴 = mean score。

| 任务 | 臂 | score | wall (s) | ×QK |
|---|---|---|---|---|
| multikey | QK | 21.0 | 53.4 | 1.00 |
| | SnapKV | 0.0 | 53.9 | 1.01 |
| | H2O | 5.0 | 52.1 | 0.98 |
| | LAQ | 29.5 | 65.9 | 1.23 |
| | LAQ++ | 21.0 | 67.0 | 1.25 |
| | **R2-cycle0** | **70.0** | **75.5** | **1.41** |
| | R2-f16 | 69.5 | 125.2 | 2.34 |
| hotpotqa | QK | 35.0 | 52.6 | 1.00 |
| | LAQ | 30.0 | 92.7 | 1.76 |
| | R2-cycle0 | 35.0 | 113.4 | 2.15 |
| | R2-f16 | 35.0 | 164.1 | 3.12 |
| 2wikimqa | QK | 55.0 | 73.3 | 1.00 |
| | LAQ | 45.0 | 99.4 | 1.36 |
| | R2-cycle0 | 45.0 | 102.6 | 1.40 |
| | R2-f16 | 45.0 | 186.4 | 2.54 |
| gov_report | QK | 6.49 | 51.5 | 1.00 |
| | LAQ | 6.24 | 51.6 | 1.00 |
| | R2-cycle0 | 6.03 | 54.9 | 1.07 |
| | R2-f16 | 6.10 | 91.6 | 1.78 |

要点：

1. **multikey 上 R2-cycle0 严格 dominate 所有压缩臂**: 分数最高（70.0），
   成本仅 1.41×QK；LAQ 1.23× 但只得 29.5。R2-f16 被 cycle0 严格 dominate
   （同分、1.7× 墙钟）。
2. longbench 上 R2-cycle0 与 QK 同分但 1.4–2.2× 成本 —— 无收益，如实呈现。
3. gov_report 上 R2-cycle0 成本与 LAQ 相当（1.07× vs 1.00×），分数略低。

## 7. Token-level case study（multikey, 3 样本）

（见 §7 表；LAQ 臂逐 token 数据由 diagnostic sink 补跑获得，逐位确定性验证：
补跑 npz 的 official_score 与冻结 sample_summary 完全一致。）

3 条 R2-cycle0 赢 LAQ 的样本（287 为机制报告已有的 clean case，280/307 为
R2c0=100 / LAQ=25 的代表性赢样本），budget=256（core 220）。每样本 4 条 needle
span（87–88 个 needle token）。LAQ per-token 分数来自 cycle-0 `shared_scores`
（补跑 npz 的 official_score 与冻结值逐样本一致： 25.0/25.0/25.0）。

| 样本 | needle token 数 | current QK rank (med / min) | R2 future-best rank (med / min) | LAQ rank (med / min) | R2 保留率 | LAQ 保留率 | cf-QK 保留率 | 结局 R2c0 / LAQ / LAQ++ / QK |
|---|---|---|---|---|---|---|---|---|
| 287 | 87 | 260 / 47 | 32 / 1 | 177 / 13 | 79.3% | 65.5% | 46.0% | **100** / 25 / 25 / 0 |
| 280 | 88 | 217 / 37 | 29.5 / 1 | 152.5 / 18 | 76.1% | 68.2% | 50.0% | **100** / 25 / 25 / 0 |
| 307 | 87 | 247 / 36 | 34 / 2 | 175 / 32 | 81.6% | 62.1% | 48.3% | **100** / 25 / 0 / 0 |

Per-span 保留率揭示失败模式（LAQ / R2）:

| 样本 | span0 | span1 | span2 | span3 |
|---|---|---|---|---|
| 280 | 0.95 / 0.68 | **0.52 / 0.83** | **0.45 / 0.64** | 0.81 / 0.90 |
| 287 | 0.86 / 0.68 | **0.55 / 0.82** | **0.57 / 0.76** | 0.64 / 0.91 |
| 307 | 0.91 / 0.68 | **0.50 / 0.86** | **0.45 / 0.82** | 0.62 / 0.90 |

解读：

1. LAQ 的 pseudo-lookahead 确实部分修复了 needle 的可见性（median rank 从
   current-QK 的 217–260 提升到 152–177），但不够——中段的 needle span
   （span1/span2）仍大片跌出 top-220（保留率仅 45–57%），而 R2 的 full-KV
   future rank（median 29–34）把每个 span 护在 64–91%。
2. 结局完全对应： R2-cycle0 找回 4/4 needle（100 分）; LAQ 只保住 span0
   （25 分 = 1/4）; LAQ++ 不更好甚至更差。
3. 这正是"degraded-cache 自举"在 token 层面的投影： LAQ 的 8 个 pseudo token
   是在已被 SnapKV 压到 256 的 cache 上生成的，被压掉的中间 needle 无法进入
   pseudo-response 的注意力，因而也进不了 Q-Cache 的打分——laq 看不到自己已经
   丢掉的东西。R2 的 rollout 在 full KV 上进行，不存在这个盲区。

## 8. CASE 判定

**CASE A 成立，但限定在 tight-budget multikey 上。**

- 前提成立： cycle0 ≈ f16（§1，全部任务 paired CI 紧贴 0），方法已定型 one-shot。
- 核心对比成立： multikey@128/+48.5 [+41,+56]、@256/+40.5 [+29.5,+51]，
  R2-cycle0 显著优于最近邻 LAQ（vs LAQ++ 优势更大）。
- 边界（如实）:
  1. **multikey@512 与 LAQ 打平**（−3.5, CI [−11.5,+4.0]；LAQ 84.5 vs R2c0 81.0）。
     budget 一松，LAQ 的 8-token pseudo-lookahead 就足够找到 needle，
     full-KV foresight 的增量消失。对 LAQ++ 仍 +12.5 [+4.5,+20.5]。
  2. **hotpotqa/2wikimqa 上对所有臂无差异**（与 QK 20/20 全 tie 或噪声内）。
  3. **gov_report 上 non-inferiority vs QK 不成立**（§5.4）。

## 9. Paper positioning（一句话 + 机制解释 + limitations）

**Positioning**: Cheap-R2 = one-shot target-model full-KV foresight at query onset —
在 query 到达时用**未经压缩的完整 KV** 做 H=32 步 rollout，直接测出每个 token 的
future utility，一次定型、strict physical eviction。相对最近邻 LAQ 的本质区别：
LAQ 必须先在 **degraded (已压缩) cache** 上自举生成 pseudo-lookahead 再来筛选
cache——在 tight budget 下这是循环依赖： 压缩已经丢掉的 evidence，pseudo-response
无从谈起，于是 LAQ 在 multikey@128/@256 上接近 0–29.5 分（自举崩溃）;
Cheap-R2 用 full-KV rollout 绕开了这个自举瓶颈。机制侧证据
（`notes/statekv_counterfactual_rank_migration_v1.md`）: 有价值的 future-demand
shift 集中在 cycle 0（稳态 overlap 0.996），且 multikey cycle-0 的
current-dormant→future-important token 恰恰集中在 needle 上（47.3% vs filler 20%）,
R2 的 needle 保留率 93.1% vs counterfactual-QK 86.2%。

**Limitations**:

1. 优势区间限于 aggressive budget + retrieval-critical 负载；@512 与 LAQ 打平。
2. LongBench QA（hotpotqa/2wikimqa）上无任何方法间差异，future foresight 无增量。
3. gov_report 上相对 QK 有小幅统计显著劣势（−0.46, CI 上界 <0）；
   相对 Full/LAQ 不显著。
4. 全部实验为单模型（Qwen3-8B-4bit）、单 decode 长度（64 cycles）、train panel;
   跨模型/跨长度泛化未验证。
5. 成本仍是 1.41×QK（multikey）, 不是免费午餐； 在 latency-critical 场景需权衡。
6. LAQ port 存在 framework-forced deviations（§3）; 结论依赖该 port 的
   faithfulness，已逐条记录在 protocol note。

## 附： 复现入口

- 合并与分析脚本： `tmp/final_audit.py`, `tmp/final_analysis.py`,
  `tmp/final_case_study.py`（分析脚本，不进 git）。
- 结果目录（gitignored）: `results/statekv_counterfactual/cheapr2_{fresh,cycle0,laq}_{multikey,longbench}_v1/`。
- 机制报告： `notes/statekv_counterfactual_rank_migration_v1.md`。
- LAQ protocol: `docs/evidence/laq_implementation_protocol.md`。
- LAQ case diagnostic 补跑： `tmp/run_laq_case_diagnostic.py`（只写
  rank_migration npz，不触碰 closed_loop 冻结产物）。
