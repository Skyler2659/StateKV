# StateKV Counterfactual — 官方分数人工核对（scoring audit）v1

日期：2026-08-26。只读分析；提取/分析脚本在 `tmp/scoring_audit_extract.py`、`tmp/scoring_audit_a.py`、`tmp/scoring_audit_b.py`、`tmp/scoring_audit_c.py`，中间产物在 `tmp/scoring_audit/`。

## 数据与方法

- 结果目录：`results/statekv_counterfactual/{cheapr2_fresh,cheapr2_cycle0,cheapr2_laq}_{longbench,multikey}_v1`，各 shard concat 后按 `(sample_id, policy, budget)` 去重。LongBench n=60（hotpotqa/2wikimqa/gov_report 各 20，indices 40–59，@256），multikey n=50（indices 280–329，@128/256/512）。
- Gold answer 来源：LongBench 三任务取 HF 缓存 jsonl 的 `answers` 字段（sample_id 的尾号即 jsonl 行号，已逐条核对）；multikey 用 `statekv/tasks.py::_synthetic_niah_multikey(seed=20260820, context_length=768, n_keys=4)` 重新生成，并用 CSV 里记录的 `needle_retrieval_accuracy` 全量校验：**1200/1200 行完全复现**，gold 无误。
- 官方分的真实计算方式（`statekv/oracle_policy_freegen.py::_metric_row`，closed-loop 复用）：
  - hotpotqa/2wikimqa：`ruler_score` → 对**整段 generation_text**（含模型自己追加的 `**Note:**` 推理段）做 gold 字符串的**小写子串包含**判断，0/100。
  - gov_report：`rouge_l`（自实现 LCS，whitespace token）×100。
  - multikey：`needle_retrieval_accuracy` = 每个 gold value 归一化子串包含的比例。
- "人工判断" = 我逐条读了全部 20×8（hotpotqa）+ 20×8（2wikimqa）条生成文本、6 条 gov_report 三组对照、8 条 multikey 案例，对照 gold 判定内容对错。

## 1. 文本一致性统计

| 对比 | 任务 | n | 逐字相同 | 归一化后相同 | 官方分相同 | 最终答案语义等价（人工） |
|---|---|---|---|---|---|---|
| R2c0 vs QK | hotpotqa | 20 | 3 | 3 | 20 | 20 |
| R2c0 vs QK | 2wikimqa | 20 | 3 | 4 | 16 | 17（差异见 §3） |
| R2 fresh vs R2 cycle0（同策略两次运行） | hotpotqa | 20 | 8 | 9 | 20 | 20 |
| R2 fresh vs R2 cycle0 | 2wikimqa | 20 | 8 | 9 | 20 | 20 |
| Full vs QK | hotpotqa | 20 | 2 | 2 | 19 | 18 |
| Full vs QK | 2wikimqa | 20 | 2 | 2 | 16 | ~16 |

要点：同一策略两次独立运行（R2 fresh vs cycle0）也只有 8/20 逐字相同——4bit 模型 decode 本身在措辞层面不稳定，但**最终答案在 20/20 上语义一致**。所以"文本逐字不同"是正常运行噪声，不构成评分差异；判断等价性要看答案串而非全文。

## 2. 评分误差的两个方向（具体案例）

官方分的本质是"gold 字符串是否作为子串出现在整段生成里"。这产生两类系统性误差：

### 判错但对（false negative）——措辞/粒度/别名不被 credit

1. **hotpotqa:41**（gold: `seasonal television specials, particularly its work in stop motion animation`）。FULL 答："best known for its seasonal television specials, usually done in stop motion animation" —— 内容**正确且最完整**，但 gold 长句不是子串 → 0 分。QK 答 "seasonal animated specials"（丢了 stop motion，半对）、R2 答 "stop motion animation"（半对），同样 0。**该样本上 FULL 明显优于所有压缩臂，官方分全部打 0，差异被抹平。**
2. **hotpotqa:42**（gold: `Dame Eileen June Atkins`）。H2O 和 LAQ 答 "Eileen Atkins" —— 同一人，**内容正确**，官方 0。而 FULL/QK/R2 答 "Juliet Aubrey" 是真错。官方分把"H2O/LAQ 答对"记成与其余臂相同的 0。
3. **hotpotqa:48**（gold 是长句 `Himalchuli has three main peaks: East (7893 m)...`）。问题问"哪座山更高"，**全部 8 个臂都答 "Himalchuli"，内容全对**，但因为 gold 长句不可能作为子串出现 → 8 臂全 0。这一条直接把所有臂的天花板各砍掉 5 分。
4. **2wikimqa:44**（gold: `Buenos Aires`）。H2O/LAQPP 答 "Argentina" —— 国家对、城市粒度不对，官方 0（半对）。
5. **2wikimqa:45**（gold: `Charlotte Amalie of Hesse-Kassel`）。FULL/QK/SnapKV/R2/LAQ 答 "Queen of Denmark (and Norway)" —— 这正是她的头衔，但此人也是题目中 Louise 本人的头衔，**按头衔指认含糊但不构成事实错误**；只有 H2O 在 Note 里写出了人名并正确推出"making her the mother-in-law"（官方 100）。其余臂 0。

### 判对但错/蒙对（false positive）——Note 推理段里出现 gold 字符串

1. **2wikimqa:54**（gold: `Tom Mix In Arabia`）。**所有 8 个臂的最终答案都是 "Space Probe Taurus"（错）**。但 QK 的生成带一段 "Step-by-Step Explanation"，其中列举了 "Tom Mix In Arabia is directed by Lynn Reynolds" —— gold 字符串出现在推理段里 → **QK 独得 100，其余 0**。这是纯评分巧合。
2. **2wikimqa:55**（gold: `Seven In The Sun`）。SnapKV/H2O 最终答案就是 "Seven In The Sun"（真对，100 应得）。**FULL 和 QK 的最终答案是 "Daughter Of The Jungle"（错）**，但 Note 里提到 "Sergio Bergonzelli, the director of *Seven in the Sun*, died in 2002" → 各拿 100。R2 也答错但 Note 没提到 gold 串 → 0。官方分在这条上**制造了 Full/QK > R2 的假差异**。
3. **2wikimqa:58**（gold: `Ruel Redinger`）。所有臂首行答案都是 "Peter Rosegger"（错）。QK 的 Note 写 "Ruel Redinger was born in 1892, making Ruel Redinger younger"（推理结论其实对了但首行答错）→ 100；SnapKV Note 里自我纠正 "**Answer:** Ruel Redinger" → 100（半对）；LAQ/LAQPP Note 含糊提及该名 → 100。FULL/H2O/R2 没有这段啰嗦 → 0。
4. **hotpotqa:58**（gold: `2010`）。H2O 首行答 "2003"（错），但 Note 里写 "aired from 2010 to 2019" → 100。唯一一例 hotpotqa false positive。

**共同机制**：模型生成 = 首行答案 + 重复的 `**Answer:**` 或一段 `**Note:**` 解释。子串匹配作用在**全文**，Note 越长、列举候选越多，越容易"蹭到" gold 字符串。各臂 Note 的啰嗦程度不同（QK 的 step-by-step 风格尤其容易列举多个候选实体），于是官方分部分地度量的是"解释段的候选覆盖率"，而不是最终答案对错。

### 人工重判 vs 官方分（0–100 尺度，20 条）

hotpotqa（判 41 的半对答案不计入）：

| | FULL | QK | SnapKV | H2O | R2f | R2c0 | LAQ | LAQPP |
|---|---|---|---|---|---|---|---|---|
| 官方 | 40 | 35 | 35 | 35 | 35 | 35 | 30 | 25 |
| 人工 | **50** | 40 | 40 | 40 | 40 | 40 | 40 | 30 |

2wikimqa：

| | FULL | QK | SnapKV | H2O | R2f | R2c0 | LAQ | LAQPP |
|---|---|---|---|---|---|---|---|---|
| 官方 | 45 | **55** | 50 | 45 | 45 | 45 | 45 | 40 |
| 人工 | 40 | 40 | 45 | 45 | **45** | **45** | 40 | 35 |

## 3. 问题 A 的结论

**A1（hotpotqa R2c0 vs QK 20/20 同分）**：文本仅 3/20 逐字相同，但差异全部在 Note 措辞和重复模式上；20/20 的最终答案语义等价（对的一样对，错的一样错，连错误答案的内容都高度一致，如 47 同答 "March 9, 1826"、54 同答 "IndyCar Series"）。**同分是真实的等价，不是评分巧合。**

**A2（评分误差规模）**：hotpotqa 上明确"判错但对"11 个臂-样本（41-FULL、42-H2O、42-LAQ、48 全 8 臂），"判对但错"1 个（58-H2O）；2wikimqa 上"判错但对/半对"约 8 个（44×2、45×6），"判对但错"4–6 个（54-QK、55-FULL、55-QK、58 的 QK/SnapKV/LAQ/LAQPP）。官方分把 hotpotqa 整体压扁约 5–10 分（48 号样本全臂误判贡献了 5 分），并在 2wikimqa 上虚增了啰嗦臂的分数。

**A3（2wikimqa Full=45 < QK=55）**：Full 错 QK 对的 3 条样本逐一核对——
- `2wikimqa:43`：FULL 答 "Lagu Kenangan" **真错**，QK 对。真实差异。
- `2wikimqa:54`：两者首行答案都是错的 "Space Probe Taurus"，QK 靠 Note 蹭到 gold 字符串 → **评分假象**。
- `2wikimqa:58`：两者首行都错，QK 靠 Note 蹭分 → **评分假象**。
另有 `2wikimqa:49` FULL/R2 答对 "Marie Laforêt" 而 QK 答错（真实，方向相反）。**净效果：官方显示的"QK 领先 Full 10 分"大部分是假象；人工重判下 QK 与 Full 打平（40:40），R2（45）反而略好于两者。** backbone 在该切片上确实 noisy（20 条里 8 条全臂答错），但评分假象是另一层独立问题。

**A4（总结论）**：hotpotqa/2wikimqa 上"方法间无差异"对 **R2c0 vs QK 这一对是真实的**（语义等价的答案），但对**臂间整体排序是评分分辨率不足 + 双向误差混合**：真实内容差异（41/42/48/49/51/57 等）存在却被 0/100 子串判据抹平或反转。人工重判下 hotpotqa 实际排序是 FULL > 所有压缩臂（50 vs 40），即"压缩无损"的结论在 hotpotqa 上其实**低估了 Full 的优势**；2wikimqa 上官方"QK>Full"不成立，"R2≈QK"在人工口径下变为"R2≥QK"。

## 4. 问题 B：gov_report R2c0 vs QK −0.46（CI[−0.75,−0.16]）

关键事实：**所有臂的 gov_report 生成都恰好 64 token，且全部在第二句中间被硬截断**（decode cap=64，而官方 LongBench gov_report 用 512）。所有生成都只是"**Summary of ...**\n\nThe report outlines/provides an overview of ..."式的泛化开场白，没有任何一臂（包括 FULL）写到具体数字、发现或建议。全体 rouge_l 均值 0.05–0.07（×100 后 5–7 分），R2c0=6.03、QK=6.49、FULL=6.23。

逐条读 rouge_l 差最大的 6 条（42/50/43/45/58/55，|Δ|≤0.016）：

- **内容覆盖三组（R2c0/QK/Full）逐条等价**：都正确识别了报告主题（NIV 签证审查、黑肺信托基金、部落宽带、FEMA 拨款管理、FHA 房贷保险、DOD 人力），都没有覆盖任何 gold 里的实质内容（金额、年份、GAO 发现）。
- 分差全部来自**措辞与 gold 的偶然重合度**。例 `gov_report:58`：gold 高频词是 "FHA insures private lenders against the possibility of borrowers defaulting on mortgages"；QK 写 "providing insurance for home mortgages... The primary function of the FHA is to insure home loans"（"insure"+"mortgages" 的 LCS 重合略多），R2c0 写 "insuring home mortgages... lenders are protected against losses if a borrower defaults"——语义一字不差地等价，rouge_l 0.100 vs 0.109。例 `gov_report:42`：R2c0 用 "screen and vet applicants for nonimmigrant visas (NIVs)"，QK 用 "the visa application process... visa adjudication process"，后者恰好与 gold 的 "adjudication" 重合更多。

**结论：−0.46 是指标噪声，不是实质退化。** 证据有三：(1) 逐条内容覆盖无差异；(2) R2c0（6.03）与 FULL（6.23）的距离比 QK（6.49）与 FULL 的距离更小——若以 Full 为金标准，R2c0 并不比 QK 差；(3) 64-token 截断使该任务对所有臂都只剩"主题句措辞 lottery"，单条 |Δ| 上限 0.016，−0.46 的均值差完全落在措辞敏感度范围内。这个 CI 统计上显著、实质内容上不显著。

## 5. 问题 C：multikey R2c0=70 vs QK=21（@256）抽查

抽 5 条 R2c0 对、QK 错的样本（280/298/307/291/319），gold 用再生成校验过（1200/1200 复现官方 needle 准确率）。**R2 的赢全部真实**。QK 的失败模式清晰一致：

1. **值损坏**：`280` 号 QK 把 gold `5312393` 写成 `5312933`（中间两位换位），并把 4 个 key 全部答成这同一个错值；
2. **值截断**：`291` QK 答 `17928`（gold `1792854` 的 5 位前缀）、`319` QK 答 `921371`（gold `9213716` 的 6 位前缀）；
3. **退化循环**：`307` QK 输出 "051908 / 21908 / 1908 / 7 / 22 22 22..."——完全是数字碎片复读，0/4；
4. **key 幻觉**：`298`/`319` QK 的叙述里把 key 写错（`lsozpn-xyz` vs 真 `lsozpn-xtpez`；`flsobk-emnv` vs 真 `flsobk-emhnv`）。

对照组 R2c0 的输出是干净的 "一行一个数字"，`280`/`319` 全 4 值精确正确。这与"QK 只保留当前 query 相关 KV、needle 数字 token 被逐出后无法精确复制 7 位数"的机制解释一致。评分本身（归一化子串包含，答案为独占一行的数字）**没有可钻的空子，70 vs 21 是真实检索能力差距**。

@512 上 LAQ>R2c0 的抽查（280/290/297）**同样真实**，且揭示了 R2 的残余失败模式：
- `290`：R2c0 第 4 个值 `6490099`（gold `6490999`，掉了一位 9）；LAQ 4/4 全对。
- `297`：R2c0 第 2 个值 `4111655`（gold `7111655`，首位 7→4）；LAQ 4/4。
- `280`：R2c0 只输出 1 个值就转入叙述段被截断；LAQ 4/4。
即 R2 在 @512 偶尔有**单个数字级损坏**（4 值中错 1 值的 1 位），LAQ 在这些样本上保持了完整数值保真。LAQ 的 @512 优势是真实的，但幅度是"个位数样本 × 单值单 digit"级别。

## 6. 哪种评分更可信（仅诊断建议，未重跑）

- **hotpotqa/2wikimqa**：当前的"整段子串包含"双向失真。最低成本修复是**只在首行/`**Answer:**` 行上做归一化包含判断**（Note 段不得分），并配合 alias 列表与"答案串包含 gold 或 gold 包含答案串"的双向判断（可解 42 的 "Eileen Atkins"、44 的粒度问题）；48 这类 gold 为长句的样本应改用 qa_f1 或首实体匹配。当前分数的方向性结论（R2c0 ≈ QK）经人工核对成立，但"压缩臂 ≈ Full"在 hotpotqa 上被高估。
- **gov_report**：先把 decode cap 从 64 提到官方 512 再谈评分；当前 64-token 截断下 ROUGE-L 不度量任何内容质量，任何臂间差异（包括本次 −0.46 和 CI）都不应解读为实质差异。
- **multikey**：现有 needle 准确率可信，无需更改。

## 附：产物清单

- `tmp/scoring_audit_extract.py`（数据合并 + gold 校验）、`tmp/scoring_audit_a.py` / `_b.py` / `_c.py`、`tmp/scoring_audit_dump_qa.py`
- `tmp/scoring_audit/{hotpotqa,2wikimqa}_all_generations.txt`（全部 20×8 生成文本，人工核对底稿）
- `tmp/scoring_audit/govreport_cases.txt`、`multikey_cases_256.txt`、`multikey_cases_512_laq.txt`
- `tmp/scoring_audit/*.pkl`、`multikey_gold.json`
