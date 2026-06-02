# LLM Judge 精度评估结果

日期：2026-06-02

本文档记录在 RVS-Movie 和 RVS-Ego 小子集上接入本地 LLM-as-judge 后的第一轮精度评估结果。它用于回答上一轮提出的问题：当前方法的“精度”到底如何体现。

---

## 1. 为什么需要 LLM Judge

RVS-Movie / RVS-Ego 是 open-ended QA，答案不是固定类别，因此不能直接用分类 accuracy。此前我们使用：

1. token-overlap F1；
2. rule proxy；
3. dense vs semantic 的 token-F1 win/tie/loss。

这些指标能发现明显崩坏，但都偏弱。尤其 token-F1 会低估同义改写，也会惩罚更长但语义正确的回答。

因此本轮新增 LLM judge：

```text
scripts/judge_qa_pairwise.py
```

输入：

```text
question
reference answer
dense prediction
sparse semantic stream prediction
```

输出：

```json
{
  "dense_correct": true/false,
  "sparse_correct": true/false,
  "relative": "better/same/worse",
  "reason": "short reason"
}
```

---

## 2. Judge 模型与设置

本轮使用服务器本地模型：

```text
judge model: /home/mllm/models/Qwen2.5-VL-7B-Instruct
mode: text-only judge
GPU: A100
```

注意：

1. 这是本地 7B judge，不是最终官方 evaluator。
2. 由于输入只包含文本答案，没有给 judge 看视频，因此它判断的是“预测答案是否覆盖 reference answer”，不是重新理解视频。
3. 该指标比 token-F1 更接近 open-ended QA accuracy，但仍需要后续用更强 judge 或官方 benchmark evaluator 复核。

本轮还尝试过 `Qwen3-0.6B`，但 0.6B judge 出现明显不稳定和自相矛盾判断，因此只作为 smoke test，不作为结果依据。

---

## 3. 脚本修正

为了让 judge 更适合开放式回答，本轮对 judge 脚本做了两点修正：

1. 支持 `Qwen2.5-VL-7B-Instruct` 作为 text-only judge。
2. 在 prompt 中明确：

```text
Do not penalize harmless extra details unless they contradict the reference answer.
```

原因是早期 judge 会把 “A kitchen with wooden floor” 判为不匹配 “A kitchen”，这会系统性低估包含额外无害细节的回答。

此外，脚本会保留模型原始 `relative_raw`，并根据 `dense_correct/sparse_correct` 归一化 `relative`：

```text
dense correct, sparse wrong -> worse
dense wrong, sparse correct -> better
```

最终分析主要使用 correctness win/tie/loss：

```text
win  = sparse_correct 且 dense_wrong
loss = dense_correct 且 sparse_wrong
tie  = 二者都正确或二者都错误
```

这个口径比模型直接生成的 better/same/worse 更稳。

---

## 4. LLM Judge 主结果

| Dataset | Config | Dense Correct | Sparse Correct | Delta | Correctness W/T/L | Speedup | Token Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RVS-Movie | r16/t0.1 | 20.8% | 33.3% | +12.5% | 3 / 21 / 0 | 5.86x | 85.3% |
| RVS-Movie | r64/t0.3 | 20.8% | 37.5% | +16.7% | 4 / 20 / 0 | 8.59x | 93.1% |
| RVS-Ego | r16/t0.1 | 54.2% | 62.5% | +8.3% | 2 / 22 / 0 | 3.17x | 84.5% |
| RVS-Ego | r64/t0.3 | 54.2% | 58.3% | +4.2% | 2 / 21 / 1 | 3.69x | 93.8% |

这里的 W/T/L 是 correctness win/tie/loss，不是 token-F1 win/tie/loss。

---

## 5. 结果解读

### 5.1 现在“精度”可以更清楚地体现

之前我们只能说：

```text
token-F1 proxy 没有明显下降。
```

现在可以更具体地说：

```text
在 Qwen2.5-VL-7B text-only judge 下，Semantic Stream 的 judged correct rate 与 dense 持平或更高；
同时获得 3.17x-8.59x 视觉编码加速和 84.5%-93.8% token reduction。
```

### 5.2 RVS-Movie dense 本身很弱

Movie 上 dense judged correct rate 只有 20.8%。这说明：

1. RVS-Movie 小子集确实比 BBB 难；
2. 当前 LLaVA-OneVision-7B dense baseline 本身不能稳定回答 MovieNet open-ended 问题；
3. semantic stream 的正确率更高不应过度解释为方法提升推理能力，更可能是减少冗余视觉上下文后回答更聚焦，或者 judge 对某些短答案更友好。

因此论文中不能直接写“方法提升精度”，更稳妥的说法是：

```text
Semantic Stream preserves QA performance under a stronger LLM-judge proxy while substantially reducing compute and context cost.
```

### 5.3 RVS-Ego 更能体现稳定性

Ego 上 dense correct rate 为 54.2%，明显高于 Movie。Semantic r16/t0.1 为 62.5%，r64/t0.3 为 58.3%。

这说明在第一视角长视频中：

1. r16/t0.1 是更稳的默认配置；
2. r64/t0.3 仍保持接近 dense 的 judged correctness，并带来更高 token reduction；
3. aggressive skip 的风险开始出现，但不是大面积崩坏。

### 5.4 r64/t0.3 的风险更具体了

Ego r64/t0.3 出现 1 个 correctness loss：

```text
Question: What setting is portrayed in the latest clip?
GT: A kitchen setting.
Dense: The latest clip shows a kitchen setting with a focus on the person's hands and the kitchen environment.
Sparse: The latest clip shows a kitchen setting with a Christmas tree in the background.
```

这个例子说明 r64 的风险不是完全漏掉大场景，而是可能引入无依据细节。它支持后续做：

1. latest-frame query 的单独分析；
2. aggressive skip 下的 hallucination 检测；
3. 必要时加入 rolling-anchor correction。

但当前 evidence 还不足以马上把 dual-anchor 作为主方法，因为 r64 总体仍是 2 win / 21 tie / 1 loss。

---

## 6. 与 token-F1 的关系

token-F1 和 LLM judge 给出的方向大体一致：semantic 没有明显低于 dense。但两者的侧重点不同。

| 指标 | 优点 | 缺点 | 当前用途 |
| --- | --- | --- | --- |
| token-F1 | 快、可复现、无模型依赖 | 低估同义改写和长答案 | 粗筛和回归测试 |
| pairwise token-F1 W/T/L | 可看相对 dense 是否退化 | 仍然基于词重叠 | 初步逐题分析 |
| Qwen2.5-VL judge | 更接近语义正确性 | 仍有 judge bias，不是官方指标 | 当前主要 QA proxy |
| official evaluator / stronger judge | 最适合论文 | 尚未接入 | 下一步必须做 |

---

## 7. 当前论文级表述

当前可以写成：

```text
On two real streaming QA subsets, the proposed semantic stream preserves or improves LLM-judge QA correctness compared with the dense visual stream, while reducing visual encoding latency by 3.17x-8.59x and visual token writing by 84.5%-93.8%.
```

但需要加限定：

```text
This is a preliminary local LLM-judge proxy; final claims require official evaluation or stronger judge validation.
```

---

## 8. 下一步

1. 对 RVS-Ego 做 3 次 repeats，并用同一个 Qwen2.5-VL judge 汇总稳定性。
2. 按问题类型统计 judged correctness：
   - scene；
   - object；
   - action；
   - latest-frame；
   - temporal event。
3. 抽取 loss cases，判断 r64 的错误是否集中在 latest-frame / object detail。
4. 接入更强 judge 或官方 evaluator。
5. 若 loss cases 显示单 anchor 漏短时事件，再进入 dual-anchor / rolling-anchor correction。

