# 当前 QA 精度指标说明与补充评估

日期：2026-06-02

本文档回答一个关键问题：目前实验里“精度”到底怎么体现？为什么之前看起来没有明确 accuracy？

结论先说清楚：

> 当前 RVS-Movie / RVS-Ego 是 open-ended QA，不是分类任务，因此没有天然的 top-1 accuracy。此前使用的 `token-F1` 和 `rule proxy` 只是自动化弱指标，只能证明方法没有明显崩坏，不能作为最终论文精度指标。为了更清楚地表达精度，本轮新增了 dense vs semantic 的逐问题 pairwise comparison，报告 win / tie / loss。

---

## 1. 为什么之前看不到清晰 accuracy

当前 RVS 的答案形式是开放式文本，例如：

```text
Question: What setting is portrayed in the latest clip?
GT: A kitchen setting.
Prediction A: The latest clip shows a kitchen setting with a focus on the person's hands.
Prediction B: The latest clip shows a kitchen setting with a Christmas tree in the background.
```

这种任务不像分类任务有固定标签，不能直接算 “预测类别是否等于 GT”。因此我们之前用了两个 proxy：

1. `token-overlap F1`：预测答案和 GT 答案的词重叠程度。
2. `rule proxy`：对部分可规则判断的问题，检查预测中是否包含关键词。

它们的问题是：

1. token-F1 会低估同义改写；
2. token-F1 会惩罚更长但语义正确的回答；
3. rule proxy 只适合部分结构化问题；
4. 两者都不能替代最终论文中的 LLM judge / official evaluator。

因此，之前的“精度”主要体现为：

```text
semantic 的 QA proxy 没有明显低于 dense，同时速度和 token reduction 大幅提升。
```

这个证据有价值，但还不够强。

---

## 2. 本轮新增的 pairwise 精度评估

新增脚本：

```text
scripts/compare_qa_predictions.py
```

它把 dense 和 semantic 的预测逐问题对齐，计算：

1. dense token-F1；
2. semantic token-F1；
3. semantic - dense 的 delta；
4. win / tie / loss；
5. F1@0.3 / F1@0.5 / F1@0.7 阈值通过率；
6. 同时记录 speedup 和 token reduction。

判定规则：

```text
if semantic_f1 - dense_f1 > 0.05: win
if dense_f1 - semantic_f1 > 0.05: loss
otherwise: tie
```

这个指标的意义是：

> 不直接问 semantic 是否绝对正确，而是问在同一个模型、同一个问题上，semantic stream 相比 dense visual stream 是否变差。

这比单独的 mean token-F1 更适合当前阶段，因为我们的目标是：

```text
在 QA 性能基本不下降的约束下，最大化视觉计算和 token/cache 写入效率。
```

---

## 3. RVS-Movie pairwise 结果

使用 repeat 0 做逐题对比，Movie 三次重复的预测基本确定，因此 repeat 0 足以看具体问题。

| comparison | mean F1 dense | mean F1 semantic | delta | win | tie | loss | speedup | token reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movie dense vs r16/t0.1 | 0.0537 | 0.0694 | +0.0157 | 4 | 18 | 2 | 5.78x | 85.3% |
| Movie dense vs r64/t0.3 | 0.0537 | 0.0504 | -0.0033 | 5 | 17 | 2 | 8.94x | 93.1% |

Movie 上的现象：

1. r16/t0.1：24 题中 18 题持平，4 题更好，2 题更差。
2. r64/t0.3：24 题中 17 题持平，5 题更好，2 题更差。
3. r64 的 mean F1 略低于 dense，但 win 数反而更多，说明少数 loss 的幅度较大。
4. Movie 的 token-F1 整体很低，说明该数据集开放式描述较难，必须引入 LLM judge。

Movie r16 的 loss 例子：

```text
Q: What escalates the tension between the characters?
GT: The revelation of the man's true identity.
Dense: The man's aggressive behavior escalates the tension.
Semantic r16: The man's actions escalate the tension as he points a gun at the woman.
```

这个例子里 semantic 并非完全错误，但 token-F1 更低，因为没有命中 “true identity”。这说明 token-F1 不能完全代表语义正确性。

Movie r64 的明显 loss 例子：

```text
Q: What escalates the tension between the characters?
GT: The revelation of the man's true identity.
Dense: The man's aggressive behavior escalates the tension.
Semantic r64: The skeleton image with red circles around the abdomen escalates the tension.
```

这个例子更像 aggressive skip 造成的视觉事件错配，值得后续做 failure analysis。

---

## 4. RVS-Ego pairwise 结果

| comparison | mean F1 dense | mean F1 semantic | delta | win | tie | loss | speedup | token reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ego dense vs r16/t0.1 | 0.2273 | 0.2529 | +0.0256 | 3 | 21 | 0 | 3.17x | 84.5% |
| Ego dense vs r64/t0.3 | 0.2273 | 0.2489 | +0.0216 | 4 | 19 | 1 | 3.69x | 93.8% |

Ego 上的现象：

1. r16/t0.1：24 题中 21 题持平，3 题更好，0 题更差。
2. r64/t0.3：24 题中 19 题持平，4 题更好，1 题更差。
3. 第一视角视频没有马上暴露 single-anchor 崩坏。
4. r64 的唯一 loss 是电子设备识别错误。

Ego r64 的 loss 例子：

```text
Q: Is there an electronic device being used in this scene?
GT: Yes, a mobile phone.
Dense: Yes.
Semantic r64: Yes, there is a laptop on the table.
```

这个 loss 很有研究价值：它不是完全漏答，而是细粒度物体类别从 mobile phone 错到 laptop。说明 aggressive semantic skip 对细粒度物体状态有风险。

Ego r16 的 win 例子：

```text
Q: What kitchen activity is underway in the clip?
GT: Cutting vegetables.
Dense: The kitchen activity underway in the clip is the preparation of ingredients for cooking.
Semantic r16: The clip shows a person preparing food, including chopping vegetables, cooking a dish, and placing food on a plate.
```

semantic 更明确命中 chopping vegetables。这类现象说明减少冗余视觉 token 有时可能让回答更聚焦，但目前不能过度声称提升精度。

---

## 5. 当前可以怎样表述“精度”

在当前阶段，最严谨的表述是：

```text
Semantic Stream achieves large speedup and token reduction while maintaining QA proxy performance comparable to the dense baseline.
```

更具体地说：

1. Movie r16/t0.1：`5.78x` speedup，`85.3%` token reduction，`4/18/2` win/tie/loss。
2. Movie r64/t0.3：`8.94x` speedup，`93.1%` token reduction，`5/17/2` win/tie/loss。
3. Ego r16/t0.1：`3.17x` speedup，`84.5%` token reduction，`3/21/0` win/tie/loss。
4. Ego r64/t0.3：`3.69x` speedup，`93.8%` token reduction，`4/19/1` win/tie/loss。

这里的“精度不下降”不是最终结论，而是当前自动 proxy 下的初步证据。

---

## 6. 对后续论文实验的要求

当前指标还不够顶会论文使用。下一步必须补强为更可信的 QA 指标：

1. **LLM judge**  
   输入 question、GT answer、dense prediction、semantic prediction，让 judge 判断：
   - dense 是否正确；
   - semantic 是否正确；
   - semantic 相对 dense 是 better / same / worse；
   - 是否存在 hallucination；
   - 是否遗漏关键事件。

2. **official evaluator**  
   如果 RVS / StreamingBench / OVO-Bench 有官方评价方式，优先接入。

3. **per-category accuracy**  
   按问题类型拆分：
   - scene summary；
   - object recognition；
   - action recognition；
   - temporal event；
   - latest-frame query；
   - long-range context query。

4. **failure-driven dual-anchor**  
   只有当 LLM judge 或 per-category 分析显示：
   - aggressive skip 漏掉短时事件；
   - mobile phone / laptop 这类细粒度物体混淆；
   - latest-frame query 明显变差；
   再引入 dual-anchor / rolling-anchor correction。

---

## 7. 当前方向判断

这轮补充评估回答了“精度怎么体现”的问题：

> 目前精度体现为 open-ended QA proxy 下相对 dense 的逐题 win/tie/loss，而不是最终 accuracy。

从结果看，Semantic Stream 在 Movie 和 Ego 上都没有明显大面积输给 dense，且伴随 3x-9x 视觉编码加速和 84%-94% token reduction。

因此下一步不是回到 feature cosine，而是继续推进：

```text
LLM judge / official evaluator -> per-category failure analysis -> targeted correction.
```

