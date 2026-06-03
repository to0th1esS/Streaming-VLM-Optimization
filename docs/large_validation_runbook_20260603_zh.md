# 大规模验证运行说明（2026-06-03）

## 1. 当前目的

本轮大规模验证用于回答两个问题：

1. `Semantic + always_recent` 是否能作为 query-decoupled、即插即用的主策略。
2. 在同样 fixed recent access 下，semantic admission 是否优于 periodic/uniform admission baseline。

## 2. 一键脚本

脚本：

```bash
scripts/run_query_decoupled_large_validation.sh
```

默认会运行：

```text
RVS-Ego:
  semantic r64/t0.3 + recency K=4 + always_recent qrb4, repeat3
  periodic r13/t999 + recency K=4 + always_recent qrb4, repeat3

RVS-Movie:
  semantic r64/t0.3 + recency K=4 + always_recent qrb4, repeat3
  periodic r13/t999 + recency K=4 + always_recent qrb4, repeat3
```

并自动生成：

```text
overlap metrics
dense-vs-method comparison
summary_all.csv
summary_all.json
```

默认输出目录：

```text
results/large_validation_query_decoupled_20260603
```

## 3. 推荐运行命令

不跑 LLM judge：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/root/miniconda3/bin/python \
bash scripts/run_query_decoupled_large_validation.sh
```

跑 rep0 的 Qwen2.5-VL judge：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/root/miniconda3/bin/python \
RUN_JUDGE=true \
JUDGE_MODEL=/home/mllm/models/Qwen2.5-VL-7B-Instruct \
bash scripts/run_query_decoupled_large_validation.sh
```

## 4. 当前阻塞

2026-06-03 本地连接 `remote-docker` 时，SSH host key 已变化，旧 key 已移除，但新的远程容器拒绝了本机公钥：

```text
root@127.0.0.1: Permission denied (publickey,password,keyboard-interactive).
```

这通常表示 Docker 容器重建后 `/root/.ssh/authorized_keys` 丢失或未包含本机公钥。

需要在服务器容器内恢复：

```bash
mkdir -p /root/.ssh
echo "<contents of C:\\Users\\Administrator\\.ssh\\id_rsa.pub>" >> /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
```

恢复后，本地应能通过：

```powershell
ssh remote-docker "echo ok"
```

随后即可继续执行一键脚本。

## 5. 后续记录

脚本跑完后，需要将以下结果写入新的中文实验记录：

```text
results/large_validation_query_decoupled_20260603/summary_all.csv
results/large_validation_query_decoupled_20260603/summary_all.json
```

重点分析：

1. RVS-Ego repeat3 是否复现 query-decoupled 结论。
2. RVS-Movie 是否跨数据集保持优势。
3. Semantic admission 相比 periodic baseline 是否在同等或更少 token budget 下更好。
4. 是否需要把 fixed recent access 作为主方法默认，把 query-aware access 作为可选增强。
