import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an impartial evaluator for open-ended video question answering.
Judge whether each answer captures the essential meaning of the reference answer.
Ignore wording differences. Penalize missing key facts, wrong objects, wrong actions, or conflicting details.
Return JSON only."""


USER_TEMPLATE = """Question:
{question}

Reference answer:
{answer}

Dense baseline answer:
{baseline_pred}

Sparse semantic stream answer:
{method_pred}

Evaluate both answers against the reference answer. Then compare the sparse answer to the dense answer.
Use this JSON schema exactly:
{{
  "dense_correct": true or false,
  "sparse_correct": true or false,
  "relative": "better" or "same" or "worse",
  "reason": "short reason"
}}"""


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_model(model_path: str, device: str, model_family: str):
    if model_family == "auto":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_family = "qwen2_5_vl" if config.model_type == "qwen2_5_vl" else "causal"
    if model_family == "qwen2_5_vl":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {"trust_remote_code": True}
        if device == "cuda":
            kwargs.update({"device_map": "auto", "torch_dtype": torch.bfloat16})
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if device != "cuda":
            model = model.to(device)
        model.eval()
        return {"family": model_family, "processor": processor, "model": model, "device": device}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs.update({"device_map": "auto", "torch_dtype": torch.bfloat16})
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return {"family": model_family, "tokenizer": tokenizer, "model": model, "device": device}


def build_prompt(row):
    return USER_TEMPLATE.format(
        question=row.get("question", ""),
        answer=row.get("answer", ""),
        baseline_pred=row.get("baseline_pred", ""),
        method_pred=row.get("method_pred", ""),
    )


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def causal_generate(runtime, row, max_new_tokens):
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(row)},
    ]
    text = apply_chat_template(tokenizer, messages)
    inputs = tokenizer([text], return_tensors="pt").to(model.device if device == "cuda" else device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def qwen2_5_vl_generate(runtime, row, max_new_tokens):
    processor = runtime["processor"]
    model = runtime["model"]
    prompt = build_prompt(row)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = output_ids[0][inputs.input_ids.shape[-1] :]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_judgment(runtime, row, max_new_tokens):
    if runtime["family"] == "qwen2_5_vl":
        return qwen2_5_vl_generate(runtime, row, max_new_tokens)
    return causal_generate(runtime, row, max_new_tokens)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def parse_judgment(text):
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        dense_correct = parse_bool(parsed.get("dense_correct"))
        sparse_correct = parse_bool(parsed.get("sparse_correct"))
        relative = str(parsed.get("relative", "")).lower().strip()
        if relative not in {"better", "same", "worse"}:
            relative = ""
        return {
            "dense_correct": dense_correct,
            "sparse_correct": sparse_correct,
            "relative": relative,
            "reason": str(parsed.get("reason", "")),
            "parse_ok": dense_correct is not None and sparse_correct is not None and bool(relative),
        }
    return {
        "dense_correct": None,
        "sparse_correct": None,
        "relative": "",
        "reason": "",
        "parse_ok": False,
    }


def summarize(rows):
    judged = [row for row in rows if row["parse_ok"] == "1"]
    total = len(rows)
    valid = len(judged)

    def rate(predicate):
        return sum(1 for row in judged if predicate(row)) / valid if valid else 0.0

    return {
        "samples": total,
        "valid_judgments": valid,
        "parse_success_rate": valid / total if total else 0.0,
        "dense_correct_rate": rate(lambda row: row["dense_correct"] == "1"),
        "sparse_correct_rate": rate(lambda row: row["sparse_correct"] == "1"),
        "relative_better_rate": rate(lambda row: row["relative"] == "better"),
        "relative_same_rate": rate(lambda row: row["relative"] == "same"),
        "relative_worse_rate": rate(lambda row: row["relative"] == "worse"),
        "better": sum(1 for row in judged if row["relative"] == "better"),
        "same": sum(1 for row in judged if row["relative"] == "same"),
        "worse": sum(1 for row in judged if row["relative"] == "worse"),
        "dense_only_correct": sum(
            1 for row in judged if row["dense_correct"] == "1" and row["sparse_correct"] == "0"
        ),
        "sparse_only_correct": sum(
            1 for row in judged if row["dense_correct"] == "0" and row["sparse_correct"] == "1"
        ),
        "both_correct": sum(
            1 for row in judged if row["dense_correct"] == "1" and row["sparse_correct"] == "1"
        ),
        "both_wrong": sum(
            1 for row in judged if row["dense_correct"] == "0" and row["sparse_correct"] == "0"
        ),
    }


def judge_file(args):
    rows = read_csv(Path(args.input_csv))
    if args.max_samples:
        rows = rows[: args.max_samples]
    runtime = load_model(args.judge_model, args.device, args.model_family)
    judged_rows = []
    for idx, row in enumerate(rows):
        raw = generate_judgment(runtime, row, args.max_new_tokens)
        parsed = parse_judgment(raw)
        judged = {
            **row,
            "judge_raw": raw,
            "judge_reason": parsed["reason"],
            "dense_correct": "1" if parsed["dense_correct"] else "0" if parsed["dense_correct"] is False else "",
            "sparse_correct": "1" if parsed["sparse_correct"] else "0" if parsed["sparse_correct"] is False else "",
            "relative": parsed["relative"],
            "parse_ok": "1" if parsed["parse_ok"] else "0",
        }
        judged_rows.append(judged)
        print(
            f"{idx + 1}/{len(rows)} parse={judged['parse_ok']} "
            f"dense={judged['dense_correct']} sparse={judged['sparse_correct']} "
            f"relative={judged['relative']}"
        )
    summary = summarize(judged_rows)
    summary.update(
        {
            "input_csv": args.input_csv,
            "judge_model": args.judge_model,
            "device": args.device,
        }
    )
    write_csv(Path(args.output_csv), judged_rows)
    write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run an LLM-as-judge evaluator for pairwise QA outputs.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--judge-model", default="/home/mllm/models/Qwen3-0.6B")
    parser.add_argument("--model-family", choices=["auto", "causal", "qwen2_5_vl"], default="auto")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main():
    judge_file(parse_args())


if __name__ == "__main__":
    main()
