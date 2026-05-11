#!/usr/bin/env python3
"""
run_self_reflection_batch.py — 第四章自反思 pipeline 批量评测入口

功能：
  - 从 JSON 配置文件读取运行参数（与第二章 run_iterative_test_batch.py 结构一致）
  - 支持断点续跑：已完成的样本索引写入 JSONL，重启后自动跳过
  - 每条样本输出 EM / F1 指标，最终汇总写入 summary JSON

用法：
  python run_self_reflection_batch.py --config experiments/self_reflection_config.json
  python run_self_reflection_batch.py --print-config   # 打印合并后的运行时配置
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CHAPTER2_DIR = ROOT_DIR.parent / "chapter2-rewrite"
SRC_DIR = CHAPTER2_DIR / "src"
for _p in [str(ROOT_DIR), str(CHAPTER2_DIR), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CONFIG = ROOT_DIR / "experiments" / "self_reflection_config.json"


def _load_completed_indices(jsonl_path: Path) -> set:
    """从已有 JSONL 输出文件中读取已完成的样本索引，用于断点续跑。"""
    if not jsonl_path.exists():
        return set()
    completed: set = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            idx = row.get("index")
            if isinstance(idx, int):
                completed.add(idx)
    return completed


def _load_config(config_path: Path) -> dict:
    """加载 JSON 配置文件，返回原始 dict。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a dict: {config_path}")
    return data


def _strip_comments(obj: dict) -> dict:
    """去掉配置 dict 中以 _comment_ 开头的注释键，保留有效配置项。"""
    return {k: v for k, v in obj.items() if not str(k).startswith("_comment_")}


def _gold_answers(sample: dict) -> list:
    """从样本中提取去重后的标准答案列表，兼容 answers（列表）和 answer（单值）两种字段格式。"""
    vals: list = []
    answers = sample.get("answers")
    if isinstance(answers, list):
        for x in answers:
            s = str(x).strip()
            if s:
                vals.append(s)
    single = str(sample.get("answer", "")).strip()
    if single:
        vals.append(single)
    deduped: list = []
    seen = set()
    for x in vals:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(x)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch self-reflection pipeline evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    conf = _load_config(args.config)
    # common 为通用配置，batch_test 为批量测试专属配置，后者覆盖前者
    common = _strip_comments(conf.get("common", {}))
    batch = _strip_comments(conf.get("batch_test", {}))
    runtime = dict(common)
    runtime.update(batch)

    if args.print_config:
        print(json.dumps(runtime, ensure_ascii=False, indent=2))
        return

    from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
    from self_reflection_workflow.pipeline import SelfReflectionPipeline, SelfReflectionConfig

    # 延迟导入：复用第二章的 EM/F1 评测器，避免启动时触发不必要的初始化
    sys.path.insert(0, str(CHAPTER2_DIR))
    from evaluate_em_f1 import EMF1Evaluator

    dataset_path = Path(str(runtime.get("dataset", "src/Dataset/hotpot_train_test_sample_200.json")))
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array: {dataset_path}")

    total_dataset_size = len(data)
    max_samples = runtime.get("max_samples", None)
    if max_samples is not None:
        data = data[: int(max_samples)]

    dataset_name = dataset_path.stem
    test_size = len(data)

    output_jsonl_cfg = runtime.get("output_jsonl", None)
    summary_json_cfg = runtime.get("summary_json", None)
    output_jsonl = Path(str(output_jsonl_cfg)) if output_jsonl_cfg else (
        ROOT_DIR / "outputs" / "self_reflection" / f"{dataset_name}-{test_size}.jsonl"
    )
    summary_json = Path(str(summary_json_cfg)) if summary_json_cfg else (
        ROOT_DIR / "outputs" / "self_reflection" / f"{dataset_name}-{test_size}-summary.json"
    )

    config = SelfReflectionConfig(
        corpus_name=str(runtime.get("corpus_name", "hotpotpqa")),
        max_iterations=int(runtime.get("max_iterations", 4)),
        retrieval_top_k=int(runtime.get("top_k", 10)),
        retrieval_buffer_k=int(runtime.get("retrieval_buffer_k", 50)),
        retrieval_bm25_top_k=int(runtime.get("bm25_top_k", 30)),
        evaluator_upper_threshold=float(runtime.get("evaluator_upper_threshold", 10.0)),
        evaluator_lower_threshold=float(runtime.get("evaluator_lower_threshold", -10.0)),
        max_tokens_rewrite=int(runtime.get("max_tokens_rewrite", 64)),
        general_on_cpu=bool(runtime.get("general_on_cpu", False)),
        subquestion_prompt_mode=str(runtime.get("subquestion_prompt_mode", "builder")),
        control_prompt_mode=str(runtime.get("control_prompt_mode", "builder")),
        random_seed=runtime.get("seed", None),
        enable_llm_cache=bool(runtime.get("enable_llm_cache", True)),
        llm_cache_path=str(runtime.get("llm_cache_path", "outputs/cache/general_llm_cache.json")),
        evaluator_model_path=str(runtime.get("evaluator_model_path", "../evaluatator_model")),
    )

    retriever = ElasticsearchRetriever(
        host=str(runtime.get("es_host", "localhost")),
        port=int(runtime.get("es_port", 9200)),
        username=str(runtime.get("es_user", "elastic")),
        password=str(runtime.get("es_password", "246247")),
    )
    pipeline = SelfReflectionPipeline(config=config, retriever=retriever)
    evaluator_em_f1 = EMF1Evaluator()

    completed_indices = _load_completed_indices(output_jsonl)
    if completed_indices:
        print(f"Resume mode: {len(completed_indices)} samples already done in {output_jsonl}")

    # 已有输出文件时追加写入，支持断点续跑
    writer_mode = "a" if output_jsonl.exists() else "w"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with output_jsonl.open(writer_mode, encoding="utf-8") as out_f:
        for i, sample in enumerate(data):
            if i in completed_indices:
                continue

            question = str(sample.get("question", "")).strip()
            gold_answers = _gold_answers(sample)
            sample_id = str(sample.get("_id", i))

            prediction, retrieved_times, precision, recall = pipeline.run_interface(main_query=question)

            if gold_answers:
                em = max(float(evaluator_em_f1.exact_match_score(prediction, gt)) for gt in gold_answers)
                f1 = max(float(evaluator_em_f1.f1_score(prediction, gt)) for gt in gold_answers)
            else:
                em, f1 = 0.0, 0.0

            record = {
                "index": i,
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "dataset_total_size": total_dataset_size,
                "dataset_eval_size": test_size,
                "question": question,
                "reference_answers": gold_answers,
                "reference_answer": gold_answers[0] if gold_answers else "",
                "final_prediction": prediction,
                "accuracy": em,
                "em": em,
                "f1": f1,
                "retrieved_times": retrieved_times,
                "precision": precision,
                "recall": recall,
                "experiment_report": {
                    "config_path": str(args.config),
                    "runtime_config": runtime,
                    "pipeline_config": asdict(config),
                    "dataset": {
                        "path": str(dataset_path),
                        "name": dataset_name,
                        "total_size": total_dataset_size,
                        "eval_size": test_size,
                    },
                },
            }
            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{test_size}")

    # 从完整输出文件重新聚合指标（支持续跑后统计全量结果）
    all_records = []
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    all_records.append(row)

    n = len(all_records)
    avg_em = sum(float(r.get("em", 0.0)) for r in all_records) / n if n else 0.0
    avg_f1 = sum(float(r.get("f1", 0.0)) for r in all_records) / n if n else 0.0
    avg_retrieved = sum(float(r.get("retrieved_times", 0.0)) for r in all_records) / n if n else 0.0

    summary = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "dataset_total_size": total_dataset_size,
        "dataset_eval_size": test_size,
        "count": n,
        "avg_em": avg_em,
        "avg_accuracy": avg_em,
        "avg_f1": avg_f1,
        "avg_retrieved_times": avg_retrieved,
        "output_jsonl": str(output_jsonl),
        "runtime_config": runtime,
        "pipeline_config": asdict(config),
        "config": {
            "evaluator_upper_threshold": config.evaluator_upper_threshold,
            "evaluator_lower_threshold": config.evaluator_lower_threshold,
            "max_iterations": config.max_iterations,
            "retrieval_top_k": config.retrieval_top_k,
            "corpus_name": config.corpus_name,
        },
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
