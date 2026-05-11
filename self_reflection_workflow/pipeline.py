#!/usr/bin/env python3
"""
pipeline.py — 自反思迭代 RAG 流程（第四章核心）

每轮迭代的控制流：
  1. 子问题生成   — 与第二章完全一致
  2. 初始检索     — 用原始子问题做 BM25 检索
  3. 自反思路由   — 判别模型评估文档质量，三路分支：
       correct   → 文档直接使用，合成子答案
       ambiguous → 约束保持重写 → 再检索 → 合并原始文档 → 合成子答案
       incorrect → 约束保持重写 → 再检索 → 替换原始文档 → 合成子答案
  4. 迭代控制     — 与第二章完全一致（CONTINUE / STOP 决策）
  5. 最终答案生成 — 与第二章完全一致

控制流框架（子问题生成、迭代控制、最终答案）完全照搬第二章，
仅在"检索"与"求解"之间插入自反思路由步骤。
"""

from __future__ import annotations

import json
import hashlib
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

try:
    import numpy as np
except ImportError:
    np = None

# ---------------------------------------------------------------------------
# 路径引导：支持从项目根目录或 chapter4 目录直接运行
# 复用第二章的 src/models、iterative_workflow 模块，避免重复实现
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
CHAPTER2_DIR = ROOT_DIR.parent / "chapter2-rewrite"
SRC_DIR = CHAPTER2_DIR / "src"
for _p in [str(ROOT_DIR), str(CHAPTER2_DIR), str(SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.chat_llm import _load_model_roles, create_general_llm, create_rewrite_llm

# 复用第二章的检索器和 prompt 构建器，控制流保持一致
from iterative_workflow.elasticsearch_retriever import ElasticsearchRetriever
from iterative_workflow.iterative_prompts import ITERATION_CONTROL_PROMPT, ITERATIVE_SUBQUESTION_PROMPT
from iterative_workflow.prompt_templates import (
    build_answer_prompt,
    build_compression_prompt,
    build_final_answer_prompt,
    build_iteration_control_prompt,
    build_subquestion_prompt,
)

from self_reflection_workflow.evaluator_service import EvaluatorService
from self_reflection_workflow.constraint_rewrite_service import ConstraintRewriteService


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelfReflectionConfig:
    """自反思 pipeline 运行时配置。

    检索、LLM 生成、缓存等参数与第二章 PipelineConfig 保持对齐，
    新增 evaluator 相关阈值和模型路径字段。
    """

    corpus_name: str = "hotpotpqa"       # Elasticsearch 索引名
    max_iterations: int = 4              # 每个样本最多迭代轮数
    retrieval_top_k: int = 10            # 每轮最终保留的文档数
    retrieval_buffer_k: int = 50         # 召回缓冲池大小
    retrieval_bm25_top_k: int = 30       # BM25 初始召回条数

    # 判别模型阈值（与 CRAG process_flag 语义一致）
    evaluator_upper_threshold: float = 10.0   # 高于此值 → correct
    evaluator_lower_threshold: float = -10.0  # 低于此值 → incorrect，中间 → ambiguous

    # 重写模型输出预算
    max_tokens_rewrite: int = 64

    # LLM 生成预算
    subquestion_temperature: float = 0.0   # 子问题生成用确定性解码
    control_temperature: float = 0.0       # 迭代控制用确定性解码
    max_tokens_subquestion: int = 96
    max_tokens_control: int = 80
    max_tokens_answer: int = 120

    # prompt 构建模式：builder（推荐）或 raw_template
    subquestion_prompt_mode: str = "builder"
    control_prompt_mode: str = "builder"

    # 兜底原始模板（通常由 builder 生成，此处作为备用）
    subquestion_prompt_template: str = ITERATIVE_SUBQUESTION_PROMPT
    control_prompt_template: str = ITERATION_CONTROL_PROMPT

    # LLM 推理结果缓存（避免重复推理，加速批量实验）
    enable_llm_cache: bool = True
    llm_cache_path: str = "outputs/cache/general_llm_cache.json"

    # 资源配置
    general_on_cpu: bool = False          # True 时 general 模型放 CPU，节省显存
    random_seed: int | None = None        # 全局随机种子，None 表示不固定

    # 判别模型路径（相对于项目根目录或绝对路径）
    evaluator_model_path: str = "../evaluatator_model"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SelfReflectionPipeline:
    """带自反思机制的迭代 RAG pipeline（第四章核心类）。

    在第二章迭代框架基础上，在每轮"检索"之后插入判别模型路由：
      correct   → 保留原始检索文档
      ambiguous → 约束保持重写 → 再检索 → 与原始文档合并
      incorrect → 约束保持重写 → 再检索 → 完全替换原始文档
    """

    def __init__(self, config: SelfReflectionConfig, retriever: ElasticsearchRetriever):
        self.config = config
        self.retriever = retriever

        self._llm_cache: Dict[str, Dict[str, Any]] = {}
        self._llm_cache_file = ROOT_DIR / self.config.llm_cache_path
        if self.config.enable_llm_cache:
            self._load_llm_cache()

        if self.config.random_seed is not None:
            self._set_global_seed(self.config.random_seed)

        # 重写模型：带约束保持 LoRA adapter，专用于查询重写
        self.rewrite_llm = create_rewrite_llm()

        # 判断 rewrite/general 是否为同一底座（无 adapter、无差异），若是则复用实例节省显存
        roles = _load_model_roles()
        rewrite_role = roles.get("rewrite", {}) if isinstance(roles, dict) else {}
        general_role = roles.get("general", {}) if isinstance(roles, dict) else {}
        same_model_roles = (
            str(rewrite_role.get("base_model_path", "")) == str(general_role.get("base_model_path", ""))
            and not rewrite_role.get("adapter_path")
            and not general_role.get("adapter_path")
            and rewrite_role.get("system_prompt") == general_role.get("system_prompt")
        )

        if same_model_roles:
            # rewrite 与 general 是同一底座且无 adapter，复用同一实例避免重复加载
            self.general_llm = self.rewrite_llm
        elif self.config.general_on_cpu:
            # 显存紧张时将 general 模型放 CPU，以 float32 运行
            self.general_llm = create_general_llm(device_map="cpu", torch_dtype=torch.float32)
        else:
            self.general_llm = create_general_llm()

        # 判别模型：轻量级 T5，用于文档质量路由
        evaluator_path = str(
            (ROOT_DIR / self.config.evaluator_model_path).resolve()
        )
        self.evaluator = EvaluatorService(
            model_path=evaluator_path,
            upper_threshold=self.config.evaluator_upper_threshold,
            lower_threshold=self.config.evaluator_lower_threshold,
        )

        # 约束保持重写服务：贪婪解码单次生成，注入 _sample_one 保持缓存行为一致
        self.constraint_rewrite = ConstraintRewriteService(
            rewrite_llm=self.rewrite_llm,
            sample_one=self._sample_one,
            max_tokens_rewrite=self.config.max_tokens_rewrite,
        )

    # ------------------------------------------------------------------ 缓存管理

    def _set_global_seed(self, seed: int) -> None:
        """统一设置随机种子，覆盖 Python / NumPy / Torch，保证实验可复现。"""
        random.seed(seed)
        if np is not None:
            np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_llm_cache(self) -> None:
        path = self._llm_cache_file
        if not path.exists():
            self._llm_cache = {}
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._llm_cache = {}
            return
        entries = data.get("entries", {}) if isinstance(data, dict) else {}
        if isinstance(entries, dict):
            self._llm_cache = {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}
        else:
            self._llm_cache = {}

    def _save_llm_cache(self) -> None:
        path = self._llm_cache_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": self._llm_cache}, ensure_ascii=False), encoding="utf-8")

    def _make_llm_cache_key(self, *, model_role, prompt, temperature, max_tokens, top_p, cache_scope) -> str:
        payload = {
            "model_role": model_role,
            "prompt": prompt,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "cache_scope": cache_scope,
        }
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------ LLM 采样

    def _sample_one(
        self,
        llm,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
        return_raw: bool = False,
        use_cache: bool = False,
        cache_scope: str = "",
    ):
        cache_key = ""
        if use_cache and self.config.enable_llm_cache:
            model_role = "general" if llm is self.general_llm else "rewrite"
            cache_key = self._make_llm_cache_key(
                model_role=model_role, prompt=prompt, temperature=temperature,
                max_tokens=max_tokens, top_p=top_p, cache_scope=cache_scope,
            )
            hit = self._llm_cache.get(cache_key)
            if hit is not None:
                if return_raw:
                    return {"raw_text": hit.get("raw_text", ""), "stripped_text": hit.get("stripped_text", "")}
                return hit.get("stripped_text", "")

        outputs = llm.sample(prompt=prompt, n=1, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        raw_text = outputs[0] if outputs else ""
        stripped = raw_text.strip()

        if use_cache and self.config.enable_llm_cache and cache_key:
            self._llm_cache[cache_key] = {"raw_text": raw_text, "stripped_text": stripped}
            self._save_llm_cache()

        if return_raw:
            return {"raw_text": raw_text, "stripped_text": stripped}
        return stripped

    # ------------------------------------------------------------------ 文本工具（与第二章保持一致，不做修改）

    def _clean_output_text(self, text: str, *, keep_newlines: bool = True) -> str:
        raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not raw:
            return ""
        cleaned = re.sub(r"<think(?:\s+[^>]*)?>[\s\S]*?</think>", " ", raw, flags=re.IGNORECASE)
        fenced = re.findall(r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```", cleaned)
        if fenced:
            cleaned = "\n".join(part.strip() for part in fenced if part.strip())
        lines = []
        for line in cleaned.split("\n"):
            s = line.strip()
            if not s:
                continue
            s = re.sub(r"^(assistant|model|response|output|answer|final answer)\s*[:：\-]\s*", "", s, flags=re.IGNORECASE)
            s = s.strip().strip("`")
            if s:
                lines.append(s)
        if not lines:
            return ""
        out = "\n".join(lines)
        out = out.strip().strip('"\'` ')
        if not keep_newlines:
            out = re.sub(r"\s+", " ", out)
        return out.strip()

    def _extract_json_block(self, text: str):
        raw = text or ""
        if not raw:
            return None
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE):
            b = block.strip()
            if b.startswith("{") and b.endswith("}"):
                return b
        s = raw
        start = s.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(s)):
                ch = s[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start: idx + 1].strip()
                        if candidate:
                            return candidate
                        break
            start = s.find("{", start + 1)
        return None

    def _safe_parse_control(self, text: str) -> Dict[str, str]:
        raw = text or ""
        cleaned = self._clean_output_text(raw, keep_newlines=True)
        parse_candidates: List[str] = []
        if cleaned:
            parse_candidates.append(cleaned)
        extracted = self._extract_json_block(raw)
        if extracted and extracted not in parse_candidates:
            parse_candidates.append(extracted)
        for candidate in parse_candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                action = str(obj.get("action", "CONTINUE")).strip().upper()
                reason = str(obj.get("reason", "")).strip()
                if action != "CONTINUE":
                    action = "CONTINUE"
                return {"action": action, "reason": reason}
        fallback_text = cleaned or raw.strip()
        return {"action": "CONTINUE", "reason": fallback_text}

    def _history_json(self, rounds: List[Dict[str, Any]]) -> str:
        compact = [
            {"iteration": item.get("iteration"), "sub_question": item.get("sub_question"), "sub_answer": item.get("sub_answer", "")}
            for item in rounds
        ]
        return json.dumps(compact, ensure_ascii=False)

    def _format_evidence_for_prompt(self, docs: List[Dict[str, Any]], *, top_k: int = 5, max_chars_per_doc: int = 520, max_total_chars: int = 2400) -> str:
        if not docs:
            return "No retrieved evidence yet."
        lines: List[str] = []
        used = 0
        kept = 0
        for idx, d in enumerate(docs[: max(1, top_k)], start=1):
            title = str(d.get("title", "")).strip() or "Untitled"
            para = str(d.get("paragraph_text", "")).replace("\r", " ").replace("\n", " ").strip()
            if not para:
                continue
            para = re.sub(r"\s+", " ", para)
            snippet = para[:max_chars_per_doc].rstrip()
            if len(para) > max_chars_per_doc:
                snippet += " ..."
            row = f"[{idx}] {title}: {snippet}"
            if used + len(row) > max_total_chars:
                break
            lines.append(row)
            used += len(row)
            kept += 1
        if not lines:
            return "No retrieved evidence yet."
        return f"Evidence snippets ({kept} docs):\n" + "\n".join(lines)

    def _collect_evidence_docs_from_rounds(self, rounds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in rounds:
            for d in item.get("retrieved_docs", []) or []:
                key = str(d.get("paragraph_text", "")).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        return merged

    def _build_qa_history(self, rounds: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        return [
            (item.get("sub_question", ""), item.get("sub_answer", ""))
            for item in rounds
            if item.get("sub_question", "")
        ]

    def _build_iteration_evidence_summary(self, rounds, *, max_chars_per_doc, max_total_chars) -> str:
        evidence_docs = self._collect_evidence_docs_from_rounds(rounds)
        return self._format_evidence_for_prompt(
            evidence_docs, top_k=max(self.config.retrieval_top_k, 5),
            max_chars_per_doc=max_chars_per_doc, max_total_chars=max_total_chars,
        )

    def _normalize_subquestion_candidates(self, cleaned_text: str) -> List[str]:
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        candidates: List[str] = []
        for line in lines:
            item = re.sub(r"^[-*•]+\s*", "", line)
            item = re.sub(r"^\d+[.)]\s*", "", item)
            item = re.sub(r"^(sub[- ]?question|next question|question|follow-up question)\s*[:：\-]\s*", "", item, flags=re.IGNORECASE)
            item = item.strip().strip('"\'` ')
            if item:
                candidates.append(item)
        return candidates

    def _pick_subquestion(self, candidates: List[str]) -> str:
        for item in candidates:
            if "?" in item or "？" in item:
                return re.sub(r"\s+", " ", item).strip('"\'` ')
        if candidates:
            return re.sub(r"\s+", " ", candidates[0]).strip('"\'` ')
        return ""

    # ------------------------------------------------------------------ 检索工具

    def _retrieve_raw(self, query_text: str) -> List[Dict[str, Any]]:
        """用单条查询做 BM25 检索，返回 top-k 文档。"""
        docs = self.retriever.retrieve_paragraphs(
            corpus_name=self.config.corpus_name,
            query_text=query_text,
            max_hits_count=self.config.retrieval_bm25_top_k,
            max_buffer_count=self.config.retrieval_buffer_k,
        )
        return docs[: self.config.retrieval_top_k]

    def _merge_dedup(self, docs_a: List[Dict[str, Any]], docs_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并两个文档列表并按段落文本去重，用于 ambiguous 分支的文档融合。"""
        merged: List[Dict[str, Any]] = []
        seen = set()
        for d in docs_a + docs_b:
            key = d.get("paragraph_text", "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged[: self.config.retrieval_top_k]

    # ------------------------------------------------------------------ 核心流程步骤（子问题生成、子答案合成、迭代控制与第二章一致）

    def _generate_next_subquestion(self, main_query: str, rounds: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        qa_history = self._build_qa_history(rounds)
        if self.config.subquestion_prompt_mode == "builder":
            prompt = build_subquestion_prompt(main_query=main_query, history=qa_history)
        else:
            prompt = self.config.subquestion_prompt_template.format(
                main_query=main_query, history_json=self._history_json(rounds)
            )
        sampled = self._sample_one(
            self.general_llm, prompt=prompt,
            temperature=self.config.subquestion_temperature,
            max_tokens=self.config.max_tokens_subquestion,
            top_p=0.95, return_raw=True,
        )
        assert isinstance(sampled, dict)
        raw_text = sampled["raw_text"]
        cleaned = self._clean_output_text(raw_text, keep_newlines=True)
        candidates = self._normalize_subquestion_candidates(cleaned)
        sub_question = self._pick_subquestion(candidates)
        return sub_question, prompt, raw_text

    def _synthesize_sub_answer(self, sub_question: str, docs: List[Dict[str, Any]]) -> str:
        doc_texts = [d.get("paragraph_text", "") for d in docs if d.get("paragraph_text", "")]
        prompt = build_answer_prompt(sub_query=sub_question, docs=doc_texts)
        return self._sample_one(
            self.general_llm, prompt=prompt, temperature=0.0,
            max_tokens=self.config.max_tokens_answer, top_p=0.9,
            use_cache=True, cache_scope="sub_answer",
        )

    def _control_iteration(self, main_query: str, rounds: List[Dict[str, Any]], iteration: int) -> Tuple[Dict[str, str], str, str]:
        qa_history = self._build_qa_history(rounds)
        evidence_summary = self._build_iteration_evidence_summary(rounds, max_chars_per_doc=560, max_total_chars=2600)
        if self.config.control_prompt_mode == "builder":
            prompt = build_iteration_control_prompt(
                main_query=main_query, history=qa_history,
                evidence_summary=evidence_summary, iteration=iteration,
                max_iterations=self.config.max_iterations,
            )
        else:
            prompt = self.config.control_prompt_template.format(
                main_query=main_query, iteration=iteration,
                max_iterations=self.config.max_iterations,
                history_json=self._history_json(rounds),
                evidence_summary=evidence_summary,
            )
        sampled = self._sample_one(
            self.general_llm, prompt=prompt,
            temperature=self.config.control_temperature,
            max_tokens=self.config.max_tokens_control,
            top_p=0.9, return_raw=True,
        )
        assert isinstance(sampled, dict)
        raw_text = sampled["raw_text"]
        parsed = self._safe_parse_control(raw_text)
        return parsed, prompt, raw_text

    # ------------------------------------------------------------------ 自反思路由（第四章新增核心逻辑）

    def _apply_self_reflection(
        self,
        sub_question: str,
        initial_docs: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """自反思路由：根据判别模型评分决定如何处理初始检索文档。

        三路路由策略：
          correct   → 文档质量足够，直接返回，不触发重写
          ambiguous → 文档部分相关，重写查询后再检索，与原始文档合并
          incorrect → 文档不相关，重写查询后再检索，完全替换原始文档

        返回：
            final_docs:       经路由处理后的最终文档列表
            reflection_trace: 路由决策和重写详情，写入轮次记录供调试追踪
        """
        agg_label, per_doc_results = self.evaluator.evaluate_docs(sub_question, initial_docs)

        trace: Dict[str, Any] = {
            "routing_label": agg_label,
            "per_doc_labels": [
                {"label": r[0], "score": r[1]} for r in per_doc_results
            ],
            "rewrite": None,
            "rewrite_docs": None,
        }

        if agg_label == EvaluatorService.LABEL_CORRECT:
            # 文档质量足够，直接使用，不触发重写
            return initial_docs, trace

        # ambiguous 和 incorrect 均触发约束保持重写
        rewrite_result = self.constraint_rewrite.rewrite(sub_question)
        rewritten_query = rewrite_result["normalized_rewrite"]
        trace["rewrite"] = rewrite_result

        rewrite_docs = self._retrieve_raw(rewritten_query)
        trace["rewrite_docs"] = rewrite_docs

        if agg_label == EvaluatorService.LABEL_AMBIGUOUS:
            # ambiguous：原始文档仍有部分价值，与重写检索结果合并
            final_docs = self._merge_dedup(initial_docs, rewrite_docs)
        else:
            # incorrect：原始文档完全不可信，用重写检索结果替换
            final_docs = rewrite_docs[: self.config.retrieval_top_k]

        return final_docs, trace

    # ------------------------------------------------------------------ 对外主入口

    def run(self, main_query: str) -> Dict[str, Any]:
        """执行完整自反思迭代流程，返回包含所有中间状态的结构化轨迹。"""
        rounds: List[Dict[str, Any]] = []

        for i in range(1, self.config.max_iterations + 1):
            sub_question, subquestion_prompt, subquestion_raw_output = self._generate_next_subquestion(main_query, rounds)
            sub_question = sub_question.strip()

            # 步骤 1：用原始子问题做初始 BM25 检索
            initial_docs = self._retrieve_raw(sub_question)

            # 步骤 2：自反思路由（判别模型评估 → 按需重写再检索）
            final_docs, reflection_trace = self._apply_self_reflection(sub_question, initial_docs)

            # 步骤 3：基于最终文档合成子答案
            sub_answer = self._synthesize_sub_answer(sub_question, final_docs)

            round_item: Dict[str, Any] = {
                "iteration": i,
                "sub_question": sub_question,
                "subquestion_prompt": subquestion_prompt,
                "subquestion_raw_output": subquestion_raw_output,
                "initial_docs": initial_docs,
                "reflection": reflection_trace,
                "retrieved_docs": final_docs,
                "sub_answer": sub_answer,
            }
            rounds.append(round_item)

            # 步骤 4：迭代控制决策（与第二章一致）
            control, control_prompt, control_raw_output = self._control_iteration(main_query, rounds, i)
            round_item["control_prompt"] = control_prompt
            round_item["control_raw_output"] = control_raw_output
            round_item["control"] = control

        return {
            "main_query": main_query,
            "config": {
                "corpus_name": self.config.corpus_name,
                "max_iterations": self.config.max_iterations,
                "retrieval_top_k": self.config.retrieval_top_k,
                "evaluator_upper_threshold": self.config.evaluator_upper_threshold,
                "evaluator_lower_threshold": self.config.evaluator_lower_threshold,
            },
            "rounds": rounds,
        }

    def _build_final_answer(self, main_query: str, rounds: List[Dict[str, Any]], documents: List[Dict[str, Any]]) -> str:
        qa_history = [
            (item.get("sub_question", ""), item.get("sub_answer", ""))
            for item in rounds if item.get("sub_question", "")
        ]
        doc_texts = [d.get("paragraph_text", "") for d in documents if d.get("paragraph_text", "")]
        prompt = build_final_answer_prompt(query=main_query, qa_history=qa_history, documents=doc_texts)
        return self._sample_one(
            self.general_llm, prompt=prompt, temperature=0.0, max_tokens=256, top_p=0.9,
            use_cache=True, cache_scope="final_answer",
        )

    def _compress_final_answer(self, main_query: str, answer: str) -> str:
        prompt = build_compression_prompt(question=main_query, predict=answer)
        return self._sample_one(
            self.general_llm, prompt=prompt, temperature=0.0, max_tokens=128, top_p=0.9,
            use_cache=True, cache_scope="compress_answer",
        )

    def run_interface(self, main_query: str, trace: Dict[str, Any] | None = None) -> Tuple[str, int, float, float]:
        """对齐第二章 run_interface 接口签名，返回 (压缩预测, 检索轮次, precision, recall)。

        precision/recall 需要黄金文档才可计算，此处统一返回 0.0，
        由批量评测脚本在外部用 EM/F1 指标替代。
        """
        if trace is None:
            trace = self.run(main_query)
        rounds = trace.get("rounds", [])
        retrieved_times = len(rounds)

        merged_docs: List[Dict[str, Any]] = []
        seen = set()
        for item in rounds:
            for d in item.get("retrieved_docs", []):
                key = d.get("paragraph_text", "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged_docs.append(d)

        final_answer = self._build_final_answer(main_query, rounds, merged_docs)
        compressed_predict = self._compress_final_answer(main_query, final_answer)
        return compressed_predict, retrieved_times, 0.0, 0.0
