"""
evaluator_service.py — 轻量级判别模型服务（第四章自反思模块）

封装 CRAG 原版 T5 序列分类模型，对"问题-段落"对打分，
将检索文档路由到三个质量等级：
  - correct   (分数 >= upper_threshold)：文档与问题高度相关，直接使用
  - ambiguous (lower_threshold <= 分数 < upper_threshold)：部分相关，触发重写后合并
  - incorrect (分数 < lower_threshold)：不相关，触发重写后替换

输入格式与 CRAG 保持一致：  "question [SEP] passage"
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import List, Tuple

from transformers import T5Tokenizer, T5ForSequenceClassification


class EvaluatorService:
    """基于 T5 的轻量级文档相关性判别器，用于自反思路由决策。"""

    # 三个路由标签常量，与 CRAG process_flag 语义对应
    LABEL_CORRECT = "correct"      # 文档相关，直接使用
    LABEL_AMBIGUOUS = "ambiguous"  # 文档部分相关，合并重写结果
    LABEL_INCORRECT = "incorrect"  # 文档不相关，替换为重写检索结果

    def __init__(
        self,
        model_path: str,
        upper_threshold: float = 10.0,   # 高于此分数判定为 correct
        lower_threshold: float = -10.0,  # 低于此分数判定为 incorrect，中间为 ambiguous
        device: str | None = None,
        max_length: int = 512,           # T5 输入最大 token 数
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_length = max_length

        # 自动选择设备：优先 GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 加载 T5 tokenizer 和分类头（num_labels=1 输出单个 logit 分数）
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForSequenceClassification.from_pretrained(
            model_path, num_labels=1
        )
        self.model.to(self.device)
        self.model.eval()

    def _score_one(self, query: str, passage: str) -> float:
        """对单个 (问题, 段落) 对打分，返回原始 logit 值。

        空段落直接返回低于下阈值的分数，避免无效推理。
        """
        if not passage or not passage.strip():
            # 空段落直接判定为不相关，跳过模型推理
            return float(self.lower_threshold) - 1.0

        # 拼接格式与 CRAG 训练时一致：query [SEP] passage
        input_text = query + " [SEP] " + passage
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            outputs = self.model(
                enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
        return float(outputs["logits"].cpu())

    def score_passages(self, query: str, passages: List[str]) -> List[float]:
        """批量对段落列表打分，返回原始 logit 分数列表。"""
        return [self._score_one(query, p) for p in passages]

    def route_passage(self, query: str, passage: str) -> Tuple[str, float]:
        """对单个段落做路由判断，返回 (标签, 分数)。"""
        score = self._score_one(query, passage)
        label = self._score_to_label(score)
        return label, score

    def route_passages(
        self, query: str, passages: List[str]
    ) -> List[Tuple[str, float]]:
        """对段落列表逐一路由，返回 [(标签, 分数), ...] 列表。"""
        return [self.route_passage(query, p) for p in passages]

    def aggregate_label(self, labels: List[str]) -> str:
        """将多个段落标签聚合为文档集合级别的单一路由标签。

        优先级：correct > ambiguous > incorrect
        只要有一个段落被判定为 correct，整体即视为 correct。
        与 CRAG process_flag 的聚合逻辑保持一致。
        """
        if self.LABEL_CORRECT in labels:
            return self.LABEL_CORRECT
        if self.LABEL_AMBIGUOUS in labels:
            return self.LABEL_AMBIGUOUS
        return self.LABEL_INCORRECT

    def evaluate_docs(
        self, query: str, docs: List[dict]
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """对检索文档列表做整体评估，返回路由决策和逐文档详情。

        参数：
            query: 当前子问题
            docs:  检索返回的文档列表，每个 dict 含 'paragraph_text' 字段

        返回：
            aggregate_label: 文档集合的整体路由标签
            per_doc_results: 每个文档的 (标签, 分数) 列表
        """
        passages = [str(d.get("paragraph_text", "")) for d in docs]
        per_doc = self.route_passages(query, passages)
        labels = [r[0] for r in per_doc]
        agg = self.aggregate_label(labels)
        return agg, per_doc

    def _score_to_label(self, score: float) -> str:
        """将原始 logit 分数映射到三分类标签。"""
        if score >= self.upper_threshold:
            return self.LABEL_CORRECT
        if score >= self.lower_threshold:
            return self.LABEL_AMBIGUOUS
        return self.LABEL_INCORRECT
