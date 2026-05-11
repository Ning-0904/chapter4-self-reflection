"""
constraint_rewrite_service.py — 约束保持查询重写服务（第四章自反思模块）

在自反思 pipeline 中，当判别模型判定检索文档质量为 ambiguous 或 incorrect 时，
调用本服务对子问题进行一次约束保持重写，再用重写后的查询重新检索。

设计约束（与第二章多模板重写的区别）：
  - 仅使用 RAFE-SFT 约束保持重写 prompt（不使用 HyDE / keyword_rewrite）
  - 贪婪解码（temperature=0.0），单次生成，不做多组采样
  - 归一化逻辑与第二章 RewriteService._normalize_rewrite_text 完全一致
"""

from __future__ import annotations

import re
from typing import Any, Callable


# RAFE-SFT 约束保持重写 prompt
# 要求：保留原始意图、消除歧义、提升检索命中率、不引入新信息
RAFE_SFT_REWRITE_PROMPT = """You are an expert in query optimizer for retrieval-augmented generation systems.

Rewrite the following user query into a retrieval-optimized query.
The rewritten query should:
1. Preserve the original intent
2. Be clear and unambiguous
3. Improve retrievability in a knowledge base
4. Avoid adding new information

User query: {query}
Rewrited query:
"""

PROMPT_NAME = "rafe_sft_rewrite"


class ConstraintRewriteService:
    """约束保持查询重写服务：对子问题生成一条重写查询，用于自反思二次检索。"""

    def __init__(
        self,
        *,
        rewrite_llm: Any,                    # 重写专用 LLM（可带 LoRA adapter）
        sample_one: Callable[..., Any],      # 注入自 pipeline，保持缓存行为一致
        max_tokens_rewrite: int = 64,        # 重写输出最大 token 数
    ):
        self.rewrite_llm = rewrite_llm
        self._sample_one = sample_one
        self.max_tokens_rewrite = max_tokens_rewrite

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def rewrite(self, query: str) -> dict:
        """对 query 生成一条约束保持重写，返回与第二章候选格式兼容的 dict。

        返回字段：
          prompt_name       : "rafe_sft_rewrite"
          prompt_text       : 实际使用的 prompt 字符串
          rewrite           : 模型原始输出（未清洗）
          normalized_rewrite: 清洗后的单行检索查询
        """
        prompt = RAFE_SFT_REWRITE_PROMPT.format(query=query)

        # 贪婪解码（temperature=0.0），单次生成，不做多组采样
        raw_output = self._sample_one(
            self.rewrite_llm,
            prompt=prompt,
            temperature=0.0,
            max_tokens=self.max_tokens_rewrite,
            top_p=1.0,
        )
        normalized = self._normalize(str(raw_output), fallback_query=query)
        return {
            "prompt_name": PROMPT_NAME,
            "prompt_text": prompt,
            "rewrite": str(raw_output).strip(),
            "normalized_rewrite": normalized,
        }

    # ------------------------------------------------------------------
    # 内部归一化（与第二章 RewriteService._normalize_rewrite_text 保持一致）
    # ------------------------------------------------------------------

    def _normalize(self, text: str, fallback_query: str) -> str:
        """将模型输出清洗为单行可用的检索查询字符串。

        处理步骤：
        1. 统一换行符
        2. 去掉项目符号、序号前缀
        3. 跳过解释性开头行（"here is"、"because" 等）
        4. 去掉常见输出前缀标签（"rewritten query:" 等）
        5. 压缩空白、去掉外围引号
        6. 若结果为空则回退到原始 query
        """
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        cleaned = ""

        for line in lines:
            # 去掉列表符号和序号
            line = re.sub(r"^[-*•]+\s*", "", line)
            line = re.sub(r"^\d+[.)]\s*", "", line)
            if not line:
                continue
            # 跳过解释性开头，这类行不是有效查询
            low = line.lower()
            if low.startswith(("here is", "explanation", "because", "i rewrote", "this query")):
                continue
            cleaned = line
            break

        if not cleaned:
            cleaned = lines[0] if lines else ""

        # 去掉常见输出前缀标签
        prefixes = [
            "optimized search query:",
            "search query:",
            "rewritten query:",
            "rewritten search query:",
            "final query:",
            "rewrite:",
            "query:",
        ]
        low = cleaned.lower()
        for prefix in prefixes:
            if low.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # 压缩多余空白，去掉外围引号/反引号
        cleaned = re.sub(r"\s+", " ", cleaned).strip('"\'` ')
        return cleaned if cleaned else fallback_query
