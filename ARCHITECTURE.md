# 第四章：基于自反思的自适应检索增强生成

## 概述

本目录实现了论文第四章的核心实验代码：在第二章迭代问题分解-检索-求解框架的基础上，引入轻量级判别模型对检索文档质量进行自反思评估，并根据评估结果自适应地触发约束保持查询重写，从而提升检索质量。

## 目录结构

```
chapter4-self-reflection/
├── self_reflection_workflow/
│   ├── __init__.py
│   ├── evaluator_service.py          # T5 判别模型服务（文档质量路由）
│   ├── constraint_rewrite_service.py # 约束保持查询重写服务
│   └── pipeline.py                   # 自反思迭代 RAG 主流程
├── experiments/
│   └── self_reflection_config.json   # 运行配置模板
├── outputs/
│   └── cache/                        # LLM 推理结果缓存
└── run_self_reflection_batch.py      # 批量评测入口脚本
```

## 与第二章的关系

| 模块 | 第二章 | 第四章 |
|------|--------|--------|
| 子问题生成 | `pipeline._generate_next_subquestion` | 完全复用，不修改 |
| 迭代控制 | `pipeline._control_iteration` | 完全复用，不修改 |
| 最终答案生成 | `pipeline._build_final_answer` | 完全复用，不修改 |
| 检索 | 多模板重写 + 融合检索 | 单次 BM25 检索 + 自反思路由 |
| 查询重写 | 三种模板 × 多组采样 | 仅约束保持重写，贪婪解码单次生成 |
| 文档质量评估 | 无 | T5 判别模型三路路由 |

## 核心流程

每轮迭代的执行顺序：

```
主问题
  │
  ▼
子问题生成（general LLM，与第二章一致）
  │
  ▼
初始检索（BM25，用原始子问题）
  │
  ▼
自反思路由（T5 判别模型评估文档质量）
  ├── correct   ──────────────────────────────────────────┐
  ├── ambiguous → 约束保持重写 → 再检索 → 合并原始文档 ──┤
  └── incorrect → 约束保持重写 → 再检索 → 替换原始文档 ──┘
                                                          │
                                                          ▼
                                                    子答案合成
                                                          │
                                                          ▼
                                                    迭代控制（与第二章一致）
```

## 模块说明

### evaluator_service.py

封装 `evaluatator_model/` 中的 T5-large 序列分类模型。

- **输入格式**：`"question [SEP] passage"`（与 CRAG 训练格式一致）
- **输出**：原始 logit 分数，映射到三个路由标签
- **阈值语义**：
  - `score >= upper_threshold` → `correct`（文档高度相关）
  - `lower_threshold <= score < upper_threshold` → `ambiguous`（部分相关）
  - `score < lower_threshold` → `incorrect`（不相关）
- **聚合规则**：多文档取最高优先级（correct > ambiguous > incorrect）

### constraint_rewrite_service.py

约束保持查询重写，仅在自反思路由判定为 ambiguous/incorrect 时触发。

- 使用 RAFE-SFT prompt（保留原始意图、消除歧义、不引入新信息）
- `temperature=0.0` 贪婪解码，单次生成（不做多组采样）
- 归一化逻辑与第二章 `RewriteService` 完全一致

### pipeline.py

主流程类 `SelfReflectionPipeline`，核心新增方法为 `_apply_self_reflection`：

```python
def _apply_self_reflection(sub_question, initial_docs):
    agg_label, per_doc_results = evaluator.evaluate_docs(sub_question, initial_docs)
    if agg_label == "correct":
        return initial_docs, trace
    rewrite_result = constraint_rewrite.rewrite(sub_question)
    rewrite_docs = retrieve_raw(rewrite_result["normalized_rewrite"])
    if agg_label == "ambiguous":
        return merge_dedup(initial_docs, rewrite_docs), trace  # 合并
    else:  # incorrect
        return rewrite_docs, trace                              # 替换
```

## 快速开始

### 1. 配置实验参数

编辑 [experiments/self_reflection_config.json](experiments/self_reflection_config.json)，主要参数：

```json
{
  "common": {
    "evaluator_model_path": "../evaluatator_model",
    "evaluator_upper_threshold": 10.0,
    "evaluator_lower_threshold": -10.0,
    "max_iterations": 4,
    "top_k": 10,
    "corpus_name": "hotpotpqa"
  },
  "batch_test": {
    "dataset": "../chapter2-rewrite/src/Dataset/hotpot_train_test_sample_200.json"
  }
}
```

### 2. 运行批量评测

```bash
cd chapter4-self-reflection
python run_self_reflection_batch.py --config experiments/self_reflection_config.json
```

打印合并后的运行时配置（不执行推理）：

```bash
python run_self_reflection_batch.py --print-config
```

### 3. 查看结果

- 逐样本结果：`outputs/self_reflection/<dataset>-<size>.jsonl`
- 汇总指标：`outputs/self_reflection/<dataset>-<size>-summary.json`

汇总文件包含 `avg_em`、`avg_f1`、`avg_retrieved_times` 等关键指标。

## 依赖

本模块复用第二章的以下组件（通过 `sys.path` 引入，无需安装）：

- `chapter2-rewrite/src/models/chat_llm.py` — LLM 加载与采样
- `chapter2-rewrite/iterative_workflow/elasticsearch_retriever.py` — BM25 检索
- `chapter2-rewrite/iterative_workflow/prompt_templates.py` — prompt 构建
- `chapter2-rewrite/evaluate_em_f1.py` — EM/F1 评测

外部依赖（与第二章一致）：

```
torch
transformers
elasticsearch
modelscope
peft (可选，用于 LoRA adapter)
```

## 实验配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `evaluator_upper_threshold` | 10.0 | T5 分数高于此值判定为 correct |
| `evaluator_lower_threshold` | -10.0 | T5 分数低于此值判定为 incorrect |
| `max_iterations` | 4 | 每个问题最多迭代轮数 |
| `top_k` | 10 | 每轮保留的最终文档数 |
| `max_tokens_rewrite` | 64 | 重写输出最大 token 数 |
| `enable_llm_cache` | true | 启用 LLM 推理结果缓存 |
| `general_on_cpu` | false | general 模型是否放 CPU（显存不足时开启） |
