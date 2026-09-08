# QA数据集处理流水线

## 📚 项目简介

完整的问答数据集处理流水线，从对话数据生成高质量的长上下文记忆评估数据集。

## ✨ 核心流程


当前流程顺序为：

1. 生成初始题目，并由同一模型执行独立的第二遍自我反思：逐题检查结构、证据、答案唯一性、类别与 label，再修订或替换有问题的题目。
2. 问题精炼与重构。
3. 将题目统一拼接为 `question + option` 展示形式，并补齐 F 选项。
4. 证据对齐，确保 `evidence_dialogues` 中的 `dia_id`、`speaker` 与原始 conversation 严格对应，并且 `utterance` 是同一原始 turn 的完整文本或连续原文片段。
5. 题目合理性检测：仅证据回答、迭代性证据删除。
6. 污染检查。
7. 最终筛选：先执行确定性的 schema / evidence / reasoning 检查，再由模型执行语义质量检查；任一检查失败的 QA 都从 final 中删除，并写入删除审计文件。

注意：答案为 `F` 的 Abstain 题仍不进入 Step 2 的仅证据回答、迭代性证据删除和 Step 3 污染检查，但会进入 Step 4 的最终 schema 与语义门禁。最终评审会检查 A-E 是否被所给证据支持、题干和选项是否存在歧义；对整部作品执行穷尽式的 A-E 逐项证伪仍可作为后续人工校验环节。

迭代性证据删除阶段会把删除证据后的剩余 conversation 以 JSONL 形式传入模型，每行都显式包含 `dia_id`、`speaker`、`utterance`。如果模型答对但没有返回可与剩余 conversation 严格对齐的新证据，本轮会重试；多次重试仍失败时，该题会被标记为应过滤，不再额外追问模型补证据。

若模型把真实对话编号误写到 `id` 字段中（例如 `"id": "D1:29"` 且缺少 `dia_id`），系统会先做安全规范化：只有当该编号确实存在于当前剩余 conversation 中时，才将其移入 `dia_id`，并把 `id` 重写为 `E1/E2/...`。规范化后仍必须通过严格证据对齐。

证据对齐允许连续原文片段作为合法 evidence，不强制保存完整 turn；但不接受改写、概括、非连续拼接、错误 `dia_id` 或错误 `speaker`。对齐报告中的 `match_type_counts` 会区分 `exact_turn`、`contiguous_excerpt` 和 `ordered_ellipsis`，其中 `contiguous_excerpt` 表示合法的连续原文片段。

## 🧱 项目结构

- `main.py`: 流水线命令行入口，只负责阶段编排。
- `config.py` / `config.example.json`: 配置加载、版本说明和示例配置。
- `src/pipeline_utils.py`: 流水线通用工具，包括版本路径、运行日志、输入回退和累积过滤。
- `src/step0_qa_generate.py`: 初始题目生成。
- `src/step1_new_qa.py`: 问题精炼与重构。
- `src/step2_evidence_check.py`: 题目合理性检测。
- `src/step3_pollution_check.py`: 污染检查。
- `src/step4_finalize.py`: 最终 schema 与语义质量门禁，以及删题审计。
- `dataset/`: 输入数据。
- `runs/`: 每次运行的独立目录，包含日志、中间版本和最终输出。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 首次配置
编辑 config.json 填入你的 API 配置，然后验证：
```bash
python main.py --show-config
```

### 输入数据格式

运行命令中的 `DATASET` 对应 `dataset/` 下的同名目录。例如：

```bash
python main.py --run An-Enemy-of-the-People
```

程序会读取：

```text
dataset/
  An-Enemy-of-the-People/
    An-Enemy-of-the-People_1.json
    An-Enemy-of-the-People_2.json
    ...
```

也可以直接传入单个 JSON 文件或一个包含多个 JSON 的文件夹：

```bash
python main.py --run dataset/standard_ebooks_trace/dracula.json
python main.py --run dataset/standard_ebooks_trace
```

传入单个文件时，该文件会作为一个独立数据集运行；传入文件夹时，文件夹内每个 `.json` 会作为独立 record 汇总到同一次运行的 v0/v1/v2/v3/final 产物中。若希望每本书生成完全独立的 runs 目录和 final 文件，可以一次传多个文件：

```bash
python main.py --run dataset/standard_ebooks_trace/dracula.json dataset/standard_ebooks_trace/jane_eyre.json
```

也可以在 `config.json` 的 `pipeline.run_targets` 中配置多个文件或文件夹，并用 `pipeline.batch_max_workers` 控制并行运行数量。此时不传 `--run` 会自动执行这些目标：

```json
"pipeline": {
  "input_dir": "dataset",
  "runs_dir": "runs",
  "run_targets": [
    "dataset/standard_ebooks_trace/dracula.json",
    "dataset/standard_ebooks_trace/jane_eyre.json",
    "dataset/standard_ebooks_trace"
  ],
  "batch_max_workers": 2
}
```

每个输入文件必须是 UTF-8 编码的 JSON，顶层可以是一个对象，也可以是只包含一个对象的数组。推荐使用对象格式：

```json
{
  "conversation": {
    "speakers": ["Nora", "Torvald"],
    "session_1_date_time": "2024-01-01 10:00:00",
    "session_1": [
      {
        "dia_id": "1-1",
        "speaker": "Nora",
        "utterance": "..."
      },
      {
        "dia_id": "1-2",
        "speaker": "Torvald",
        "utterance": "..."
      }
    ],
    "session_2_date_time": "2024-01-02 10:00:00",
    "session_2": [
      {
        "dia_id": "2-1",
        "speaker": "Nora",
        "utterance": "..."
      }
    ]
  },
  "qa": []
}
```

字段说明：

- `conversation`: 必填，表示当前文件对应的剧本/对话内容。
- `conversation.speakers`: 推荐填写，说话者列表。若缺失，程序会尝试从 `speaker_1`、`speaker_2` 等字段提取；仍缺失时会使用默认兜底角色列表。
- `session_N`: 推荐格式，表示第 N 段对话，值为对话轮次数组。
- `session_N_date_time`: 可选，表示该段对话时间。
- `dia_id`: 推荐填写，表示单条对话证据 ID，后续证据定位会使用。
- `speaker`: 推荐填写，表示该轮说话者。
- `utterance`: 推荐填写，表示该轮对话文本。
- `qa`: 可选。若已有题目且 `force_generate_new_qa=false`，步骤 0 会复用已有 `qa`；若为空或强制重建，则由模型生成初始题目。

已有 `qa` 推荐格式如下：

```json
{
  "question": "...",
  "option": [
    "A. ...",
    "B. ...",
    "C. ...",
    "D. ...",
    "E. ..."
  ],
  "answer": "A",
  "category": 2,
  "label": "Fact Extraction (Multiple Dialogues)",
  "evidence_dialogues": [
    {
      "id": "E1",
      "dia_id": "1-1",
      "speaker": "Nora",
      "utterance": "..."
    }
  ],
  "reasoning_steps": ["..."]
}
```

单选题的 `option` 可以只提供 A-E，流水线会在格式化阶段自动补齐：

```text
F. Cannot infer the answer based on the given information.
```

### Question Type

- 未提供 `question_type` 时自动补为 `single_choice`；值为空或无法识别时也按单选题处理。
- `question_type: "multiple_choice"` 表示多选题，答案格式为 `"(A,E)"`；评分要求选项集合完整一致，但不要求顺序一致。
- `question_type: "ordering"` 表示排序题，答案格式为 `"(B,A,D,C)"`；评分要求选项及顺序完全一致。
- `answer` 也兼容字母数组，例如 `["A", "B", "E", "F"]` 会被视为一个组合答案；多选题按集合评分，排序题保留数组顺序。单选题仅接受单元素数组。
- `answer_fixed` 的数组仍表示多个可接受的候选答案，不会被合并成一个组合答案。
- 多选题和排序题不会自动补充 F 选项。单选题继续自动补充 F，并保持旧数据兼容。

### Category 定义

- `Category 1 - User Profile Category`: 稳定长期属性，例如 demographics、core values、persistent habits，用于检查 persona consistency。
- `Category 2 - Event-based Category`: 离散结构化事实与具体行为，强调 5W1H。
- `Category 3 - Temporal Evolution Category`: 随时间变化的状态转移，要求模型识别新信息如何更新旧记忆。
- `Category 4 - Social Relationship & Interaction Category`: 人际网络、互动模式、显式关系与隐式情绪细节。
- `Category 5 - Fine-grained Data Category`: 高精度细节记忆，例如具体数字、字符串、代码片段或角色特定表述。
- `Category 6 - Lessons Learned Category`: 反思过往反馈并将纠错经验迁移到未来策略。
- `Category 7 - Plans & Commitments Category`: 前瞻性记忆，包括未来任务、计划事件和承诺触发条件。

### Label 定义

- `Fact Extraction (Single Dialogue)`: 正确答案可由单个 dialogue session 完全推出，干扰项来自其他 session 或角色。
- `Fact Extraction (Multiple Dialogues)`: 关键线索分散在多个彼此分离的 dialogue turns 或 passages 中，必须组合才能得到答案；同一 session 内跨度较大的分离片段也符合此定义。
- `Memory Update`: 同一事实在不同时间被更新，题目应奖励识别最新版本，旧版本应作为强干扰项。
- `Multi-hop`: 至少需要两个由对话证据支撑的推理步骤，单个 utterance 不足以直接推出答案。
- `Abstain`: A-E 必须全部是看似合理但错误或无法由对话支持的干扰项，`answer` 必须设为 `F`。

`Temporal Evolution` 只作为 Category 3 的能力类别使用，不再作为 label。对于
`question_type: "ordering"` 的题目，label 仍按证据组织方式确定：多个分散 dialogue
明确给出待排序事件时使用 `Fact Extraction (Multiple Dialogues)`；需要用新信息覆盖旧
状态时使用 `Memory Update`；顺序依赖额外身份、因果或关系推导时使用 `Multi-hop`。

### 运行完整流水线
```bash
python main.py --run An-Enemy-of-the-People
```

### 只运行特定步骤
```bash
python main.py --run An-Enemy-of-the-People --start 3 --end 3
```

### 从已有运行记录继续

```bash
python main.py --resume-run runs/An-Enemy-of-the-People_20260828_120000
```

恢复运行会复用原目录中的 `temp/`、`result/` 和 `run.log`，自动定位第一个未完成阶段。当前 Step 2 的 v2a 与 v2b 会保存完整题目快照，并跳过已有有效终态的题目；API 超时、网络异常或响应解析耗尽的题目会在下次恢复时重试。也可以用 `--start` 和 `--end` 限制恢复范围。

### 使用自定义配置文件
```bash
python main.py --run An-Enemy-of-the-People --config my_config.json
```

### 配置项说明

**步骤配置（steps）**
- step_0_generate_qa: 生成原始问答对，执行模型自我反思和修订，并统一格式化为 question + A-E/F 选项展示文本
- step_1_refine_qa: 问题精炼与重构，并再次统一格式化为 question + A-E/F 选项展示文本 (v0 → v1_refined)
- step_2_evidence_check: 题目合理性检测 (v1_refined → v2a → v2b)
- step_3_pollution_check: 污染检查 (v2b → v3)
- step_4_finalize: 累积前序过滤规则，执行最终 schema 与语义检查，删除不合格 QA 并生成 final

**Pipeline 配置**
- input_dir: 输入数据集目录
- runs_dir: 每次运行的独立输出目录根路径
- max_workers: 各步骤未单独配置时使用的并发兜底值

**分步骤并发配置**
- `step_0_generate_qa.max_workers`: 同一文件内角色批次的并发数；`speaker_batch_size` 决定每个任务包含多少角色
- `step_0_generate_qa.enable_self_reflection`: 是否对模型第一遍生成结果执行独立的反思修订回合，默认 `true`
- `step_1_refine_qa.max_workers`: 按角色并发执行问题精炼
- `step_2_evidence_check.only_evidence_max_workers`: 仅证据回答的并发数
- `step_2_evidence_check.iterative_ablation_max_workers`: 五轮迭代证据消融的并发数
- `step_2_evidence_check.max_workers`: 单独运行其他证据检查模式时的并发数
- `step_2_evidence_check.checkpoint_every_questions`: Step 2 每完成多少道题原子更新一次阶段快照，默认 `1`
- `step_3_pollution_check.max_workers`: 每轮无上下文污染回答的并发数
- `step_4_finalize.enable_schema_check`: 是否执行最终确定性结构检查，默认 `true`
- `step_4_finalize.enable_semantic_check`: 是否执行最终逐题语义检查，默认 `true`
- `step_4_finalize.max_workers`: 最终语义检查的并发数

Step 4 的语义检查启用时需要 LLM。优先使用 `step_4_finalize.llm`；为兼容旧配置，如果该项未配置，会回退使用 `step_3_pollution_check.llm`。最终保留题写入 `*_final.json`，删除原因写入同目录的 `*_final_review.json`。如果任何语义评审请求最终失败，Step 4 会整体失败且不写出新的 final，避免把 API 故障误判成应删题。

Step 0 的第二遍输出还包含精简的 `reflection_summary`（发现问题数、修订题数和问题代码）；流水线把该摘要写入运行日志，只把修订后的 `qa` 写入 v0，不保存模型的隐式推理过程。

各步骤存在版本依赖，因此步骤之间保持顺序执行；上述并发均发生在步骤内部。并发过高可能触发模型服务的 TPM/RPM 限制，迭代消融建议从 3 开始调整。

**长文本的迭代证据消融**

步骤 2 的 `v2b` 消融会先估算删除当前证据后的完整提示长度。提示可容纳时继续使用原来的整段上下文流程；超出上限时，按完整对话 turn 分块并保留原始 `dia_id` 和全局顺序。系统先检索最相关的块以尽快发现可继续删除的新证据；检索没有发现新证据时，会扫描全部剩余块并归并精确证据后再作答。检索结果不能单独证明“已无证据”，只有完整扫描成功才能判定通过；任一块反复失败都会将题目标记为 `needs_rerun`。

- `step_2_evidence_check.ablation_context_limit`: 模型上下文容量，默认 `32768`
- `step_2_evidence_check.ablation_prompt_safety_tokens`: 为指令和输出预留的 token，默认 `4096`
- `step_2_evidence_check.ablation_chunk_tokens`: 每个对话源块的最大 token，默认 `8192`
- `step_2_evidence_check.ablation_retrieval_chunks`: 快速检索阶段扫描的块数，默认 `6`
- `step_2_evidence_check.ablation_chunk_max_workers`: 单题内部并行扫描块的上限，默认 `4`

`iterative_ablation_max_workers` 控制同时处理多少道题，`ablation_chunk_max_workers` 控制每道长文本题内部同时扫描多少块；两者相乘会放大并发请求量，应结合模型服务限额调整。

每次运行会创建：

```text
runs/{dataset}_{时间戳}/
  run.log
  temp/
    xxx_v0.json
    xxx_v1_refined.json
    xxx_v2a.json
    xxx_v2b.json
    xxx_v3.json
  result/
    xxx_final.json
```

`run.log` 会记录本次运行的结构化过程信息：

- `RUN START`: 数据集、步骤范围、运行目录、配置摘要。
- `STAGE N START`: 当前阶段目标、输入文件快照、预计输出、阶段配置。
- `Execution detail`: 阶段内部的详细执行日志。
- `STAGE N END`: 阶段状态、耗时、实际输出文件统计。
- `RUN SUMMARY`: 总耗时、各阶段耗时、v0/v1_refined/v2a/v2b/v3/final 的最终产物快照。
