# QA数据集处理流水线

## 📚 项目简介

完整的问答数据集处理流水线，从对话数据生成高质量的长上下文记忆评估数据集。

## ✨ 核心流程

![alt text](<image/Category-Based Evidence-2026-02-12-144152.png>)

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

### 运行完整流水线
```bash
python main.py --run An-Enemy-of-the-People
```

### 只运行特定步骤
```bash
python main.py --run An-Enemy-of-the-People --start 3 --end 3
```

### 使用自定义配置文件
```bash
python main.py --run An-Enemy-of-the-People --config my_config.json
```

### 配置文件结构
```json
{
        "pipeline": {
                "input_dir": "输入目录",
                "output_dir": "输出目录",
                "temp_dir": "临时目录",
                "max_workers": 4
        },
        "embedding": {
                "model_name": "嵌入模型名称",
                "clustering_threshold": 0.55
        },
        "steps": {
                "step_0_generate_qa": {
                        "function": "generate_qa_v0",
                        "api_required": true,
                        "description": "步骤 0: 生成原始问答对",
                        "llm": { "model": "...", "base_url": "...", "api_key": "..." }
                },
                "step_1_pollution_check": {
                        "function": "pollution_check_main",
                        "api_required": true,
                        "description": "步骤 1: 选项打乱和污染检查",
                        "enable_contamination_check": true,
                        "llm": { "model": "...", "base_url": "...", "api_key": "..." }
                },
                "step_2_full_context": {
                        "function": "full_context_main",
                        "api_required": true,
                        "description": "步骤 2: 题目合理性验证",
                        "llm": { "model": "...", "base_url": "...", "api_key": "..." }
                },
                "step_3_label": {
                        "function": "label_main",
                        "api_required": true,
                        "description": "步骤 3: 问题标注",
                        "llm": { "model": "...", "base_url": "...", "api_key": "..." }
                },
                "step_4_new_qa": {
                        "function": "new_qa_main",
                        "api_required": true,
                        "description": "步骤 4: 问答精炼重构",
                        "llm": { "model": "...", "base_url": "...", "api_key": "..." }
                }
        },
        "tools": {
                "option_perturbation": {
                        "function": "option_perturbation",
                        "api_required": true,
                        "description": "选项扰动生成和评分",
                        "llm": { "gen_model": "...", "score_model": "...", "base_url": "...", "api_key": "..." }
                }
        }
}
```

### 配置项说明

**步骤配置（steps）**
- step_0_generate_qa: 生成原始问答对
- step_1_pollution_check: 选项打乱和污染检查
- step_2_full_context: 题目合理性验证
- step_3_label: 问题标注
- step_4_new_qa: 问答精炼重构

**工具配置（tools）**
- option_perturbation: 选项扰动生成与评分

**Pipeline 配置**
- input_dir: 输入数据集目录
- output_dir: 最终输出目录
- temp_dir: 中间文件目录
- max_workers: 并发处理数

**Embedding 配置**
- model_name: SentenceTransformer 模型名称
- clustering_threshold: 语义聚类阈值

## 📁 文件结构

```
personal_memory_copy/
├── config.json           # 配置文件（主要修改这个）
├── config.py             # 配置加载器
├── main.py               # 主入口程序
├── option_perturbation.py
├── overlap_curve.py
├── dataset/
├── result/
├── temp/
└── src/
        ├── new_qa.py
        ├── label.py
        ├── full_context.py
        ├── pollution_check.py
        ├── qa_generate.py
        └── qa_only_response.py
```

