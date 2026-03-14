# QA数据集处理流水线

## 📚 项目简介

完整的问答数据集处理流水线，从对话数据生成高质量的长上下文记忆评估数据集。

## ✨ 核心流程

![alt text](<image/Flowchart.png>)

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

### 配置项说明

**步骤配置（steps）**
- step_0_generate_qa: 生成原始问答对
- step_1_evidence_context: 题目合理性检测 (v0 → v1)
- step_2_label: 题目标注 (v1 → v2)
- step_3_new_qa: 问答精炼重构 (v2 → v3)
- step_4_pollution_check: 题目乱序 (v3 → v4)
- step_5_finalize: 生成最终版本 (v4 → final)

**工具配置（tools）**
- option_perturbation: 选项扰动生成与评分

**Pipeline 配置**
- input_dir: 输入数据集目录
- output_dir: 最终输出目录
- temp_dir: 中间文件目录
- max_workers: 并发处理数

