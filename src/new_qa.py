import json
import os
import re
import numpy as np
from collections import defaultdict, OrderedDict
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("NEW_QA_MODEL", "Qwen/Qwen3-14B")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEFAULT_CLUSTERING_THRESHOLD = float(os.getenv("CLUSTERING_THRESHOLD", "0.55"))

BASE_URL = DEFAULT_BASE_URL
API_KEY = DEFAULT_API_KEY
MODEL = DEFAULT_MODEL
CLUSTERING_THRESHOLD = DEFAULT_CLUSTERING_THRESHOLD

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

def gen_chat(prompt: str, temp=0.7) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"!!! 调用模型时发生错误: {e}")
        return ""

# ================= 核心处理类 =================

class UltimateMemoryRefiner:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.raw_data = []
        self.original_data = []  # 保存原始的完整数据结构

    def load_data(self):
        """加载数据并标记 episode_index"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 如果是单个 dict，转换为列表
        if isinstance(data, dict):
            data = [data]
        
        # 保存原始数据结构
        self.original_data = data
        
        # 提取所有 QA 并添加 qid（全局唯一ID）、episode_index 和 source_index
        qid = 1
        for idx, item in enumerate(data):
            if "qa" in item:
                for qa_item in item["qa"]:
                    # 如果原来没有 qid，生成一个
                    if "qid" not in qa_item:
                        qa_item["qid"] = qid
                        qid += 1
                    qa_item["episode_index"] = idx
                    qa_item["source_index"] = idx  # 标记来源item
                    self.raw_data.append(qa_item)
        
        print(f"--- [Step 1] 数据加载完成，共计 {len(self.raw_data)} 条原始 QA，来自 {len(data)} 个数据源 ---")

    def semantic_clustering(self, qa_list, threshold=None):
        """语义聚类并输出最大簇 Size"""
        if threshold is None:
            threshold = CLUSTERING_THRESHOLD
        
        sentences = [q["question"] for q in qa_list]
        embeddings = embedder.encode(sentences, convert_to_tensor=True)
        
        # 社区检测算法
        clusters = util.community_detection(embeddings, min_community_size=2, threshold=threshold)
        
        if not clusters:
            return [[qa] for qa in qa_list], 1
        
        max_cluster_size = max(len(c) for c in clusters)
        
        clustered_data = []
        assigned_indices = set()
        for cluster in clusters:
            clustered_data.append([qa_list[idx] for idx in cluster])
            assigned_indices.update(cluster)
            
        remaining = [qa_list[i] for i in range(len(qa_list)) if i not in assigned_indices]
        for r in remaining: clustered_data.append([r])
            
        return clustered_data, max_cluster_size

    def scan_cluster_logic(self, subject, cluster):
        """
        [新增环节]：使用 LLM 扫描聚类结果，判断其是否具备“重构价值”
        """
        if len(cluster) < 2: return cluster # 孤立点不扫描
        
        prompt = f"""
你是一个逻辑分析师。请审视以下关于实体【{subject}】的语义相关问题簇。
你的任务是：
1. 识别这些问题是否围绕同一个属性或事件展开。
2. 重点寻找随序列号(episode_index)增加而发生的状态冲突（记忆更新点）或信息互补（逻辑整合点）。
3. 如果这组数据逻辑混乱或无关联，请返回 "REJECT"，否则返回简短的逻辑演变描述。

### 数据簇：
{json.dumps(cluster, ensure_ascii=False, indent=2)}
"""
        res = gen_chat(prompt, temp=0.3)
        return None if "REJECT" in res else cluster

    def build_refine_prompt(self, subject, final_chunk):
        """最终生成 Prompt"""
        return f"""
# Task Description
You are a high-difficulty long-context logical evaluation and question generation system.

You will be given a set of original QA data about a specific entity {subject}.
All data is ordered temporally, but information is distributed across multiple non-contiguous semantic chunks.

Your task is to generate a set of challenging logical evaluation questions that assess:
- Long-term memory update ability
- Cross-chunk logical integration ability

All questions must be derived strictly from the provided data.
They must require temporal reasoning, state comparison, conflict resolution, and multi-evidence integration, rather than surface-level paraphrasing.

# Dimension 1: Memory Update

When an entity’s state S is A at time t1 and is later explicitly or implicitly updated to B at time t2, you must construct questions around this state transition, including but not limited to:

1. Causal (Why-based)
   - Ask why state A became invalid
   - Ask which explicitly mentioned events, decisions, or conditions caused or enabled the transition to state B

2. Boundary / Timing
   - Ask for the specific point at which the old state was irreversibly overturned
   - This point is not necessarily the first anomaly, but when the update became final

3. Final-State Verification
   - Explicitly include early state A as a strong distractor in the question
   - Ask about the entity’s final state at the end of the full data sequence
   - Designed to detect reliance on outdated memory

For each subtype above:
If multiple updates, reversals, or influencing factors exist in the data, you should generate multiple questions from different analytical perspectives, not just a single question.


# Dimension 2: Integrated Logic Across Chunks

You must actively identify related facts or patterns distributed across multiple non-adjacent semantic chunks and construct questions that require joint reasoning across them, including but not limited to:

1. Set Construction / Inductive Aggregation
   - Ask the model to enumerate all moments or behaviors matching
     an abstract property
   - The property must not be explicitly summarized in any single chunk

2. Trend / Frequency Analysis
   - Ask whether a behavior, attitude, or decision pattern changes over time
   - Changes may involve escalation, attenuation, or structural shifts

3. Multi-Chunk Dependency (Fragment Assembly)
   - The correct answer must depend on information from at least two different semantic chunks
   - Missing any chunk should lead to an incomplete or incorrect answer

The same cross-chunk pattern may be queried from multiple angles, and multiple questions should be generated when appropriate.


# Mandatory Constraints

1. No External Knowledge
   - All questions, options, and answers must be based exclusively on the provided data
   - No background knowledge, common sense completion, or assumptions allowed

2. No Meta-Context References
   - Do not mention “episodes”, “chapters”, “earlier text”, or similar notions
   - Treat the input strictly as a complete and standalone data sequence

3. Implicit Reasoning Requirement
   - Questions must implicitly require deep reasoning
   - The model is NOT required to expose reasoning in its answer


# Option Construction Constraints

1. Each question must provide multiple options (e.g., A / B / C / D / E).

2. Incorrect options must be plausible but unambiguously wrong:
   - They must be partially supported by the text
   - But invalidated by later updates or cross-chunk evidence

3. Common sources of incorrect options include:
   - Reliance on early states while ignoring updates
   - Use of a single chunk while ignoring others
   - Confusing correlation with causation

4. The correct option must not be obtainable via keyword matching alone;
   it must require temporal ordering, state comparison, or evidence integration.


# Answer Field Constraints

1. The `answer` field must contain ONLY the final conclusion:
   - e.g., the correct option letter, final state, key timestamp, or final set
   - No explanation, justification, or reasoning text

2. The answer must be unique and deterministic.
   Ambiguous or multi-valid answers are not allowed.


# Explanation / Reasoning Field Constraints

1. The `reasoning` (or `explanation`) field must document:
   - The logical basis for the correct answer
   - How memory updates occurred
   - How information from multiple chunks was integrated
   - How conflicts were identified and resolved

2. This field exists for:
   - Annotation quality control
   - Debugging and error analysis
   - Benchmark interpretability

3. Every key claim in the reasoning must be traceable
   to a specific semantic chunk or time point.

4. "category": The question type label, which must remain consistent with the category of the selected original question(s) from which this item is constructed.

5. "original_qa_qid": A list of qid values corresponding to the original QA items that were selected, referenced, or integrated to construct the current question.

# Input Semantic Cluster
{json.dumps(final_chunk, ensure_ascii=False, indent=4)}


# Output Format (JSON)
[
    {{
        "question": "Complete question text with necessary distractors",
        "option": ["A ...", "B ...", "C ...", "D ...", "E ..."],
        "answer": "Final conclusion or correct option label",
        "reasoning": "Detailed explanation of how the answer follows from the full data",
        "label": "memory_update / integrated_logic",
        "category": 1,
        "evidence_chunks": [0, 2, 5],
        "is_conflict": true / false,
        "original_qa": ["Referenced original question(s) or summaries"],
        "original_qa_qid": [1, 4] 
    }}
]

"""

    def process(self):
        """
        处理流程：对所有问题进行全局聚类分析和重构
        """
        self.load_data()
        
        print(f"\n>>> 开始全局聚类分析和重构处理")
        
        # 按角色分组（全局）
        subject_buckets = defaultdict(list)
        for qa in self.raw_data:
            subject_buckets[qa.get("character", "Unknown")].append(qa)
        
        # 存储所有重构后的新问题
        all_refined_qa = []
        
        for subject, subject_qa_list in subject_buckets.items():
            if subject == "Unknown": 
                continue
            
            print(f"\n>>> 正在处理角色【{subject}】(共 {len(subject_qa_list)} 个问题)...")
            
            # 语义聚类
            semantic_groups, max_size = self.semantic_clustering(subject_qa_list)
            print(f"   √ 语义聚类完成。最大簇大小: {max_size}")
            
            # LLM 逻辑审校和重构
            print(f"   正在进行 LLM 逻辑审校 (共 {len(semantic_groups)} 个话题簇)...")
            for i, group in enumerate(semantic_groups):
                if len(group) < 2:
                    # 孤立点跳过，不生成新问题
                    print(f"      - 簇 {i} 为孤立点，跳过")
                    continue
                
                passed_group = self.scan_cluster_logic(subject, group)
                if not passed_group:
                    print(f"      - 簇 {i} 被逻辑审校拒绝 (无冲突或关联)，跳过")
                    continue
                
                # 生成重构后的问题
                refined = None
                max_retries = 10
                
                for attempt in range(max_retries):
                    current_prompt = self.build_refine_prompt(subject, passed_group)
                    if attempt > 0:
                        current_prompt += "\n\n**重要修正**：请直接输出 JSON 数组格式（以 [ 开头，以 ] 结束），严禁包含任何 Markdown 代码块标签、前言、解释或结尾总结。"
                    
                    response = gen_chat(current_prompt)
                    refined = self.extract_json(response)
                    
                    if refined:
                        all_refined_qa.extend(refined)
                        print(f"      √ 簇 {i} 重构成功 (尝试 {attempt+1} 次): {len(refined)} 道新题")
                        break
                    else:
                        print(f"      ! 簇 {i} 第 {attempt+1} 次解析失败，正在重试...")
                
                if not refined:
                    print(f"      × 簇 {i} 在 {max_retries} 次尝试后失败，跳过")
        
        # Statistics reported before processing started
        self.save(all_refined_qa)

    def extract_json(self, text):
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(match.group()) if match else None
        except: return None

    def save(self, refined_qa_list):
        """
        保存重构后的数据，合并新问题和原始问题
        """
        print(f"\n>>> 开始合并新问题和原始问题...")
        
        # 1. 处理新问题：标签映射，收集要删除的原始问题 qid
        remove_qid = set()
        processed_new_qa = []
        
        for new_q in refined_qa_list:
            # 跳过包含 "session" 的问题
            if "session" in new_q.get("question", ""):
                continue
            
            # 标签映射
            if new_q.get("label") == "memory_update":
                new_q["label"] = "记忆更新"
                print(f"  [记忆更新] {new_q.get('question', '')[:50]}...")
                # 收集要删除的原始问题
                for qid in new_q.get("original_qa_qid", []):
                    remove_qid.add(qid)
            elif new_q.get("label") == "integrated_logic":
                new_q["label"] = "事实提取（多对话）"
                # 收集要删除的原始问题
                for qid in new_q.get("original_qa_qid", []):
                    remove_qid.add(qid)
            
            processed_new_qa.append(new_q)
        
        # Processing stats removed - reports only at start
        
        # 2. 收集所有保留的原始问题（深拷贝）
        import copy
        all_qa = []
        for original_q in self.raw_data:
            # 跳过包含 "session" 的问题
            if "session" in original_q.get("question", ""):
                continue
            
            # 如果 qid 在删除列表中，跳过
            if original_q.get("qid") in remove_qid:
                continue
            
            # 深拷贝以避免修改原始数据
            q_copy = copy.deepcopy(original_q)
            # 移除临时标记
            q_copy.pop("episode_index", None)
            q_copy.pop("source_index", None)
            all_qa.append(q_copy)
        
        # Processing stats removed - reports only at start
        
        # 3. 合并新问题和原始问题
        all_qa.extend(processed_new_qa)
        
        # 4. 重新分配 qid
        for idx, q in enumerate(all_qa, start=1):
            q["qid"] = idx
        
        # 5. 合并所有 conversation
        merged_conversation = OrderedDict()
        global_session_idx = 1
        
        # 分别处理 speaker 和 session
        speakers = {}  # 用于去重 speaker
        sessions = []  # 收集所有 session
        
        for item in self.original_data:
            conversation = item.get("conversation", {})
            for key, content in conversation.items():
                if key.startswith("speaker_"):
                    # speaker 需要去重
                    if key not in speakers:
                        speakers[key] = content
                elif key.startswith("session_"):
                    # session 直接收集
                    sessions.append(content)
        
        # 先添加去重后的 speakers
        merged_conversation.update(speakers)
        
        # 再添加重新编号的 sessions
        for session_content in sessions:
            new_session_key = f"session_{global_session_idx}"
            merged_conversation[new_session_key] = session_content
            global_session_idx += 1
        
        # Merge stats removed - reports only at start
        
        # 6. 构建最终输出结构
        final_data = [
            {
                "qa": all_qa,
                "conversation": merged_conversation
            }
        ]
        
        # 7. 保存文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        
        # Final completion stats removed - reports only at start


def new_qa_main(input_file_path: str, output_file_path: str,
                api_key: str = DEFAULT_API_KEY, base_url: str = DEFAULT_BASE_URL,
                model: str = DEFAULT_MODEL,
                embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD) -> str:
    """
    问答精炼重构主函数
    
    Args:
        input_file_path: 输入文件路径（v3版本）
        output_file_path: 输出文件路径（v4版本）
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称
    
    Returns:
        处理后的文件路径
    """
    print("\n" + "="*60)
    print("🔄 步骤 4: 问答精炼重构（语义聚类和逻辑增强）")
    print("="*60)
    print(f"📥 输入文件: {input_file_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 计算待处理问题数
    with open(input_file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    total_questions = sum(len(item.get("qa", [])) for item in input_data)
    print(f"--- 步骤 4 开始处理：共 {total_questions} 个问题 ---")
    
    global client, embedder, MODEL, BASE_URL, API_KEY, CLUSTERING_THRESHOLD
    API_KEY = api_key
    BASE_URL = base_url
    MODEL = model
    CLUSTERING_THRESHOLD = clustering_threshold
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    embedder = SentenceTransformer(embedding_model_name)

    processor = UltimateMemoryRefiner(
        input_file=input_file_path, 
        output_file=output_file_path
    )
    processor.process()
    
    return output_file_path


if __name__ == "__main__":
    # 示例用法
    processor = UltimateMemoryRefiner(
        input_file="./An-Enemy-of-the-People_merged.json", 
        output_file="An-Enemy-of-the-People_new2.json"
    )
    processor.process()


