#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Memory Dataset 处理流水线主入口
支持通过配置文件管理所有参数
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
from config import get_config, VersionManager, PipelineConfig

def show_config():
    """显示当前配置"""
    config_loader = get_config()
    config_loader.print_config_summary()


def show_versions():
    """显示版本信息"""
    config_loader = get_config()
    pipeline_cfg = config_loader.get_pipeline_config()
    
    # 创建一个临时的 PipelineConfig 对象
    from config import LLMConfig
    temp_cfg = PipelineConfig(
        input_dir=pipeline_cfg.get('input_dir', 'dataset'),
        output_dir=pipeline_cfg.get('output_dir', 'result'),
        temp_dir=pipeline_cfg.get('temp_dir', 'temp'),
    )
    
    version_manager = VersionManager(temp_cfg)
    version_manager.print_version_info()


def run_pipeline(dataset_name: str, start_step: int = 1, end_step: int = 4):
    """
    运行完整流水线
    
    Args:
        dataset_name: 数据集名称（不含扩展名）
        start_step: 起始步骤（1-4）
        end_step: 结束步骤（1-4）
    """
    print("\n" + "="*60)
    print(f"开始处理数据集: {dataset_name}")
    print(f"执行步骤: {start_step} -> {end_step}")
    print("="*60 + "\n")
    
    config_loader = get_config()
    pipeline_cfg = config_loader.get_pipeline_config()
    
    input_dir = pipeline_cfg.get('input_dir', 'dataset')
    temp_dir = pipeline_cfg.get('temp_dir', 'temp')
    output_dir = pipeline_cfg.get('output_dir', 'result')
    
    # 确保目录存在
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件路径
    v0_path = os.path.join(temp_dir, f"{dataset_name}_v0.json")
    v1_path = os.path.join(temp_dir, f"{dataset_name}_v1.json")
    v2_path = os.path.join(temp_dir, f"{dataset_name}_v2.json")
    v3_path = os.path.join(temp_dir, f"{dataset_name}_v3.json")
    final_path = os.path.join(output_dir, f"{dataset_name}_final.json")
    
    # 步骤 0: 生成原始问答对
    if start_step <= 0 <= end_step:
        from src.qa_generate import generate_v0
        
        step0_llm = config_loader.get_step_llm("step_0_generate_qa")
        v0_path = generate_v0(dataset_name, input_dir, v0_path, step0_llm)
        if not v0_path:
            return
    else:
        # 如果 v0 不存在，尝试从 input_dir 读取原始文件
        if not os.path.exists(v0_path):
            source_path = os.path.join(input_dir, dataset_name, f"{dataset_name}_1.json")
            if os.path.exists(source_path):
                import shutil
                shutil.copy(source_path, v0_path)
                print(f"✓ 从源文件复制到 v0: {source_path}")
            else:
                print(f"❌ 错误: 找不到源文件 {source_path}")
                return

    # 步骤 1: 选项打乱和污染检查
    if start_step <= 1 <= end_step:
        from src.pollution_check import pollution_check_main
        from argparse import Namespace

        step1_llm = config_loader.get_step_llm("step_1_pollution_check")
        enable_contamination_check = config_loader.get_step_flag(
            "step_1_pollution_check",
            "enable_contamination_check",
            False
        )

        # 创建简化的 args 对象
        args = Namespace(
            answer_llm_model=step1_llm.model,
            answer_llm_base_url=step1_llm.base_url,
            answer_llm_api_key=step1_llm.api_key,
            max_workers=pipeline_cfg.get('max_workers', 4)
        )

        pollution_check_main(
            args,
            v0_path,
            v1_path,
            enable_contamination_check=enable_contamination_check
        )
        print(f"✓ 步骤 1 完成: {v1_path}")
    
    # 步骤 2: 题目合理性验证
    if start_step <= 2 <= end_step:
        from src.full_context import full_context_main
        from argparse import Namespace
        
        if not os.path.exists(v1_path):
            print(f"❌ 错误: 输入文件不存在 {v1_path}")
            return
        
        # 获取步骤 2 配置
        step2_llm = config_loader.get_step_llm("step_2_full_context")
        
        # 创建简化的 args 对象
        args = Namespace(
            answer_llm_model=step2_llm.model,
            answer_llm_base_url=step2_llm.base_url,
            answer_llm_api_key=step2_llm.api_key,
            max_workers=pipeline_cfg.get('max_workers', 4)
        )
        
        full_context_main(args, v1_path, v2_path)
        print(f"✓ 步骤 2 完成: {v2_path}")
    
    # 步骤 3: 问题标注
    if start_step <= 3 <= end_step:
        from src.label import label_main
        
        if not os.path.exists(v2_path):
            print(f"❌ 错误: 输入文件不存在 {v2_path}")
            return

        step3_llm = config_loader.get_step_llm("step_3_label")

        label_main(
            v2_path,
            v3_path,
            api_key=step3_llm.api_key,
            base_url=step3_llm.base_url,
            model_name=step3_llm.model
        )
        print(f"✓ 步骤 3 完成: {v3_path}")
    
    # 步骤 4: 问答精炼重构
    if start_step <= 4 <= end_step:
        from src.new_qa import new_qa_main
        
        if not os.path.exists(v3_path):
            print(f"❌ 错误: 输入文件不存在 {v3_path}")
            return

        step4_llm = config_loader.get_step_llm("step_4_new_qa")
        embedding_cfg = config_loader.get_embedding_config()

        new_qa_main(
            v3_path,
            final_path,
            api_key=step4_llm.api_key,
            base_url=step4_llm.base_url,
            model=step4_llm.model,
            embedding_model_name=embedding_cfg.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            clustering_threshold=embedding_cfg.get("clustering_threshold", 0.55)
        )
        print(f"✓ 步骤 4 完成: {final_path}")
    
    print("\n" + "="*60)
    print("✅ 流水线执行完成")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Personal Memory Dataset 处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 显示配置信息
  python main.py --show-config
  
  # 显示版本信息
  python main.py --show-versions
  
  # 运行完整流水线
  python main.py --run An-Enemy-of-the-People
  
    # 从指定步骤开始运行
    python main.py --run An-Enemy-of-the-People --start 2 --end 4
  
    # 只运行特定步骤
    python main.py --run An-Enemy-of-the-People --start 3 --end 3

    # 从步骤 0 开始完整生成
    python main.py --run An-Enemy-of-the-People --start 0 --end 4
        """
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='显示当前配置信息'
    )
    
    parser.add_argument(
        '--show-versions',
        action='store_true',
        help='显示版本信息'
    )
    
    parser.add_argument(
        '--run',
        type=str,
        metavar='DATASET',
        help='运行流水线，指定数据集名称'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='起始步骤 (0-4)，默认为 0'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help='结束步骤 (0-4)，默认为 4'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置文件路径，默认为 config.json'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config != 'config.json':
        config_loader = get_config()
        config_loader.load_config(args.config)
    
    # 处理命令
    if args.show_config:
        show_config()
    elif args.show_versions:
        show_versions()
    elif args.run:
        if args.start > args.end:
            print("❌ 错误: 起始步骤不能大于结束步骤")
            sys.exit(1)
        run_pipeline(args.run, args.start, args.end)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# python main.py --run An-Enemy-of-the-People > pipeline.log 2>&1 --start 3
