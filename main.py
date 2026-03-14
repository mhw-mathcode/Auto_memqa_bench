#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Memory Dataset 处理流水线主入口
支持通过配置文件管理所有参数

流程说明:
  步骤 0: v0 生成原始问答对
  步骤 1: v0 → v1 题目合理性检测
  步骤 2: v1 → v2 题目标注
  步骤 3: v2 → v3 new_qa (问答精炼重构)
  步骤 4: v3 → v4 题目乱序 (污染检查)
  步骤 5: v4 → final 生成最终版本
"""

import os
import sys
import json
import argparse
import time
import shutil
from pathlib import Path
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


def run_pipeline(dataset_name: str, start_step: int = 1, end_step: int = 5):
    """
    运行完整流水线
    
    Args:
        dataset_name: 数据集名称（不含扩展名）
        start_step: 起始步骤（0-5）
        end_step: 结束步骤（0-5）
    """
    print("\n" + "="*60)
    print(f"开始处理数据集: {dataset_name}")
    print(f"执行步骤: {start_step} -> {end_step}")
    print("="*60)
    print("\n流程说明:")
    print("  步骤 0: v0 生成原始问答对")
    print("  步骤 1: v0 → v1 题目合理性检测")
    print("  步骤 2: v1 → v2 题目标注")
    print("  步骤 3: v2 → v3 new_qa (问答精炼重构)")
    print("  步骤 4: v3 → v4 题目乱序 (污染检查)")
    print("  步骤 5: v4 → final 生成最终版本")
    print("="*60 + "\n")
    
    # 记录总体开始时间和各步骤耗时
    pipeline_start_time = time.time()
    step_times = {}
    
    config_loader = get_config()
    pipeline_cfg = config_loader.get_pipeline_config()
    
    input_dir = pipeline_cfg.get('input_dir', 'dataset')
    temp_dir = pipeline_cfg.get('temp_dir', 'temp')
    output_dir = pipeline_cfg.get('output_dir', 'result')
    
    input_dir = pipeline_cfg.get('input_dir', 'dataset')
    temp_dir = pipeline_cfg.get('temp_dir', 'temp')
    output_dir = pipeline_cfg.get('output_dir', 'result')
    
    # 确保目录存在
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件路径（新流程）
    # v0: 原始问答对（步骤 0）
    # v1: 题目合理性检测（步骤 1: evidence_check）
    # v2: 题目标注（步骤 2: label）
    # v3: new_qa（步骤 3: new_qa）
    # v4: 题目乱序（步骤 4: pollution_check）
    # final: 最终版本（步骤 5）
    v0_path = os.path.join(temp_dir, f"{dataset_name}_v0.json")
    v1_path = os.path.join(temp_dir, f"{dataset_name}_v1.json")  # 合理性检测
    v2_path = os.path.join(temp_dir, f"{dataset_name}_v2.json")  # 题目标注
    v3_path = os.path.join(temp_dir, f"{dataset_name}_v3.json")  # new_qa
    v4_path = os.path.join(temp_dir, f"{dataset_name}_v4.json")  # 题目乱序
    final_path = os.path.join(output_dir, f"{dataset_name}_final.json")
    
    # 步骤 0: 生成原始问答对
    if start_step <= 0 <= end_step:
        step_start = time.time()
        print("\n[步骤 0] 生成原始问答对...")
        from src.qa_generate import generate_v0
        
        step0_llm = config_loader.get_step_llm("step_0_generate_qa")
        v0_path = generate_v0(dataset_name, input_dir, v0_path, step0_llm)
        if not v0_path:
            return
        step_times['步骤 0: 生成原始问答对'] = time.time() - step_start
        print(f"✓ 步骤 0 完成，耗时: {step_times['步骤 0: 生成原始问答对']:.2f} 秒")
        
        # 在 v0 上打乱选项顺序（不生成新文件，直接修改 v0）
        print("\n[步骤 0 后处理] 打乱 v0 选项顺序...")
        try:
            from src.pollution_check import rename_and_shuffle_options
            rename_and_shuffle_options(v0_path, v0_path)
        except Exception as e:
            print(f"❌ 打乱选项时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 如果 v0 不存在，尝试从 input_dir 读取原始文件
        if not os.path.exists(v0_path):
            source_path = os.path.join(input_dir, dataset_name, f"{dataset_name}_1.json")
            if os.path.exists(source_path):
                import shutil
                shutil.copy(source_path, v0_path)
                print(f"✓ 从源文件复制到 v0: {source_path}")
                
                # 打乱 v0 选项顺序
                print("\n[后处理] 打乱 v0 选项顺序...")
                try:
                    from src.pollution_check import rename_and_shuffle_options
                    rename_and_shuffle_options(v0_path, v0_path)
                except Exception as e:
                    print(f"❌ 打乱选项时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ 错误: 找不到源文件 {source_path}")
                return

    # 步骤 1: 题目合理性检测
    if start_step <= 1 <= end_step:
        step_start = time.time()
        print("\n[步骤 1] 题目合理性检测...")
        from src.evidence_check import evidence_check_main
        from argparse import Namespace
        
        if not os.path.exists(v0_path):
            print(f"❌ 错误: 输入文件不存在 {v0_path}")
            return
        
        # 获取步骤 1 配置
        step1_llm = config_loader.get_step_llm("step_1_full_context")
        
        # 创建简化的 args 对象
        args = Namespace(
            answer_llm_model=step1_llm.model,
            answer_llm_base_url=step1_llm.base_url,
            answer_llm_api_key=step1_llm.api_key,
            max_workers=pipeline_cfg.get('max_workers', 4)
        )
        
        output_path, kept = evidence_check_main(args, v0_path, v1_path)
        # 更新 v2 的输入路径：使用步骤 1 的输出（v1）
        v2_path = output_path.replace("_v1.json", "_v2.json")
        step_times['步骤 1: 题目合理性检测'] = time.time() - step_start
        print(f"✓ 步骤 1 完成: {output_path}，耗时: {step_times['步骤 1: 题目合理性检测']:.2f} 秒")
    
    # 步骤 2: 题目标注
    if start_step <= 2 <= end_step:
        step_start = time.time()
        print("\n[步骤 2] 题目标注...")
        from src.label import label_main
        
        # 步骤 2 的输入是步骤 1 的输出
        step1_output_path = output_path if 'output_path' in locals() else v1_path
        
        if not os.path.exists(step1_output_path):
            print(f"❌ 错误: 输入文件不存在 {step1_output_path}")
            return

        step2_llm = config_loader.get_step_llm("step_2_label")

        label_main(
            step1_output_path,
            v2_path,
            api_key=step2_llm.api_key,
            base_url=step2_llm.base_url,
            model_name=step2_llm.model
        )
        step_times['步骤 2: 题目标注'] = time.time() - step_start
        print(f"✓ 步骤 2 完成: {v2_path}，耗时: {step_times['步骤 2: 题目标注']:.2f} 秒")
    
    # 步骤 3: new_qa (问答精炼重构)
    if start_step <= 3 <= end_step:
        step_start = time.time()
        print("\n[步骤 3] new_qa (问答精炼重构)...")
        from src.new_qa import new_qa_main
        
        if not os.path.exists(v2_path):
            print(f"❌ 错误: 输入文件不存在 {v2_path}")
            return

        step3_llm = config_loader.get_step_llm("step_3_new_qa")

        new_qa_main(
            v2_path,
            v3_path,
            api_key=step3_llm.api_key,
            base_url=step3_llm.base_url,
            model=step3_llm.model
        )
        step_times['步骤 3: new_qa'] = time.time() - step_start
        print(f"✓ 步骤 3 完成: {v3_path}，耗时: {step_times['步骤 3: new_qa']:.2f} 秒")
    
    # 步骤 4: 题目乱序 (污染检查)
    if start_step <= 4 <= end_step:
        step_start = time.time()
        print("\n[步骤 4] 题目乱序 (污染检查)...")
        from src.pollution_check import pollution_check_main
        from argparse import Namespace

        step4_llm = config_loader.get_step_llm("step_4_pollution_check")
        enable_contamination_check = config_loader.get_step_flag(
            "step_4_pollution_check",
            "enable_contamination_check",
            False
        )
        cleanup_temp_files = config_loader.get_step_flag(
            "step_4_pollution_check",
            "cleanup_temp_files",
            True
        )

        # 创建简化的 args 对象
        args = Namespace(
            answer_llm_model=step4_llm.model,
            answer_llm_base_url=step4_llm.base_url,
            answer_llm_api_key=step4_llm.api_key,
            max_workers=pipeline_cfg.get('max_workers', 4)
        )

        pollution_check_main(
            args,
            v3_path,
            v4_path,
            enable_contamination_check=enable_contamination_check,
            cleanup_temp_files=cleanup_temp_files
        )
        step_times['步骤 4: 题目乱序'] = time.time() - step_start
        print(f"✓ 步骤 4 完成: {v4_path}，耗时: {step_times['步骤 4: 题目乱序']:.2f} 秒")
    
    # 步骤 5: 生成最终版本
    if start_step <= 5 <= end_step:
        step_start = time.time()
        print("\n[步骤 5] 生成最终版本...")
        import shutil
        
        if not os.path.exists(v4_path):
            print(f"❌ 错误: 输入文件不存在 {v4_path}")
            return
        
        shutil.copy(v4_path, final_path)
        step_times['步骤 5: 生成最终版本'] = time.time() - step_start
        print(f"✓ 步骤 5 完成: {final_path}，耗时: {step_times['步骤 5: 生成最终版本']:.2f} 秒")
    
    # 计算总耗时
    total_time = time.time() - pipeline_start_time
    
    print("\n" + "="*60)
    print("✅ 流水线执行完成")
    print("="*60)
    
    # 输出时间统计
    print("\n⏱️  时间统计:")
    print("-" * 60)
    for step_name, step_time in step_times.items():
        minutes = int(step_time // 60)
        seconds = step_time % 60
        print(f"  {step_name}: {minutes}分{seconds:.2f}秒 ({step_time:.2f}秒)")
    print("-" * 60)
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    print(f"  总耗时: {total_minutes}分{total_seconds:.2f}秒 ({total_time:.2f}秒)")
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
  python main.py --run An-Enemy-of-the-People --start 2 --end 5
  
  # 只运行特定步骤
  python main.py --run An-Enemy-of-the-People --start 3 --end 3

  # 从步骤 0 开始完整生成
  python main.py --run An-Enemy-of-the-People --start 0 --end 5
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
        choices=[0, 1, 2, 3, 4, 5],
        help='起始步骤 (0-5)，默认为 0'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=5,
        choices=[0, 1, 2, 3, 4, 5],
        help='结束步骤 (0-5)，默认为 5'
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

# python main.py --run An-Enemy-of-the-People > pipeline2.log 2>&1
# python main.py --run the-man-from-earth-script > pipeline.log 2>&1
# python main.py --run 12_Angry_Men > pipeline3.log 2>&1
