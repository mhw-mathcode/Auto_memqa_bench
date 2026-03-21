#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Personal Memory Dataset 处理流水线主入口
支持通过配置文件管理所有参数

流程说明:
  步骤 0: v0 生成原始问答对
    步骤 1: v0 → v1a → v1b 题目合理性检测
    步骤 2: v1b → v2 题目标注
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
import tempfile
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
    print("  步骤 1: v0 → v1a → v1b 题目合理性检测")
    print("  步骤 2: v1b → v2 题目标注")
    print("  步骤 3: v2 → v3 new_qa (问答精炼重构)")
    print("  步骤 4: v3 → v4 题目乱序 (污染检查)")
    print("  步骤 5: v4 → final 生成最终版本")
    print("="*60 + "\n")
    
    # 记录总体开始时间和各步骤耗时
    pipeline_start_time = time.time()
    step_times = {}
    
    config_loader = get_config()
    pipeline_cfg = config_loader.get_pipeline_config()

    def should_run_step(step_idx: int, step_key: str) -> bool:
        """步骤执行条件：在 start/end 范围内且未配置 skip。"""
        if not (start_step <= step_idx <= end_step):
            return False
        try:
            is_skip = bool(config_loader.get_step_flag(step_key, "skip", False))
        except ValueError:
            is_skip = False
        if is_skip:
            print(f"\n[步骤 {step_idx}] 配置为 skip=true，跳过执行")
            return False
        return True
    
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
    # v1a: 只使用证据检查（步骤 1a）
    # v1b: 迭代删除证据检查（步骤 1b）
    # v2: 题目标注（步骤 2，输入 v1b）
    # v3: new_qa（步骤 3: new_qa）
    # v4: 题目乱序（步骤 4: pollution_check）
    # final: 最终版本（步骤 5）
    v0_path = os.path.join(temp_dir, f"{dataset_name}_v0.json")
    v1a_path = os.path.join(temp_dir, f"{dataset_name}_v1a.json")  # 合理性检测阶段A
    v1b_path = os.path.join(temp_dir, f"{dataset_name}_v1b.json")  # 合理性检测阶段B
    legacy_v1_path = os.path.join(temp_dir, f"{dataset_name}_v1.json")  # 兼容旧版
    v2_path = os.path.join(temp_dir, f"{dataset_name}_v2.json")  # 题目标注
    v3_path = os.path.join(temp_dir, f"{dataset_name}_v3.json")  # new_qa
    v4_path = os.path.join(temp_dir, f"{dataset_name}_v4.json")  # 题目乱序
    final_path = os.path.join(output_dir, f"{dataset_name}_final.json")

    def resolve_step_input(step_label, preferred_path, fallback_paths):
        """按顺序选择存在的输入文件，支持步骤间自动回退。"""
        checked = []
        ordered_candidates = [preferred_path] + list(fallback_paths)

        for candidate in ordered_candidates:
            if not candidate or candidate in checked:
                continue
            checked.append(candidate)
            if os.path.exists(candidate):
                if candidate != preferred_path:
                    print(
                        f"⚠️ {step_label}: 输入 {os.path.basename(preferred_path)} 不存在，"
                        f"回退使用 {os.path.basename(candidate)}"
                    )
                return candidate

        checked_text = ", ".join(os.path.basename(path) for path in checked)
        print(f"❌ 错误: {step_label} 未找到可用输入文件，已检查: {checked_text}")
        return None

    def _passes_v1a_rule(qa_item):
        only_check = qa_item.get("only_evidence_check")
        if isinstance(only_check, dict):
            return only_check.get("result") == "right"
        # 缺失字段时不强制删除，保持向后兼容。
        return True

    def _passes_v1b_rule(qa_item):
        """v1b 删题规则：仅当存在 round==3 且其 result != wrong 才删除。"""
        # v3 生成的新题没有稳定的 iterative_evidence_ablation 语义，直接豁免。
        if qa_item.get("is_generated_qa") is True:
            return True

        if "iterative_evidence_ablation" not in qa_item:
            return True

        ablation = qa_item.get("iterative_evidence_ablation")
        if not isinstance(ablation, list):
            return True

        for record in ablation:
            if not isinstance(record, dict):
                continue
            if record.get("round") == 3:
                return record.get("result") == "wrong"

        # 未找到 round == 3 则保留
        return True

    def _passes_v4_rule(qa_item):
        """v4 删题规则：仅当存在 pollution_check.result 且其不为 good 才删除。
        
        豁免：由 v3 生成的新题（is_generated_qa=true）不受 v4 规则影响。
        """
        # v3 生成的题目豁免 v4 规则
        if qa_item.get("is_generated_qa") is True:
            return True

        if "pollution_check" not in qa_item:
            return True

        pollution_check = qa_item.get("pollution_check")
        if not isinstance(pollution_check, dict):
            return True

        if "result" not in pollution_check:
            return True

        return pollution_check.get("result") == "good"

    def apply_cumulative_rules(input_path, rule_names, stage_label, output_path=None):
        """对输入文件应用累积删题规则；仅在指定 output_path 时落盘。"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        rule_checkers = {
            "v1a": _passes_v1a_rule,
            "v1b": _passes_v1b_rule,
            "v4": _passes_v4_rule,
        }

        total_before = 0
        total_after = 0
        removed_by_rule = {rule: 0 for rule in rule_names}

        for section in data:
            qa_list = section.get("qa", [])
            if not isinstance(qa_list, list):
                continue

            filtered = []
            for qa_item in qa_list:
                total_before += 1
                removed_rule = None

                for rule in rule_names:
                    checker = rule_checkers.get(rule)
                    if checker and not checker(qa_item):
                        removed_rule = rule
                        break

                if removed_rule:
                    removed_by_rule[removed_rule] += 1
                    continue

                filtered.append(qa_item)
                total_after += 1

            section["qa"] = filtered

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

        removed_total = total_before - total_after
        print(
            f"  [{stage_label}] 规则过滤: {total_before} -> {total_after}，"
            f"移除 {removed_total}"
        )
        for rule in rule_names:
            print(f"    - {rule} 规则移除: {removed_by_rule.get(rule, 0)}")

        return data

    def run_with_temp_filtered_input(input_path, rule_names, stage_label, runner):
        """在临时文件中传递过滤结果，执行后立即删除，避免持久化中间文件。"""
        filtered_data = apply_cumulative_rules(
            input_path,
            rule_names,
            stage_label
        )

        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                prefix=f"{dataset_name}_{stage_label}_",
                encoding="utf-8",
                delete=False,
            ) as temp_file:
                json.dump(filtered_data, temp_file, indent=4, ensure_ascii=False)
                temp_file_path = temp_file.name

            return runner(temp_file_path)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    # 步骤 0: 生成原始问答对
    if should_run_step(0, "step_0_generate_qa"):
        step_start = time.time()
        print("\n[步骤 0] 生成原始问答对...")
        from src.qa_generate import generate_v0
        
        step0_llm = config_loader.get_step_llm("step_0_generate_qa")
        step0_force_generate_new_qa = config_loader.get_step_flag(
            "step_0_generate_qa",
            "force_generate_new_qa",
            False,
        )
        step0_batch_size = config_loader.get_step_flag(
            "step_0_generate_qa",
            "speaker_batch_size",
            8,
        )
        v0_path = generate_v0(
            dataset_name,
            input_dir,
            v0_path,
            step0_llm,
            force_generate_new_qa=bool(step0_force_generate_new_qa),
            initial_batch_size=int(step0_batch_size),
        )
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
    if should_run_step(1, "step_1_full_context"):
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
        
        output_path, kept = evidence_check_main(args, v0_path, v1b_path)
        step_times['步骤 1: 题目合理性检测'] = time.time() - step_start
        print(f"✓ 步骤 1 完成: {output_path}，耗时: {step_times['步骤 1: 题目合理性检测']:.2f} 秒")
    
    # 步骤 2: 题目标注
    if should_run_step(2, "step_2_label"):
        step_start = time.time()
        print("\n[步骤 2] 题目标注...")
        from src.label import label_main
        
        # 步骤 2 的输入是步骤 1 的输出；若缺失则向前回退。
        preferred_step2_input = output_path if 'output_path' in locals() else v1b_path
        step2_input_path = resolve_step_input(
            "步骤 2",
            preferred_step2_input,
            [v1b_path, legacy_v1_path, v1a_path, v0_path],
        )

        if not step2_input_path:
            return

        step2_llm = config_loader.get_step_llm("step_2_label")

        run_with_temp_filtered_input(
            step2_input_path,
            ["v1a", "v1b"],
            "step2",
            lambda filtered_path: label_main(
                filtered_path,
                v2_path,
                api_key=step2_llm.api_key,
                base_url=step2_llm.base_url,
                model_name=step2_llm.model
            ),
        )
        step_times['步骤 2: 题目标注'] = time.time() - step_start
        print(f"✓ 步骤 2 完成: {v2_path}，耗时: {step_times['步骤 2: 题目标注']:.2f} 秒")
    
    # 步骤 3: new_qa (问答精炼重构)
    if should_run_step(3, "step_3_new_qa"):
        step_start = time.time()
        print("\n[步骤 3] new_qa (问答精炼重构)...")
        from src.new_qa import new_qa_main

        step3_input_path = resolve_step_input(
            "步骤 3",
            v2_path,
            [v1b_path, legacy_v1_path, v1a_path, v0_path],
        )
        if not step3_input_path:
            return

        step3_llm = config_loader.get_step_llm("step_3_new_qa")

        run_with_temp_filtered_input(
            step3_input_path,
            ["v1a", "v1b"],
            "step3",
            lambda filtered_path: new_qa_main(
                filtered_path,
                v3_path,
                api_key=step3_llm.api_key,
                base_url=step3_llm.base_url,
                model=step3_llm.model
            ),
        )
        step_times['步骤 3: new_qa'] = time.time() - step_start
        print(f"✓ 步骤 3 完成: {v3_path}，耗时: {step_times['步骤 3: new_qa']:.2f} 秒")
    
    # 步骤 4: 题目乱序 (污染检查)
    if should_run_step(4, "step_4_pollution_check"):
        step_start = time.time()
        print("\n[步骤 4] 题目乱序 (污染性检查)...")
        from src.pollution_check import pollution_check_main
        from argparse import Namespace

        step4_input_path = resolve_step_input(
            "步骤 4",
            v3_path,
            [v2_path, v1b_path, legacy_v1_path, v1a_path, v0_path],
        )
        if not step4_input_path:
            return

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

        run_with_temp_filtered_input(
            step4_input_path,
            ["v1a", "v1b"],
            "step4",
            lambda filtered_path: pollution_check_main(
                args,
                filtered_path,
                v4_path,
                enable_contamination_check=enable_contamination_check,
                cleanup_temp_files=cleanup_temp_files
            ),
        )
        step_times['步骤 4: 题目乱序 (污染性检查)'] = time.time() - step_start
        print(f"✓ 步骤 4 完成: {v4_path}，耗时: {step_times['步骤 4: 题目乱序 (污染性检查)']:.2f} 秒")
    
    # 步骤 5: 生成最终版本
    if should_run_step(5, "step_5_finalize"):
        step_start = time.time()
        print("\n[步骤 5] 生成最终版本（过滤污染题目）...")

        step5_input_path = resolve_step_input(
            "步骤 5",
            v4_path,
            [v3_path, v2_path, v1b_path, legacy_v1_path, v1a_path, v0_path],
        )
        if not step5_input_path:
            return

        apply_cumulative_rules(
            step5_input_path,
            ["v1a", "v1b", "v4"],
            "步骤 5",
            output_path=final_path,
        )

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

