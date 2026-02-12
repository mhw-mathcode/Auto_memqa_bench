"""
配置管理模块
统一管理项目中的所有配置信息和版本控制
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """LLM 配置"""
    model: str
    base_url: str
    api_key: str

    def to_dict(self):
        """转换为字典格式"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key
        }


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 输入输出路径
    input_dir: str
    output_dir: str
    temp_dir: str = "temp"
    
    # LLM 配置
    qa_llm: Optional[LLMConfig] = None
    answer_llm: Optional[LLMConfig] = None
    
    # 并发配置
    max_workers: int = 4
    
    # 版本控制
    version_prefix: str = "v"
    
    def __post_init__(self):
        """初始化后创建必要的目录"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_version_path(self, version: str, filename: str) -> str:
        """
        获取特定版本的文件路径
        
        Args:
            version: 版本号（如 "v0", "v1"）
            filename: 文件名
        
        Returns:
            完整的文件路径
        """
        return os.path.join(self.temp_dir, f"{filename}_{version}.json")


class VersionManager:
    """
    版本管理器
    管理数据处理流程的各个版本
    
    版本说明：
    - v0: 原始生成的问答对
    - v1: 打乱选项和污染检查后的数据
    - v2: 全上下文验证后的数据
    - v3: 精炼重构后的问答（new_qa）
    - v4: 标注分类后的最终数据（label）
    """
    
    VERSION_DESCRIPTIONS = {
        "v0": "原始生成的问答对",
        "v1": "打乱选项和污染检查后",
        "v2": "全上下文验证后",
        "v3": "问题标注后",
        "v4": "精炼重构后（最终版本）"
    }
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def get_path(self, version: str, dataset_name: str) -> str:
        """获取指定版本的文件路径"""
        return self.config.get_version_path(version, dataset_name)
    
    def get_description(self, version: str) -> str:
        """获取版本描述"""
        return self.VERSION_DESCRIPTIONS.get(version, "未知版本")
    
    def print_version_info(self):
        """打印所有版本信息"""
        print("\n" + "="*60)
        print("数据处理流程版本信息")
        print("="*60)
        for version, description in self.VERSION_DESCRIPTIONS.items():
            print(f"  {version}: {description}")
        print("="*60 + "\n")


def build_llm_config(model: str, base_url: str, api_key: str) -> LLMConfig:
    """
    构建 LLM 配置
    
    Args:
        model: 模型名称
        base_url: API 基础 URL
        api_key: API 密钥
    
    Returns:
        LLMConfig 对象
    """
    return LLMConfig(model=model, base_url=base_url, api_key=api_key)


class ConfigLoader:
    """
    配置加载器
    从 config.json 文件加载所有配置
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load_config()
    
    def load_config(self, config_path: str = "config.json"):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径，默认为 config.json
        """
        # 如果传入的是相对路径，则相对于项目根目录
        if not os.path.isabs(config_path):
            # 获取当前文件所在目录（config.py 所在目录）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
        
        print(f"✓ 配置文件加载成功: {config_path}")
    
    def get_step_config(self, step_key: str) -> Dict[str, Any]:
        """
        获取指定步骤配置
        
        Args:
            step_key: 步骤名称（如 'step_2_full_context'）
        """
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load_config()")
        
        steps = self._config.get("steps", {})
        if step_key not in steps:
            raise ValueError(f"未找到步骤配置: {step_key}")
        
        return steps[step_key]

    def get_step_llm(self, step_key: str) -> LLMConfig:
        """
        获取指定步骤的 LLM 配置
        """
        step_cfg = self.get_step_config(step_key)
        llm_cfg = step_cfg.get("llm", {})
        return LLMConfig(
            model=llm_cfg.get("model", ""),
            base_url=llm_cfg.get("base_url", ""),
            api_key=llm_cfg.get("api_key", "")
        )

    def get_step_flag(self, step_key: str, flag_name: str, default: Any = None) -> Any:
        """
        获取步骤中的布尔/参数配置
        """
        step_cfg = self.get_step_config(step_key)
        return step_cfg.get(flag_name, default)
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """获取流水线配置"""
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load_config()")
        return self._config.get("pipeline", {})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入模型配置"""
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load_config()")
        return self._config.get("embedding", {})
    
    def get_tool_config(self, tool_key: str) -> Dict[str, Any]:
        """
        获取指定工具配置
        """
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load_config()")
        
        tools = self._config.get("tools", {})
        if tool_key not in tools:
            raise ValueError(f"未找到工具配置: {tool_key}")
        
        return tools[tool_key]

    def get_option_perturbation_models(self) -> tuple:
        """获取选项扰动生成和评分模型配置"""
        tool_cfg = self.get_tool_config("option_perturbation")
        llm_cfg = tool_cfg.get("llm", {})
        return llm_cfg.get("gen_model"), llm_cfg.get("score_model")
    
    def print_config_summary(self):
        """打印配置摘要"""
        if self._config is None:
            print("配置未加载")
            return
        
        print("\n" + "="*60)
        print("当前配置摘要")
        print("="*60)
        
        print("\n【步骤配置】")
        for name, cfg in self._config.get("steps", {}).items():
            desc = cfg.get("description", "")
            api_required = cfg.get("api_required", False)
            model = cfg.get("llm", {}).get("model", "")
            print(f"  {name}: {desc}")
            print(f"    api_required: {api_required}")
            if model:
                print(f"    model: {model}")

        print("\n【工具配置】")
        for name, cfg in self._config.get("tools", {}).items():
            desc = cfg.get("description", "")
            api_required = cfg.get("api_required", False)
            print(f"  {name}: {desc}")
            print(f"    api_required: {api_required}")
        
        print("\n【流水线配置】")
        for key, value in self._config.get("pipeline", {}).items():
            print(f"  {key}: {value}")
        
        print("\n【嵌入模型配置】")
        for key, value in self._config.get("embedding", {}).items():
            print(f"  {key}: {value}")
        
        print("="*60 + "\n")


def get_config() -> ConfigLoader:
    """
    获取配置加载器实例（单例）
    
    Returns:
        ConfigLoader 实例
    """
    return ConfigLoader()

