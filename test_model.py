#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试LLM模型加载和生成功能
自动检测并适应项目结构
"""

import importlib.util
import os
import sys
import time

# 输出当前工作目录
print(f"当前工作目录: {os.getcwd()}")


def find_module(module_name, possible_paths):
    """查找模块的位置"""
    for path in possible_paths:
        full_path = os.path.join(os.getcwd(), path)
        if os.path.exists(full_path):
            print(f"找到模块: {full_path}")
            module_dir = os.path.dirname(full_path)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            return full_path
    return None


# 查找Config类
config_paths = [
    'utils/config.py',
    'ai_server/utils/config.py',
]
config_path = find_module('config', config_paths)

# 查找LLMModel类
llm_paths = [
    'ai_server/nlp/llm_model.py',
    'nlp/llm_model.py',
]
llm_path = find_module('llm_model', llm_paths)

# 查找logger
logger_paths = [
    'logger.py',
    'ai_server/logger.py',
    'utils/logger.py',
]
logger_path = find_module('logger', logger_paths)


# 动态导入模块
def import_from_path(module_name, file_path):
    """从文件路径动态导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if not config_path or not llm_path or not logger_path:
    print("错误: 无法找到必要的模块，请检查项目结构")
    sys.exit(1)

# 动态导入模块
config_module = import_from_path('config', config_path)
llm_module = import_from_path('llm_model', llm_path)
logger_module = import_from_path('logger', logger_path)

# 获取类和函数
Config = getattr(config_module, 'Config')
LLMModel = getattr(llm_module, 'LLMModel')
setup_logger = getattr(logger_module, 'setup_logger')

# 设置日志
logger = setup_logger('test_llm')


def test_llm():
    print("=" * 50)
    print("测试LLM模型加载和生成功能")
    print("=" * 50)

    try:
        # 加载配置
        print("\n1. 加载配置...")
        config = Config()
        print(f"模型名称: {config.get('llm.model_name')}")
        print(f"模型路径: {config.get('llm.model_path')}")
        print(f"使用LoRA: {config.get('llm.use_lora', False)}")

        # 初始化LLM模型
        print("\n2. 初始化LLM模型...")
        start_time = time.time()
        llm = LLMModel(config)
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")

        # 测试生成功能
        print("\n3. 测试生成功能...")
        test_prompts = [
            "你好，请简单介绍一下自己",
            "什么是编译原理？",
            "学生: 我对计算机编程感兴趣，从哪里开始学习呢？\nNAO助教:"
        ]

        for i, prompt in enumerate(test_prompts):
            print(f"\n测试提示 {i + 1}:")
            print(f"提示: {prompt}")

            gen_start = time.time()
            response = llm.generate(prompt)
            gen_time = time.time() - gen_start

            print(f"响应:")
            print(f"{response}")
            print(f"生成耗时: {gen_time:.2f}秒")

        print("\n测试完成！模型加载和生成功能正常。")

    except Exception as e:
        print(f"\n测试时出错: {e}")
        logger.error(f"测试LLM时出错: {e}", exc_info=True)


if __name__ == "__main__":
    test_llm()