#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试LLM模型加载和生成功能
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_server.utils.config import Config
from ai_server.nlp.llm_model import LLMModel
from ai_server.logger import setup_logger



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