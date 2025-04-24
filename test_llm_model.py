#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
from ai_server.utils.config import Config
from ai_server.nlp.llm_model import LLMModel


def test_llm_model(config_path):
    """
    测试大语言模型
    """
    try:
        print("加载配置...")
        config = Config(config_path)

        print("初始化模型...")
        model = LLMModel(config)

        # 测试问答
        test_questions = [
            "你能解释一下什么是函数吗？",
            "一次函数和二次函数有什么区别？",
            "请给我举一个牛顿第二定律的例子"
        ]

        for question in test_questions:
            print("\n问题: {}".format(question))

            # 构造提示
            prompt = "学生: {}\nNAO助教: ".format(question)

            # 计时
            start_time = time.time()

            # 生成回答
            response = model.generate(prompt)

            # 计算耗时
            elapsed_time = time.time() - start_time

            print("回答: {}".format(response))
            print("生成耗时: {:.2f}秒".format(elapsed_time))

        print("\n测试完成!")

    except Exception as e:
        print("测试失败: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大语言模型测试")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")

    args = parser.parse_args()
    test_llm_model(args.config)