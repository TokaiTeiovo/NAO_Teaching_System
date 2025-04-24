#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import asyncio
from ai_server.server import AIServer


def start_server(args):
    """
    启动AI服务器
    """
    print("启动AI服务器...")
    print("主机: {}".format(args.host))
    print("端口: {}".format(args.port))

    # 创建服务器实例
    server = AIServer(args.config)

    # 启动服务器
    asyncio.run(server.start_server(args.host, args.port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人智能辅助教学系统 - AI服务器")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口号")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")

    args = parser.parse_args()
    start_server(args)