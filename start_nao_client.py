#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from nao_control.main import main as nao_main


def start_client(args):
    """
    启动NAO客户端
    """
    print("启动NAO客户端...")
    print("IP: {}".format(args.ip))
    print("Port: {}".format(args.port))

    # 调用NAO客户端主函数
    nao_main(args.ip, args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人智能辅助教学系统 - 客户端")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="NAO机器人的IP地址")
    parser.add_argument("--port", type=int, default=9559, help="NAO机器人的端口号")

    args = parser.parse_args()
    start_client(args)