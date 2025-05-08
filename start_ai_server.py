#!/usr/bin/env python
# -*- coding: utf-8 -*-
import eventlet

eventlet.monkey_patch()

import argparse
import asyncio
import sys
import os

# 导入项目的日志模块
from ai_server.logger import setup_logger

# 设置日志
logger = setup_logger('ai_server_starter')


def start_web_monitor_process(web_host, web_port):
    """在单独进程中启动Web监控服务"""
    try:
        import subprocess

        # 检查web_monitor.py是否存在
        monitor_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_monitor.py")
        if not os.path.exists(monitor_script):
            logger.warning(f"Web监控脚本不存在: {monitor_script}")
            print(f"未找到Web监控脚本，跳过启动Web监控")
            return None

        # 启动独立进程
        web_process = subprocess.Popen([
            sys.executable,
            monitor_script,
            "--host", web_host,
            "--port", str(web_port)
        ])

        logger.info(f"Web监控服务已在独立进程中启动: PID={web_process.pid}")
        print(f"Web监控服务已启动: http://{web_host}:{web_port}/")
        return web_process
    except Exception as e:
        logger.error(f"启动Web监控服务时出错: {e}")
        print(f"启动Web监控服务时出错: {e}")
        return None


async def start_server(args):
    """
    启动AI服务器
    """
    try:
        # 导入需要的组件
        from ai_server.utils.config import Config
        from ai_server.nlp.llm_model import LLMModel
        from ai_server.server import AIWebSocketServer
        from ai_server.nlp.conversation import ConversationManager
        from ai_server.knowledge.knowledge_graph import KnowledgeGraph
        from ai_server.knowledge.recommender import KnowledgeRecommender
        from ai_server.emotion.fusion import EmotionFusion

        logger.info("加载配置...")
        config = Config()

        # 初始化各个组件
        logger.info("初始化知识图谱...")
        print("正在连接知识图谱数据库...")
        kg = KnowledgeGraph(config)

        logger.info("初始化情感融合模块...")
        emotion_fusion = EmotionFusion(config)

        logger.info("初始化知识推荐器...")
        recommender = KnowledgeRecommender(kg, config)

        logger.info("初始化大语言模型...")
        print("正在加载大语言模型，这可能需要几分钟...")
        llm = LLMModel(config)
        print("大语言模型已加载完成")

        logger.info("初始化对话管理器...")
        conversation = ConversationManager(llm)

        # 创建服务器实例
        logger.info(f"创建AI服务器: {args.host}:{args.port}")

        # 创建服务传递所有必要组件
        server = AIWebSocketServer(
            host=args.host,
            port=args.port,
            config=config,
            llm=llm,
            conversation=conversation,
            knowledge_graph=kg,
            recommender=recommender,
            emotion_fusion=emotion_fusion
        )

        # 启动WebSocket服务器
        print(f"正在启动WebSocket服务器: {args.host}:{args.port}")
        server_context = await server.start_server()

        # 使用异步上下文管理器启动服务器
        async with server_context:
            print("服务器运行中，按Ctrl+C退出...")
            # 保持服务器运行
            await asyncio.Future()

    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器...")
        print("接收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}", exc_info=True)
        print(f"启动服务器时出错: {e}")
    finally:
        # 停止服务器
        if 'server' in locals() and hasattr(server, 'stop_server'):
            await server.stop_server()

        # 关闭知识图谱连接
        if 'kg' in locals() and hasattr(kg, 'close'):
            kg.close()

        print("服务器已停止")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NAO机器人智能辅助教学系统 - AI服务器")
    parser.add_argument("--host", type=str, default="localhost", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8765, help="服务器端口号")

    # Web监控相关参数
    parser.add_argument("--web-monitor", action="store_true", help="启动Web监控界面")
    parser.add_argument("--web-host", type=str, default="0.0.0.0", help="Web监控主机地址")
    parser.add_argument("--web-port", type=int, default=5000, help="Web监控端口号")

    args = parser.parse_args()

    print("解析命令行参数完成")
    print(f"主机: {args.host}, 端口: {args.port}")
    print(f"Web监控: {'启用' if args.web_monitor else '禁用'}")

    # 启动Web监控(如果需要)
    web_process = None
    if args.web_monitor:
        print(f"正在启动Web监控: {args.web_host}:{args.web_port}")
        web_process = start_web_monitor_process(args.web_host, args.web_port)

    # 启动AI服务器
    try:
        print("正在启动AI服务器...")
        asyncio.run(start_server(args))
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        # 终止Web监控进程(如果存在)
        if web_process:
            try:
                web_process.terminate()
                print("Web监控服务已停止")
            except:
                pass

        print("程序已退出")