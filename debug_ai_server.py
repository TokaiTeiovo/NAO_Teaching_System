#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import sys

import websockets

from logger import setup_logger

logger = setup_logger('debug_ai_server')


# 简单的客户端处理函数
async def handle_client(websocket):
    client_id = id(websocket)
    logger.info(f"新客户端连接: {client_id}")
    print(f"新客户端连接: {client_id}")

    try:
        # 发送欢迎消息
        await websocket.send(json.dumps({
            "type": "welcome",
            "id": "welcome_msg",
            "data": {"message": "欢迎连接到AI服务器", "server_time": asyncio.get_event_loop().time()}
        }))

        # 接收和处理消息
        async for message in websocket:
            try:
                # 解析消息
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                msg_id = data.get("id", "")
                content = data.get("data", {})

                logger.info(f"收到消息: 客户端={client_id}, 类型={msg_type}, ID={msg_id}")
                print(f"收到消息: 类型={msg_type}, ID={msg_id}")

                # 简单处理不同类型的消息
                if msg_type == "text":
                    text = content.get("text", "")
                    # 简单的文本响应
                    await websocket.send(json.dumps({
                        "type": "text_result",
                        "id": msg_id,
                        "data": {
                            "text": f"你好！我收到了你的消息: '{text}'",
                            "actions": []
                        }
                    }))
                elif msg_type == "command":
                    command = content.get("command", "")
                    # 简单的命令响应
                    if command == "init_session":
                        session_id = f"session_{int(asyncio.get_event_loop().time())}"
                        await websocket.send(json.dumps({
                            "type": "command_result",
                            "id": msg_id,
                            "data": {"session_id": session_id}
                        }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "command_result",
                            "id": msg_id,
                            "data": {"message": f"收到命令: {command}"}
                        }))
                else:
                    # 默认响应
                    await websocket.send(json.dumps({
                        "type": f"{msg_type}_result",
                        "id": msg_id,
                        "data": {"message": f"收到类型为 {msg_type} 的消息"}
                    }))

            except json.JSONDecodeError:
                logger.error(f"JSON解析错误: {message[:100]}")
                # 发送错误响应
                await websocket.send(json.dumps({
                    "type": "error",
                    "id": "",
                    "data": {"error_type": "无效消息", "message": "无法解析JSON消息"}
                }))
            except Exception as e:
                logger.error(f"处理消息时出错: {e}", exc_info=True)
                # 发送错误响应
                await websocket.send(json.dumps({
                    "type": "error",
                    "id": "",
                    "data": {"error_type": "处理错误", "message": str(e)}
                }))

    except websockets.exceptions.ConnectionClosedError as e:
        logger.info(f"客户端连接已关闭: {client_id}, 原因: {e}")
    except Exception as e:
        logger.error(f"处理客户端时出错: {e}", exc_info=True)
    finally:
        logger.info(f"客户端断开连接: {client_id}")
        print(f"客户端断开连接: {client_id}")


async def start_debug_server():
    """启动调试版AI服务器"""
    host = "localhost"
    port = 8765

    print(f"启动WebSocket服务器: {host}:{port}")

    # 使用websockets 15.0.1的异步上下文管理器方式
    async with websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10 * 1024 * 1024
    ):
        print(f"WebSocket服务器已启动在 {host}:{port}")
        logger.info(f"WebSocket服务器已启动: {host}:{port}")

        try:
            # 保持服务器运行
            print("服务器运行中，按Ctrl+C退出...")
            await asyncio.Future()
        except KeyboardInterrupt:
            print("接收到中断信号，服务器关闭中...")
        except Exception as e:
            print(f"服务器运行时出错: {e}")
            logger.error(f"服务器运行时出错: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        asyncio.run(start_debug_server())
        print("服务器已正常关闭")
    except KeyboardInterrupt:
        print("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行出错: {e}")
        sys.exit(1)