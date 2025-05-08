# simple_ai_server.py
import asyncio
import json
import logging

import websockets

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_ai_server')


async def handle_client(websocket, path):
    """处理客户端连接"""
    client_id = id(websocket)
    logger.info(f"新客户端连接: {client_id}")

    try:
        # 发送欢迎消息
        await websocket.send(json.dumps({
            "type": "welcome",
            "data": {"message": "欢迎连接到简化版AI服务器"}
        }))

        # 接收和处理消息
        async for message in websocket:
            logger.info(f"收到消息: {message[:100]}")

            # 简单的回复
            await websocket.send(json.dumps({
                "type": "response",
                "data": {"text": f"收到您的消息，长度为{len(message)}字节"}
            }))

    except websockets.exceptions.ConnectionClosedError as e:
        logger.info(f"客户端连接已关闭: {client_id}, 原因: {e}")
    except Exception as e:
        logger.error(f"处理客户端时出错: {e}", exc_info=True)
    finally:
        logger.info(f"客户端断开连接: {client_id}")


async def main():
    """主函数"""
    host = "localhost"
    port = 8765

    print(f"启动简化版AI服务器，地址: {host}:{port}...")

    server = await websockets.serve(
        handle_client,
        host,
        port,
        ping_interval=30,
        ping_timeout=10
    )

    print(f"服务器已启动，等待连接...")

    try:
        # 永久运行，直到被取消
        await asyncio.Future()
    except KeyboardInterrupt:
        print("接收到中断信号，服务器关闭中...")
    finally:
        server.close()
        await server.wait_closed()
        print("服务器已关闭")


if __name__ == "__main__":
    asyncio.run(main())