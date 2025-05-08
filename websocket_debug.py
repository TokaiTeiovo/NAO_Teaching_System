# websocket_debug.py
import asyncio

import websockets


async def handle_client(websocket):
    print("新客户端连接")
    try:
        async for message in websocket:
            print(f"收到消息: {message}")
            await websocket.send(f"收到: {message}")
    except Exception as e:
        print(f"处理客户端时出错: {e}")
    finally:
        print("客户端断开连接")


async def main():
    host = "localhost"
    port = 8765

    print(f"启动WebSocket测试服务器: {host}:{port}")

    # 在websockets 15.0.1中使用的启动方式
    async with websockets.serve(handle_client, host, port):
        print("服务器启动成功，按Ctrl+C退出")
        await asyncio.Future()  # 保持运行


if __name__ == "__main__":
    asyncio.run(main())