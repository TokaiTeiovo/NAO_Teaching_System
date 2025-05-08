# test_websocket.py
import asyncio

import websockets


async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)


async def main():
    print("开始测试WebSocket服务器...")
    try:
        server = await websockets.serve(echo, "localhost", 8765)
        print("WebSocket服务器已启动")

        # 保持服务器运行
        await asyncio.Future()  # 运行直到被取消
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())