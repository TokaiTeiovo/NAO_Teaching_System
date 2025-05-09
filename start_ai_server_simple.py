#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
�򻯰��AI�����������ű�
"""

import asyncio
import logging

# ������־
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_ai_server')

async def handle_client(websocket, path):
    """����ͻ�������"""
    client_id = id(websocket)
    logger.info(f"�¿ͻ�������: {client_id}")

    try:
        # ���ͻ�ӭ��Ϣ
        await websocket.send("��ӭ���ӵ��򻯰�AI������!")

        # ���պʹ�����Ϣ
        async for message in websocket:
            logger.info(f"�յ���Ϣ: {message[:100]}")
            await websocket.send(f"Echo: {message}")

    except Exception as e:
        logger.error(f"����ͻ���ʱ����: {e}")
    finally:
        logger.info(f"�ͻ��˶Ͽ�����: {client_id}")

async def start_server(host="localhost", port=8765):
    """����WebSocket������"""
    import websockets

    print(f"����WebSocket������: {host}:{port}")

    try:
        server = await websockets.serve(handle_client, host, port)

        print(f"WebSocket�������������� {host}:{port}")
        print("�����������У���Ctrl+C�˳�...")

        # ���ַ���������
        await asyncio.Future()
    except KeyboardInterrupt:
        print("���յ��ж��źţ��������ر���...")
    except Exception as e:
        print(f"����������ʱ����: {e}")
        logger.error(f"����������ʱ����: {e}")
    finally:
        print("�������ѹر�")

def main():
    """������"""
    import argparse

    parser = argparse.ArgumentParser(description="�򻯰�AI������")
    parser.add_argument("--host", default="localhost", help="������ַ")
    parser.add_argument("--port", type=int, default=8765, help="�˿ں�")

    args = parser.parse_args()

    # ����������
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("�����û��ж�")
    except Exception as e:
        print(f"�������г���: {e}")

if __name__ == "__main__":
    main()
