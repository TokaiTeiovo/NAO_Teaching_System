#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NAO Robot Simulator - English Version (Python 2.7)
"""

import json
import sys
import threading
import time

import websocket

# For Python 2.7
reload(sys)
sys.setdefaultencoding('utf-8')

class NAOSimulator(object):
    def __init__(self):
        self.is_speaking = False

    def say(self, text):
        print("\n[NAO says] " + text)
        self.is_speaking = True
        time.sleep(len(text) * 0.02)
        self.is_speaking = False
        return True

    def perform_gesture(self, gesture_name):
        print("\n[NAO gesture] " + gesture_name)

        if gesture_name == "explaining":
            print("[Details] Hands open, explaining posture")
        elif gesture_name == "pointing":
            print("[Details] Right hand pointing forward")
        elif gesture_name == "thinking":
            print("[Details] Head slightly tilted, hand on chin")
        elif gesture_name == "greeting":
            print("[Details] Right hand raised, waving")

        return True

class WebSocketClient(object):
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.ws = None
        self.connected = False
        self.simulator = NAOSimulator()

    def connect(self):
        try:
            print("Connecting to server: " + self.server_url)

            websocket.enableTrace(False)

            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )

            thread = threading.Thread(target=self.ws.run_forever)
            thread.daemon = True
            thread.start()

            timeout = 5
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)

            return self.connected
        except Exception as e:
            print("Connection failed: " + str(e))
            return False

    def on_open(self, ws):
        self.connected = True
        print("Connected to AI server")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            print("Received message type: " + msg_type)

            if msg_type == "text_result":
                text = data.get("data", {}).get("text", "")
                actions = data.get("data", {}).get("actions", [])

                if text:
                    self.simulator.say(text)

                for action in actions:
                    self.simulator.perform_gesture(action)
        except Exception as e:
            print("Error processing message: " + str(e))

    def on_error(self, ws, error):
        print("WebSocket error: " + str(error))

    def on_close(self, ws, *args):
        self.connected = False
        print("WebSocket connection closed")

    def send_text(self, text):
        if not self.connected:
            print("Not connected to server")
            return False

        try:
            message = {
                "type": "text",
                "id": str(time.time()),
                "data": {
                    "text": text
                }
            }

            self.ws.send(json.dumps(message))
            print("Sent text: " + text)
            return True
        except Exception as e:
            print("Error sending text: " + str(e))
            return False

    def run_interactive(self):
        if not self.connected:
            print("Not connected to server")
            return

        print("\n=== NAO Simulator Interactive Mode ===")
        print("Type 'exit' to quit")

        while True:
            try:
                text = raw_input("\n[Student] ")

                if text.lower() in ["exit", "quit"]:
                    break

                self.send_text(text)
                time.sleep(1)
            except KeyboardInterrupt:
                break

        print("Interactive session ended")

    def run_teaching_demo(self):
        topics = [
            "Hello, I am a student",
            "What is a variable?",
            "What is the difference between integer and floating-point variables?",
            "How do I define an integer variable in C?",
            "What is the basic structure of a for loop?",
            "Can you give me an example of an if statement?",
            "Thank you for explaining"
        ]

        print("\n=== Starting C Programming Teaching Demo ===\n")

        for topic in topics:
            print("[Student] " + topic)
            self.send_text(topic)
            time.sleep(8)

        print("\n=== Teaching Demo Ended ===\n")

def main():
    server_url = "ws://localhost:8765"
    mode = "interactive"

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--server-url" and i+1 < len(sys.argv):
            server_url = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--mode" and i+1 < len(sys.argv):
            mode = sys.argv[i+1]
            i += 2
        else:
            i += 1

    print("Using server URL: " + server_url)
    print("Running mode: " + mode)

    client = WebSocketClient(server_url)

    if client.connect():
        print("Successfully connected to AI server")

        if mode == "demo":
            client.run_teaching_demo()
        else:
            client.run_interactive()
    else:
        print("Connection failed, please check server address")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program execution error: " + str(e))