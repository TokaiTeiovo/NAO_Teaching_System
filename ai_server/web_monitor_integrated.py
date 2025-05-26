#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - Web监控界面
Flask服务器 + 完整前端界面
"""

import argparse
import logging
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
import psutil
from flask import Flask, render_template_string, jsonify

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 定义路径常量
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 日志配置 ====================
def setup_logger(name, log_level="INFO", log_file=None):
    """设置带颜色的日志记录器"""
    if log_file:
        log_file = LOG_DIR / log_file if not Path(log_file).is_absolute() else Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if getattr(logger, '_configured', False):
        return logger

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    color_mapping = {
        'DEBUG': 'white', 'INFO': 'blue', 'WARNING': 'yellow',
        'ERROR': 'red', 'CRITICAL': 'bold_red',
    }

    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors=color_mapping, style='%'
    )

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._configured = True
    return logger


# ==================== 系统监控器 ====================
class SystemMonitor:
    """系统资源监控器"""

    def __init__(self):
        self.logger = setup_logger('system_monitor', log_file='web_monitor.log')
        self.monitoring = False
        self.stats = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'gpu_info': [],
            'disk_usage': 0.0
        }

    def start_monitoring(self):
        """开始监控系统资源"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        self.logger.info("系统监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.logger.info("系统监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # CPU使用率
                self.stats['cpu_percent'] = psutil.cpu_percent(interval=1)

                # 内存使用率
                memory = psutil.virtual_memory()
                self.stats['memory_percent'] = memory.percent
                self.stats['memory_used'] = memory.used / (1024 ** 3)  # GB
                self.stats['memory_total'] = memory.total / (1024 ** 3)  # GB

                # 磁盘使用率
                disk = psutil.disk_usage('/')
                self.stats['disk_usage'] = disk.percent

                # GPU信息（如果可用）
                self.stats['gpu_info'] = self._get_gpu_info()

            except Exception as e:
                self.logger.error(f"监控系统资源时出错: {e}")

            time.sleep(2)  # 每2秒更新一次

    def _get_gpu_info(self):
        """获取GPU信息"""
        gpu_info = []
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

                # GPU使用率
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # 显存信息
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = memory_info.used / (1024 ** 3)  # GB
                memory_total = memory_info.total / (1024 ** 3)  # GB
                memory_percent = (memory_info.used / memory_info.total) * 100

                gpu_info.append({
                    'name': name,
                    'utilization': utilization.gpu,
                    'memory_used': round(memory_used, 2),
                    'memory_total': round(memory_total, 2),
                    'memory_percent': round(memory_percent, 1)
                })

        except ImportError:
            # pynvml不可用
            pass
        except Exception as e:
            self.logger.error(f"获取GPU信息时出错: {e}")

        return gpu_info

    def get_stats(self):
        """获取当前统计信息"""
        return self.stats.copy()


# ==================== Flask应用 ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal_teaching_system_2025'

# 全局变量
system_monitor = SystemMonitor()
logger = setup_logger('web_monitor', log_file='web_monitor.log')

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多模态智能教学系统 - Web监控界面</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .header h1 {
            color: #4a5568;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            text-align: center;
            color: #718096;
            font-size: 1.1em;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .panel h2 {
            color: #4a5568;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .status-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }

        .status-card:hover {
            transform: translateY(-5px);
        }

        .status-card h3 {
            font-size: 1.2em;
            margin-bottom: 10px;
        }

        .status-card .value {
            font-size: 2em;
            font-weight: bold;
        }

        .chat-container {
            grid-column: 1 / -1;
            height: 600px;
            display: flex;
            flex-direction: column;
        }

        .chat-messages {
            flex: 1;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 15px;
            overflow-y: auto;
            background: #f8fafc;
            margin-bottom: 15px;
        }

        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }

        .message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
        }

        .message.assistant {
            background: #e2e8f0;
            color: #4a5568;
        }

        .message.system {
            background: #fed7d7;
            color: #9b2c2c;
            text-align: center;
            margin: 0 auto;
        }

        .message .timestamp {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }

        .chat-input-container {
            display: flex;
            gap: 10px;
        }

        #messageInput {
            flex: 1;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s ease;
        }

        #messageInput:focus {
            border-color: #667eea;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #e2e8f0;
            color: #4a5568;
        }

        .btn-secondary:hover {
            background: #cbd5e0;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .status-dot.connected {
            background: #48bb78;
            box-shadow: 0 0 10px rgba(72, 187, 120, 0.5);
        }

        .status-dot.disconnected {
            background: #f56565;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }

        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .resource-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 10px;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        .resource-name {
            font-weight: bold;
            color: #4a5568;
        }

        .resource-value {
            color: #667eea;
            font-weight: bold;
        }

        .gpu-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }

        .gpu-name {
            font-weight: bold;
            color: #4a5568;
            margin-bottom: 10px;
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2em;
            }

            .controls {
                flex-direction: column;
            }
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 页面头部 -->
        <div class="header">
            <h1>🤖 多模态智能教学系统</h1>
            <p>基于大语言模型的智能教学助手 - 实时监控与交互界面</p>
        </div>

        <!-- 系统状态面板 -->
        <div class="status-grid">
            <div class="status-card">
                <h3>🔗 连接状态</h3>
                <div class="value" id="connectionStatus">未连接</div>
            </div>
            <div class="status-card">
                <h3>💬 消息总数</h3>
                <div class="value" id="messageCount">0</div>
            </div>
            <div class="status-card">
                <h3>⏱️ 响应时间</h3>
                <div class="value" id="responseTime">-</div>
            </div>
            <div class="status-card">
                <h3>😊 当前情感</h3>
                <div class="value" id="currentEmotion">中性</div>
            </div>
        </div>

        <div class="main-grid">
            <!-- 系统监控面板 -->
            <div class="panel">
                <h2>📊 系统监控</h2>

                <div class="connection-status">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">未连接到AI服务器</span>
                </div>

                <div class="controls">
                    <button class="btn btn-primary" onclick="connectToServer()">连接服务器</button>
                    <button class="btn btn-secondary" onclick="disconnectFromServer()">断开连接</button>
                    <button class="btn btn-secondary" onclick="clearChat()">清空对话</button>
                </div>

                <div>
                    <strong>服务器地址:</strong>
                    <input type="text" id="serverUrl" value="ws://localhost:8765" 
                           style="width: 100%; margin: 5px 0; padding: 8px; border: 1px solid #ccc; border-radius: 5px;">
                </div>

                <h3 style="margin-top: 20px; color: #4a5568;">系统资源</h3>
                <div id="systemResources">
                    <div class="resource-item">
                        <span class="resource-name">CPU使用率:</span>
                        <span class="resource-value" id="cpuUsage">0%</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-name">内存使用:</span>
                        <span class="resource-value" id="memoryUsage">0% (0GB/0GB)</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-name">磁盘使用:</span>
                        <span class="resource-value" id="diskUsage">0%</span>
                    </div>
                </div>

                <div id="gpuInfo" style="margin-top: 15px;"></div>
            </div>

            <!-- 情感分析面板 -->
            <div class="panel">
                <h2>😊 情感分析</h2>
                <div id="emotionDisplay">
                    <p style="text-align: center; color: #718096;">暂无情感数据</p>
                </div>

                <h3 style="margin-top: 20px; color: #4a5568;">学习状态评估</h3>
                <div id="learningStates">
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>注意力:</span>
                            <span id="attentionValue">50%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="attentionBar" style="width: 50%"></div>
                        </div>
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>参与度:</span>
                            <span id="engagementValue">50%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="engagementBar" style="width: 50%"></div>
                        </div>
                    </div>
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>理解度:</span>
                            <span id="understandingValue">50%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="understandingBar" style="width: 50%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 对话界面 -->
        <div class="panel chat-container">
            <h2>💬 智能对话测试</h2>

            <div class="chat-messages" id="chatMessages">
                <div class="message system">
                    <div>欢迎使用多模态智能教学系统！</div>
                    <div>请先连接到AI服务器，然后开始对话测试。</div>
                    <div class="timestamp">系统启动时间: <span id="startTime"></span></div>
                </div>
            </div>

            <div class="chat-input-container">
                <input type="text" id="messageInput" placeholder="输入您的问题..." 
                       onkeypress="handleKeyPress(event)" disabled>
                <button class="btn btn-primary" onclick="sendMessage()" id="sendButton" disabled>
                    发送
                </button>
            </div>
        </div>
    </div>

    <script>
        class AISystemMonitor {
            constructor() {
                this.websocket = null;
                this.isConnected = false;
                this.messageCount = 0;
                this.startTime = new Date();
                this.lastMessageTime = null;

                this.initializeUI();
                this.startSystemMonitoring();
            }

            initializeUI() {
                document.getElementById('startTime').textContent = this.startTime.toLocaleString();
                this.updateConnectionStatus();
            }

            startSystemMonitoring() {
                // 每5秒更新一次系统资源信息
                setInterval(() => {
                    this.updateSystemResources();
                }, 5000);

                // 立即更新一次
                this.updateSystemResources();
            }

            async updateSystemResources() {
                try {
                    const response = await fetch('/api/system_stats');
                    const stats = await response.json();

                    // 更新CPU使用率
                    document.getElementById('cpuUsage').textContent = `${stats.cpu_percent.toFixed(1)}%`;

                    // 更新内存使用率
                    const memoryText = `${stats.memory_percent.toFixed(1)}% (${stats.memory_used.toFixed(1)}GB/${stats.memory_total.toFixed(1)}GB)`;
                    document.getElementById('memoryUsage').textContent = memoryText;

                    // 更新磁盘使用率
                    document.getElementById('diskUsage').textContent = `${stats.disk_usage.toFixed(1)}%`;

                    // 更新GPU信息
                    this.updateGPUInfo(stats.gpu_info);

                } catch (error) {
                    console.error('更新系统资源信息失败:', error);
                }
            }

            updateGPUInfo(gpuInfo) {
                const gpuContainer = document.getElementById('gpuInfo');

                if (gpuInfo && gpuInfo.length > 0) {
                    let gpuHTML = '<h3 style="color: #4a5568; margin-bottom: 10px;">GPU信息</h3>';

                    gpuInfo.forEach((gpu, index) => {
                        gpuHTML += `
                            <div class="gpu-card">
                                <div class="gpu-name">${gpu.name}</div>
                                <div class="resource-item">
                                    <span class="resource-name">GPU使用率:</span>
                                    <span class="resource-value">${gpu.utilization}%</span>
                                </div>
                                <div class="resource-item">
                                    <span class="resource-name">显存使用:</span>
                                    <span class="resource-value">${gpu.memory_percent}% (${gpu.memory_used}GB/${gpu.memory_total}GB)</span>
                                </div>
                            </div>
                        `;
                    });

                    gpuContainer.innerHTML = gpuHTML;
                } else {
                    gpuContainer.innerHTML = '<p style="color: #718096; text-align: center;">未检测到GPU或GPU信息不可用</p>';
                }
            }

            async connectToServer() {
                const serverUrl = document.getElementById('serverUrl').value;

                try {
                    this.addSystemMessage('正在连接到AI服务器...');

                    this.websocket = new WebSocket(serverUrl);

                    this.websocket.onopen = () => {
                        this.isConnected = true;
                        this.updateConnectionStatus();
                        this.addSystemMessage('已成功连接到AI服务器！');

                        // 启用输入控件
                        document.getElementById('messageInput').disabled = false;
                        document.getElementById('sendButton').disabled = false;
                    };

                    this.websocket.onmessage = (event) => {
                        this.handleServerMessage(JSON.parse(event.data));
                    };

                    this.websocket.onclose = () => {
                        this.isConnected = false;
                        this.updateConnectionStatus();
                        this.addSystemMessage('与AI服务器的连接已断开');

                        // 禁用输入控件
                        document.getElementById('messageInput').disabled = true;
                        document.getElementById('sendButton').disabled = true;
                    };

                    this.websocket.onerror = (error) => {
                        console.error('WebSocket错误:', error);
                        this.addSystemMessage('连接错误: 无法连接到AI服务器');
                    };

                } catch (error) {
                    console.error('连接失败:', error);
                    this.addSystemMessage('连接失败: ' + error.message);
                }
            }

            disconnectFromServer() {
                if (this.websocket) {
                    this.websocket.close();
                    this.websocket = null;
                }
            }

            sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();

                if (!message || !this.isConnected) {
                    return;
                }

                // 显示用户消息
                this.addUserMessage(message);

                // 清空输入框
                input.value = '';

                // 发送到服务器
                const messageData = {
                    type: 'text',
                    id: this.generateMessageId(),
                    data: {
                        text: message,
                        context: {
                            timestamp: Date.now(),
                            session_id: 'web_monitor_session'
                        }
                    }
                };

                this.lastMessageTime = Date.now();
                this.websocket.send(JSON.stringify(messageData));
                this.messageCount++;
                this.updateMessageCount();
            }

            handleServerMessage(data) {
                console.log('收到服务器消息:', data);

                if (data.type === 'text_result') {
                    // AI回复消息
                    const responseTime = this.lastMessageTime ? Date.now() - this.lastMessageTime : 0;
                    this.updateResponseTime(responseTime);

                    this.addAssistantMessage(data.data.text || '收到回复但内容为空');

                } else if (data.type === 'emotion_result') {
                    // 情感分析结果
                    this.updateEmotionDisplay(data.data);

                } else if (data.type === 'server_info') {
                    // 服务器信息
                    this.addSystemMessage(data.data.message || '收到服务器信息');

                } else if (data.type === 'error') {
                    // 错误消息
                    this.addSystemMessage('错误: ' + (data.data.message || '未知错误'));
                }
            }

            addUserMessage(text) {
                this.addMessage('user', '👤 您', text);
            }

            addAssistantMessage(text) {
                this.addMessage('assistant', '🤖 AI助教', text);
            }

            addSystemMessage(text) {
                this.addMessage('system', '🔧 系统', text);
            }

            addMessage(type, sender, text) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;

                messageDiv.innerHTML = `
                    <div><strong>${sender}:</strong> ${this.escapeHtml(text)}</div>
                    <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                `;

                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }

            updateConnectionStatus() {
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');
                const connectionStatus = document.getElementById('connectionStatus');

                if (this.isConnected) {
                    statusDot.className = 'status-dot connected';
                    statusText.textContent = '已连接到AI服务器';
                    connectionStatus.textContent = '已连接';
                } else {
                    statusDot.className = 'status-dot disconnected';
                    statusText.textContent = '未连接到AI服务器';
                    connectionStatus.textContent = '未连接';
                }
            }

            updateMessageCount() {
                document.getElementById('messageCount').textContent = this.messageCount;
            }

            updateResponseTime(time) {
                const responseTimeElement = document.getElementById('responseTime');
                if (time > 0) {
                    responseTimeElement.textContent = `${(time / 1000).toFixed(1)}s`;
                }
            }

            updateEmotionDisplay(emotionData) {
                const emotionDisplay = document.getElementById('emotionDisplay');
                const currentEmotion = document.getElementById('currentEmotion');

                if (emotionData.emotions) {
                    // 更新情感显示
                    let emotionHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;">';

                    Object.entries(emotionData.emotions).forEach(([emotion, value]) => {
                        const percentage = (value * 100).toFixed(1);
                        emotionHTML += `
                            <div style="text-align: center; padding: 10px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0;">
                                <div style="font-weight: bold; color: #4a5568; margin-bottom: 5px;">${emotion}</div>
                                <div style="font-size: 1.2em; color: #667eea;">${percentage}%</div>
                                <div class="progress-bar">
                                    <div class="progress-fill" style="width: ${percentage}%"></div>
                                </div>
                            </div>
                        `;
                    });

                    emotionHTML += '</div>';
                    emotionDisplay.innerHTML = emotionHTML;

                    // 更新主导情感
                    if (emotionData.emotion) {
                        currentEmotion.textContent = emotionData.emotion;
                    }
                }

                // 更新学习状态
                if (emotionData.learning_states) {
                    this.updateLearningStates(emotionData.learning_states);
                }
            }

            updateLearningStates(states) {
                Object.entries(states).forEach(([state, value]) => {
                    const percentage = (value * 100).toFixed(0);
                    const stateName = state === '注意力' ? 'attention' : 
                                    state === '参与度' ? 'engagement' : 'understanding';

                    const valueElement = document.getElementById(`${stateName}Value`);
                    const barElement = document.getElementById(`${stateName}Bar`);

                    if (valueElement && barElement) {
                        valueElement.textContent = `${percentage}%`;
                        barElement.style.width = `${percentage}%`;
                    }
                });
            }

            clearChat() {
                const messagesContainer = document.getElementById('chatMessages');
                messagesContainer.innerHTML = `
                    <div class="message system">
                        <div>对话记录已清空</div>
                        <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                    </div>
                `;
                this.messageCount = 0;
                this.updateMessageCount();
            }

            generateMessageId() {
                return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }

            escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    this.sendMessage();
                }
            }
        }

        // 全局实例
        let monitor;

        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            monitor = new AISystemMonitor();

            // 绑定全局函数
            window.connectToServer = () => monitor.connectToServer();
            window.disconnectFromServer = () => monitor.disconnectFromServer();
            window.sendMessage = () => monitor.sendMessage();
            window.clearChat = () => monitor.clearChat();
            window.handleKeyPress = (event) => monitor.handleKeyPress(event);
        });

        // 页面卸载时清理连接
        window.addEventListener('beforeunload', function() {
            if (monitor && monitor.websocket) {
                monitor.websocket.close();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页面"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/system_stats')
def get_system_stats():
    """获取系统统计信息API"""
    return jsonify(system_monitor.get_stats())


@app.route('/api/health')
def health_check():
    """健康检查API"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'uptime': time.time() - system_monitor.stats.get('start_time', time.time())
    })


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - Web监控界面")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="Web服务器端口号")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    logger.info(f"启动Web监控界面: {args.host}:{args.port}")

    # 启动系统监控
    system_monitor.start_monitoring()
    system_monitor.stats['start_time'] = time.time()

    try:
        # 启动Flask应用
        logger.info("正在启动Flask Web服务器...")
        print(f"\n🌐 多模态智能教学系统 - Web监控界面")
        print(f"📍 访问地址: http://{args.host}:{args.port}")
        print(f"🔧 系统监控已启动")
        print(f"💡 使用说明:")
        print(f"   1. 在浏览器中访问上述地址")
        print(f"   2. 点击'连接服务器'按钮")
        print(f"   3. 开始与AI助教对话")
        print(f"   4. 按 Ctrl+C 退出\n")

        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False  # 避免重载时重复启动监控
        )

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"启动Web服务器时出错: {e}")
    finally:
        # 停止系统监控
        system_monitor.stop_monitoring()
        logger.info("Web监控界面已关闭")


if __name__ == "__main__":
    main()