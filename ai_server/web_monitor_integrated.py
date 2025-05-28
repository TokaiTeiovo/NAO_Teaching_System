#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - 增强版Web界面
包含两个页面：知识图谱生成页面 和 智能对话页面
"""

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import colorlog
import psutil
from flask import Flask, render_template_string, jsonify, request
from werkzeug.utils import secure_filename

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 定义路径常量
LOG_DIR = PROJECT_ROOT / "logs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
SHARED_DIR = PROJECT_ROOT / "shared"

# 确保目录存在
for directory in [LOG_DIR, UPLOAD_DIR, SHARED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 允许的文件扩展名
ALLOWED_PDF_EXTENSIONS = {'pdf'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
ALLOWED_JSON_EXTENSIONS = {'json'}


def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


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

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self.stats['cpu_percent'] = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                self.stats['memory_percent'] = memory.percent
                self.stats['memory_used'] = memory.used / (1024 ** 3)
                self.stats['memory_total'] = memory.total / (1024 ** 3)
                disk = psutil.disk_usage('/')
                self.stats['disk_usage'] = disk.percent
                self.stats['gpu_info'] = self._get_gpu_info()
            except Exception as e:
                self.logger.error(f"监控系统资源时出错: {e}")
            time.sleep(2)

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
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = memory_info.used / (1024 ** 3)
                memory_total = memory_info.total / (1024 ** 3)
                memory_percent = (memory_info.used / memory_info.total) * 100
                gpu_info.append({
                    'name': name,
                    'utilization': utilization.gpu,
                    'memory_used': round(memory_used, 2),
                    'memory_total': round(memory_total, 2),
                    'memory_percent': round(memory_percent, 1)
                })
        except:
            pass
        return gpu_info

    def get_stats(self):
        """获取当前统计信息"""
        return self.stats.copy()


# ==================== Flask应用 ====================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'multimodal_teaching_system_2025'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 全局变量
system_monitor = SystemMonitor()
logger = setup_logger('web_monitor', log_file='web_monitor.log')

# 知识图谱生成页面HTML模板
KNOWLEDGE_GRAPH_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>知识图谱生成器 - 多模态智能教学系统</title>
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
            max-width: 1200px;
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
            text-align: center;
        }

        .header h1 {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
        }

        .nav-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.3s ease;
        }

        .nav-btn:hover {
            transform: translateY(-2px);
        }

        .nav-btn.active {
            background: linear-gradient(135deg, #764ba2, #667eea);
        }

        .main-content {
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

        .upload-section {
            grid-column: 1 / -1;
        }

        .upload-area {
            border: 2px dashed #cbd5e0;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }

        .upload-area.dragover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }

        .file-input {
            display: none;
        }

        .upload-icon {
            font-size: 3em;
            color: #a0aec0;
            margin-bottom: 15px;
        }

        .upload-text {
            font-size: 1.2em;
            color: #4a5568;
            margin-bottom: 10px;
        }

        .upload-hint {
            color: #718096;
            font-size: 0.9em;
        }

        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #e2e8f0;
        }

        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }

        .tab.active {
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #4a5568;
        }

        .form-control {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s ease;
        }

        .form-control:focus {
            outline: none;
            border-color: #667eea;
        }

        select.form-control {
            cursor: pointer;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
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

        .progress-container {
            margin: 20px 0;
            display: none;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }

        .status-message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }

        .status-success {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }

        .status-error {
            background: #fed7d7;
            color: #9b2c2c;
            border: 1px solid #feb2b2;
        }

        .status-info {
            background: #bee3f8;
            color: #2a69ac;
            border: 1px solid #90cdf4;
        }

        .system-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }

        .stat-card {
            background: #f8fafc;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .stat-title {
            font-size: 0.9em;
            color: #718096;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4a5568;
        }

        .log-area {
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            margin-top: 20px;
        }
        
        /* 步骤进度条样式 */
        .steps-progress {
            margin-bottom: 20px;
        }
        
        .step-item {
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8fafc;
            border-radius: 10px;
            border-left: 4px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .step-item.active {
            background: rgba(102, 126, 234, 0.1);
            border-left-color: #667eea;
            transform: translateX(5px);
        }
        
        .step-item.completed {
            background: rgba(72, 187, 120, 0.1);
            border-left-color: #48bb78;
        }
        
        .step-item.error {
            background: rgba(245, 101, 101, 0.1);
            border-left-color: #f56565;
        }
        
        .step-number {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #4a5568;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            transition: all 0.3s ease;
        }
        
        .step-item.active .step-number {
            background: #667eea;
            color: white;
        }
        
        .step-item.completed .step-number {
            background: #48bb78;
            color: white;
        }
        
        .step-item.error .step-number {
            background: #f56565;
            color: white;
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-title {
            font-weight: bold;
            color: #4a5568;
            font-size: 1.1em;
            margin-bottom: 3px;
        }
        
        .step-description {
            color: #718096;
            font-size: 0.9em;
        }
        
        .step-status {
            font-size: 1.5em;
            margin-left: 15px;
        }
        
        .step-item.active .step-status {
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .overall-progress {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }
        
        /* 隐藏旧的进度条容器 */
        .progress-container {
            display: none;
        }
        
        .progress-container.show {
            display: block;
        }
        
        .spinner {
            animation: spin 1.5s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        /* 步骤进度条样式 */
        .steps-progress {
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .step-item {
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8fafc;
            border-radius: 10px;
            border-left: 4px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .step-item.waiting {
            border-left-color: #e2e8f0;
        }
        
        .step-item.active {
            background: rgba(102, 126, 234, 0.1);
            border-left-color: #667eea;
            transform: translateX(5px);
        }
        
        .step-item.completed {
            background: rgba(72, 187, 120, 0.1);
            border-left-color: #48bb78;
            opacity: 0.8;
        }
        
        .step-item.error {
            background: rgba(245, 101, 101, 0.1);
            border-left-color: #f56565;
        }
        
        .step-number {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #4a5568;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
            transition: all 0.3s ease;
        }
        
        .step-item.active .step-number {
            background: #667eea;
            color: white;
        }
        
        .step-item.completed .step-number {
            background: #48bb78;
            color: white;
        }
        
        .step-item.error .step-number {
            background: #f56565;
            color: white;
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-title {
            font-weight: bold;
            color: #4a5568;
            font-size: 1.1em;
            margin-bottom: 3px;
        }
        
        .step-description {
            color: #718096;
            font-size: 0.9em;
        }
        
        .step-status {
            font-size: 1.5em;
            margin-left: 15px;
        }
        
        .overall-progress {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
        }
        
        /* 隐藏/显示进度容器 */
        .progress-container {
            display: none;
        }
        
        .progress-container.show {
            display: block;
        }
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2em;
            }

            .nav-buttons {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 页面头部 -->
        <div class="header">
            <h1>📊 知识图谱生成器</h1>
            <p>上传PDF、图片文件夹或JSON文件，自动生成知识图谱并导入Neo4j数据库</p>
            <div class="nav-buttons">
                <a href="/" class="nav-btn active">知识图谱</a>
                <a href="/chat" class="nav-btn">智能对话</a>
            </div>
        </div>

        <div class="main-content">
            <!-- 文件上传区域 -->
            <div class="panel upload-section">
                <h2>📁 文件上传</h2>

                <!-- 选项卡 -->
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('pdf')">📄 PDF文件</div>
                    <div class="tab" onclick="switchTab('images')">🖼️ 图片文件夹</div>
                    <div class="tab" onclick="switchTab('json')">📋 JSON文件</div>
                </div>

                <!-- PDF上传 -->
                <div id="pdf-tab" class="tab-content active">
                    <form id="pdf-form" method="POST" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('pdf-file').click()">
                            <div class="upload-icon">📄</div>
                            <div class="upload-text">点击选择PDF文件</div>
                            <div class="upload-hint">
                                支持格式: PDF (最大100MB)<br>
                                <small style="color: #e53e3e;">大文件建议: 页数超过100页的文档将自动限制处理前50页</small>
                            </div>
                        </div>
                        <input type="file" id="pdf-file" name="file" class="file-input" accept=".pdf" onchange="handlePDFFileSelect(this)">
                
                        <div class="form-group">
                            <label for="pdf-domain">知识领域</label>
                            <select id="pdf-domain" name="domain" class="form-control">
                                <option value="计算机科学">计算机科学</option>
                                <option value="数学">数学</option>
                                <option value="物理学">物理学</option>
                                <option value="化学">化学</option>
                                <option value="生物学">生物学</option>
                                <option value="医学">医学</option>
                                <option value="心理学">心理学</option>
                                <option value="经济学">经济学</option>
                                <option value="哲学">哲学</option>
                                <option value="语言学">语言学</option>
                            </select>
                        </div>
                
                        <div class="form-group">
                            <label for="pdf-batch-size">批次大小</label>
                            <input type="number" id="pdf-batch-size" name="batch_size" class="form-control" value="5" min="1" max="20">
                            <small style="color: #718096;">建议: 大文件使用较小的批次(3-5页)</small>
                        </div>
                
                        <div class="form-group">
                            <label for="pdf-max-pages">最大处理页数</label>
                            <input type="number" id="pdf-max-pages" name="max_pages" class="form-control" value="50" min="1" max="200">
                            <small style="color: #718096;">为避免超时，建议限制处理页数</small>
                        </div>
                
                        <button type="button" class="btn btn-primary" onclick="processPDF()">🚀 生成知识图谱</button>
                    </form>
                </div>

                <!-- 图片文件夹上传 -->
                <div id="images-tab" class="tab-content">
                    <form id="images-form" method="POST" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('images-folder').click()">
                            <div class="upload-icon">🖼️</div>
                            <div class="upload-text">选择图片文件夹</div>
                            <div class="upload-hint">支持格式: PNG, JPG, JPEG, GIF, BMP, TIFF</div>
                        </div>
                        <input type="file" id="images-folder" name="files" class="file-input" webkitdirectory directory accept="image/*" onchange="handleFileSelect(this, 'images')">
                        <div class="form-group">
                            <label for="images-domain">知识领域</label>
                            <select id="images-domain" name="domain" class="form-control">
                                <option value="计算机科学">计算机科学</option>
                                <option value="数学">数学</option>
                                <option value="物理学">物理学</option>
                                <option value="化学">化学</option>
                                <option value="生物学">生物学</option>
                                <option value="医学">医学</option>
                                <option value="心理学">心理学</option>
                                <option value="经济学">经济学</option>
                                <option value="哲学">哲学</option>
                                <option value="语言学">语言学</option>
                            </select>
                        </div>

                        <button type="button" class="btn btn-primary" onclick="processImages()">🚀 生成知识图谱</button>
                    </form>
                </div>

                <!-- JSON文件上传 -->
                <div id="json-tab" class="tab-content">
                    <form id="json-form" method="POST" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('json-file').click()">
                            <div class="upload-icon">📋</div>
                            <div class="upload-text">选择JSON文件</div>
                            <div class="upload-hint">已提取的文本JSON文件</div>
                        </div>
                        <input type="file" id="json-file" name="file" class="file-input" accept=".json" onchange="handleFileSelect(this, 'json')">

                        <div class="form-group">
                            <label for="json-domain">知识领域</label>
                            <select id="json-domain" name="domain" class="form-control">
                                <option value="计算机科学">计算机科学</option>
                                <option value="数学">数学</option>
                                <option value="物理学">物理学</option>
                                <option value="化学">化学</option>
                                <option value="生物学">生物学</option>
                                <option value="医学">医学</option>
                                <option value="心理学">心理学</option>
                                <option value="经济学">经济学</option>
                                <option value="哲学">哲学</option>
                                <option value="语言学">语言学</option>
                            </select>
                        </div>

                        <button type="button" class="btn btn-primary" onclick="processJSON()">🚀 生成知识图谱</button>
                    </form>
                </div>

                <!-- 详细步骤进度条 -->
                <div id="progress-container" class="progress-container">
                    <div class="steps-progress">
                        <!-- 步骤将通过JavaScript动态生成 -->
                    </div>
                    
                    <!-- 总体进度条 -->
                    <div class="overall-progress">
                        <div class="progress-bar">
                            <div id="progress-fill" class="progress-fill"></div>
                        </div>
                        <div id="progress-text" style="text-align: center; margin-top: 10px;">准备中...</div>
                    </div>
                </div>

                <!-- 状态消息 -->
                <div id="status-message" class="status-message"></div>

                <!-- 处理日志 -->
                <div id="log-area" class="log-area" style="display: none;"></div>
            </div>

            <!-- 系统状态面板 -->
            <div class="panel">
                <h2>🖥️ 系统状态</h2>
                <div class="system-stats" id="system-stats">
                    <div class="stat-card">
                        <div class="stat-title">CPU使用率</div>
                        <div class="stat-value" id="cpu-usage">0%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">内存使用</div>
                        <div class="stat-value" id="memory-usage">0%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">磁盘使用</div>
                        <div class="stat-value" id="disk-usage">0%</div>
                    </div>
                </div>
                <div id="gpu-info"></div>
            </div>

            <!-- Neo4j配置面板 -->
            <div class="panel">
                <h2>🗄️ Neo4j配置</h2>
                <div class="form-group">
                    <label for="neo4j-uri">数据库URI</label>
                    <input type="text" id="neo4j-uri" class="form-control" value="bolt://localhost:7687">
                </div>
                <div class="form-group">
                    <label for="neo4j-user">用户名</label>
                    <input type="text" id="neo4j-user" class="form-control" value="neo4j">
                </div>
                <div class="form-group">
                    <label for="neo4j-password">密码</label>
                    <input type="password" id="neo4j-password" class="form-control" value="admin123">
                </div>
                <button type="button" class="btn btn-secondary" onclick="testNeo4jConnection()">🔍 测试连接</button>
            </div>
        </div>
    </div>

    <script>
        let currentProcessId = null;
        let currentFlow = [];  // 确保这个变量在全局作用域中定义
        let currentFileType = '';

        // 步骤配置 - 根据不同文件类型定义不同流程
        const PROCESSING_FLOWS = {
            pdf: [
                { id: 1, name: '文件上传', description: '上传并验证PDF文件' },
                { id: 2, name: 'PDF转图像', description: '将PDF页面转换为图像' },
                { id: 3, name: 'OCR文本提取', description: '使用OCR技术提取文本内容' },
                { id: 4, name: 'LLM知识提取', description: '使用大语言模型提取概念和定义' },
                { id: 5, name: '关系抽取', description: '分析概念间的关联关系' },
                { id: 6, name: '知识图谱生成', description: '构建完整的知识图谱' },
                { id: 7, name: '数据库导入', description: '导入到Neo4j图数据库' }
            ],
            images: [
                { id: 1, name: '文件上传', description: '上传并验证图片文件' },
                { id: 2, name: 'OCR文本提取', description: '使用OCR技术提取图片文本' },
                { id: 3, name: 'LLM知识提取', description: '使用大语言模型提取概念和定义' },
                { id: 4, name: '关系抽取', description: '分析概念间的关联关系' },
                { id: 5, name: '知识图谱生成', description: '构建完整的知识图谱' },
                { id: 6, name: '数据库导入', description: '导入到Neo4j图数据库' }
            ],
            json: [
                { id: 1, name: '文件上传', description: '上传并验证JSON文件' },
                { id: 2, name: 'LLM知识提取', description: '使用大语言模型提取概念和定义' },
                { id: 3, name: '关系抽取', description: '分析概念间的关联关系' },
                { id: 4, name: '知识图谱生成', description: '构建完整的知识图谱' },
                { id: 5, name: '数据库导入', description: '导入到Neo4j图数据库' }
            ]
        };
        
        // 动态生成步骤进度条
        function generateStepsHTML(flowType) {
            const steps = PROCESSING_FLOWS[flowType];
            let stepsHTML = '';
            
            steps.forEach(step => {
                stepsHTML += `
                    <div class="step-item" id="step-${step.id}">
                        <div class="step-number">${step.id}</div>
                        <div class="step-content">
                            <div class="step-title">${step.name}</div>
                            <div class="step-description">${step.description}</div>
                        </div>
                        <div class="step-status">⏳</div>
                    </div>
                `;
            });
            
            return stepsHTML;
        }
    
        // 初始化步骤进度条
        function initializeSteps(fileType) {
            currentFileType = fileType;
            currentFlow = PROCESSING_FLOWS[fileType];
            
            const stepsContainer = document.querySelector('.steps-progress');
            stepsContainer.innerHTML = generateStepsHTML(fileType);
            
            // 显示进度容器
            const progressContainer = document.getElementById('progress-container');
            progressContainer.classList.add('show');
            
            // 重置所有步骤状态
            currentFlow.forEach(step => {
                updateStepStatus(step.id, 'waiting');
            });
        }
    
        // 更新步骤状态
        function updateStepStatus(stepNumber, status, message = '') {
            const stepElement = document.getElementById(`step-${stepNumber}`);
            if (!stepElement) return;
    
            // 移除所有状态类
            stepElement.classList.remove('active', 'completed', 'error', 'waiting');
            
            // 添加新状态
            stepElement.classList.add(status);
            
            // 更新状态图标
            const statusElement = stepElement.querySelector('.step-status');
            switch(status) {
                case 'active':
                    statusElement.innerHTML = '<div class="spinner">🔄</div>';
                    statusElement.title = '正在处理...';
                    break;
                case 'completed':
                    statusElement.textContent = '✅';
                    statusElement.title = '已完成';
                    break;
                case 'error':
                    statusElement.textContent = '❌';
                    statusElement.title = '处理失败';
                    break;
                case 'waiting':
                default:
                    statusElement.textContent = '⏳';
                    statusElement.title = '等待中';
            }
            
            // 更新描述
            if (message) {
                const descElement = stepElement.querySelector('.step-description');
                descElement.textContent = message;
            }
        }
    
        // 显示详细进度
        function showDetailedProgress(currentStepId, message = '') {
            // 确保currentFlow已初始化
            if (!currentFlow || currentFlow.length === 0) {
                console.warn('currentFlow未初始化，使用默认PDF流程');
                currentFlow = PROCESSING_FLOWS.pdf;
                currentFileType = 'pdf';
            }
            
            const totalSteps = currentFlow.length;
            const progressPercent = Math.round((currentStepId / totalSteps) * 100);
            
            // 更新总体进度条
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            
            if (progressFill) progressFill.style.width = progressPercent + '%';
            if (progressText) progressText.textContent = `步骤 ${currentStepId}/${totalSteps}: ${message}`;
            
            // 更新步骤状态
            currentFlow.forEach(step => {
                if (step.id < currentStepId) {
                    updateStepStatus(step.id, 'completed');
                } else if (step.id === currentStepId) {
                    updateStepStatus(step.id, 'active', message);
                } else {
                    updateStepStatus(step.id, 'waiting');
                }
            });
            
            // 滚动到当前步骤
            const currentStepElement = document.getElementById(`step-${currentStepId}`);
            if (currentStepElement) {
                currentStepElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    
        // 显示步骤错误
        function showStepError(stepId, errorMessage) {
            updateStepStatus(stepId, 'error', errorMessage);
            showStatus(`步骤失败: ${errorMessage}`, 'error');
            
            // 更新进度文本
            const progressText = document.getElementById('progress-text');
            if (progressText) {
                progressText.textContent = `处理失败: ${errorMessage}`;
            }
        }
    
        // 显示所有步骤完成
        function showAllStepsCompleted(result) {
            // 确保currentFlow已初始化
            if (!currentFlow || currentFlow.length === 0) {
                console.warn('currentFlow未初始化，使用默认PDF流程');
                currentFlow = PROCESSING_FLOWS.pdf;
            }
            
            currentFlow.forEach(step => {
                updateStepStatus(step.id, 'completed');
            });
            
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');
            
            if (progressFill) progressFill.style.width = '100%';
            if (progressText) progressText.textContent = '所有步骤已完成！';
            
            const conceptsCount = result.concepts_count || 'N/A';
            const relationsCount = result.relations_count || 'N/A';
            showStatus(`知识图谱生成完成！共提取 ${conceptsCount} 个概念，${relationsCount} 个关系`, 'success');
        }
    
        // 根据文件类型和进度映射步骤
        function mapProgressToStep(fileType, progress) {
            const steps = PROCESSING_FLOWS[fileType];
            const totalSteps = steps.length;
            
            if (fileType === 'pdf') {
                if (progress <= 5) return { step: 1, message: '文件验证完成' };
                if (progress <= 25) return { step: 2, message: 'PDF转图像处理中...' };
                if (progress <= 40) return { step: 3, message: 'OCR文本提取中...' };
                if (progress <= 60) return { step: 4, message: 'LLM知识提取中...' };
                if (progress <= 75) return { step: 5, message: '关系抽取中...' };
                if (progress <= 85) return { step: 6, message: '知识图谱生成中...' };
                if (progress <= 95) return { step: 7, message: 'Neo4j数据库导入中...' };
                return { step: 7, message: '处理完成' };
            } else if (fileType === 'images') {
                if (progress <= 10) return { step: 1, message: '图片文件验证完成' };
                if (progress <= 40) return { step: 2, message: '图片OCR文本提取中...' };
                if (progress <= 60) return { step: 3, message: 'LLM知识提取中...' };
                if (progress <= 75) return { step: 4, message: '关系抽取中...' };
                if (progress <= 90) return { step: 5, message: '知识图谱生成中...' };
                return { step: 6, message: 'Neo4j数据库导入中...' };
            } else if (fileType === 'json') {
                if (progress <= 15) return { step: 1, message: 'JSON文件验证完成' };
                if (progress <= 40) return { step: 2, message: 'LLM知识提取中...' };
                if (progress <= 65) return { step: 3, message: '关系抽取中...' };
                if (progress <= 85) return { step: 4, message: '知识图谱生成中...' };
                return { step: 5, message: 'Neo4j数据库导入中...' };
            }
            
            return { step: 1, message: '处理中...' };
        }
        
        // 切换选项卡
        function switchTab(tabName) {
            // 隐藏所有选项卡内容
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });

            // 移除所有选项卡的激活状态
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });

            // 显示当前选项卡内容
            document.getElementById(tabName + '-tab').classList.add('active');

            // 激活当前选项卡
            event.target.classList.add('active');
        }

        // 处理文件选择
        function handleFileSelect(input, type) {
            const files = input.files;
            if (files.length > 0) {
                let message = '';
                if (type === 'images') {
                    // 显示文件夹信息
                    if (files.length > 0) {
                        const folderName = files[0].webkitRelativePath.split('/')[0];
                        message = `已选择文件夹: ${folderName} (包含 ${files.length} 个图片文件)`;
                    }
                } else {
                    message = `已选择文件: ${files[0].name}`;
                }
                showStatus(message, 'info');
            }
        }

        // 显示状态消息
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status-message');
            statusDiv.textContent = message;
            statusDiv.className = `status-message status-${type}`;
            statusDiv.style.display = 'block';

            if (type === 'success' || type === 'error') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 5000);
            }
        }

        // 显示进度
        function showProgress(percent, text) {
            const progressContainer = document.getElementById('progress-container');
            const progressFill = document.getElementById('progress-fill');
            const progressText = document.getElementById('progress-text');

            progressContainer.style.display = 'block';
            progressFill.style.width = percent + '%';
            progressText.textContent = text;

            if (percent >= 100) {
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 3000);
            }
        }

        // 添加日志
        function addLog(message) {
            const logArea = document.getElementById('log-area');
            logArea.style.display = 'block';
            const timestamp = new Date().toLocaleTimeString();
            logArea.innerHTML += `[${timestamp}] ${message}\n`;
            logArea.scrollTop = logArea.scrollHeight;
        }

        // 处理PDF
        async function processPDF() {
            const fileInput = document.getElementById('pdf-file');
            const file = fileInput.files[0];
        
            if (!file) {
                showStatus('请选择PDF文件', 'error');
                return;
            }
        
            const formData = new FormData();
            formData.append('file', file);
            formData.append('domain', document.getElementById('pdf-domain').value);
            formData.append('batch_size', document.getElementById('pdf-batch-size').value);
            formData.append('import_neo4j', 'true');
            formData.append('neo4j_uri', document.getElementById('neo4j-uri').value);
            formData.append('neo4j_user', document.getElementById('neo4j-user').value);
            formData.append('neo4j_password', document.getElementById('neo4j-password').value);
        
            try {
                // 步骤1: 文件上传
                showDetailedProgress(1, 6, '上传PDF文件到服务器...');
                addLog('开始PDF知识图谱生成流程');
        
                const response = await fetch('/api/process_pdf', {
                    method: 'POST',
                    body: formData
                });
        
                if (response.ok) {
                    const result = await response.json();
                    currentProcessId = result.process_id;
                    
                    // 步骤2: 开始OCR处理
                    showDetailedProgress(2, 6, '开始OCR文本提取...');
                    addLog('文件上传完成，开始OCR处理');
        
                    // 开始轮询处理状态
                    pollProcessStatus();
                } else {
                    const error = await response.json();
                    showStepError(1, error.message);
                    addLog(`错误: ${error.message}`);
                }
            } catch (error) {
                showStepError(1, error.message);
                addLog(`网络错误: ${error.message}`);
            }
        }

        // 处理图片
        async function processImages() {
            const fileInput = document.getElementById('images-folder');
            const files = fileInput.files;

            if (files.length === 0) {
                showStatus('请选择图片文件', 'error');
                return;
            }

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            formData.append('domain', document.getElementById('images-domain').value);
            formData.append('import_neo4j', 'true');
            formData.append('neo4j_uri', document.getElementById('neo4j-uri').value);
            formData.append('neo4j_user', document.getElementById('neo4j-user').value);
            formData.append('neo4j_password', document.getElementById('neo4j-password').value);

            try {
                showStatus(`开始处理 ${files.length} 个图片文件...`, 'info');
                showProgress(10, '上传文件中...');
                addLog(`开始图片知识图谱生成流程 (${files.length} 个文件)`);

                const response = await fetch('/api/process_images', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    currentProcessId = result.process_id;
                    pollProcessStatus();
                } else {
                    const error = await response.json();
                    showStatus(`处理失败: ${error.message}`, 'error');
                    addLog(`错误: ${error.message}`);
                }
            } catch (error) {
                showStatus(`网络错误: ${error.message}`, 'error');
                addLog(`网络错误: ${error.message}`);
            }
        }

        // 处理JSON
        async function processJSON() {
            const fileInput = document.getElementById('json-file');
            const file = fileInput.files[0];

            if (!file) {
                showStatus('请选择JSON文件', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);
            formData.append('domain', document.getElementById('json-domain').value);
            formData.append('import_neo4j', 'true');
            formData.append('neo4j_uri', document.getElementById('neo4j-uri').value);
            formData.append('neo4j_user', document.getElementById('neo4j-user').value);
            formData.append('neo4j_password', document.getElementById('neo4j-password').value);

            try {
                showStatus('开始处理JSON文件...', 'info');
                showProgress(10, '上传文件中...');
                addLog('开始JSON知识图谱生成流程');

                const response = await fetch('/api/process_json', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const result = await response.json();
                    currentProcessId = result.process_id;
                    pollProcessStatus();
                } else {
                    const error = await response.json();
                    showStatus(`处理失败: ${error.message}`, 'error');
                    addLog(`错误: ${error.message}`);
                }
            } catch (error) {
                showStatus(`网络错误: ${error.message}`, 'error');
                addLog(`网络错误: ${error.message}`);
            }
        }

        // 轮询处理状态
         async function pollProcessStatus() {
                if (!currentProcessId) return;
        
                try {
                    const response = await fetch(`/api/process_status/${currentProcessId}`);
                    const status = await response.json();
        
                    if (status.status === 'processing') {
                        // 根据文件类型和进度映射到具体步骤
                        const stepInfo = mapProgressToStep(currentFileType, status.progress);
                        showDetailedProgress(stepInfo.step, stepInfo.message);
                        addLog(status.message || stepInfo.message);
                        setTimeout(pollProcessStatus, 2000);
                        
                    } else if (status.status === 'completed') {
                        showAllStepsCompleted(status.result);
                        addLog(`处理完成: ${status.result.concepts_count || 'N/A'} 个概念, ${status.result.relations_count || 'N/A'} 个关系`);
                        if (status.result.neo4j_imported) {
                            addLog('成功导入到Neo4j数据库');
                        }
                        currentProcessId = null;
                        
                    } else if (status.status === 'error') {
                        // 根据当前文件类型和错误信息确定错误步骤
                        const currentStep = getCurrentStepFromError(status.error);
                        showStepError(currentStep, status.error);
                        addLog(`处理失败: ${status.error}`);
                        currentProcessId = null;
                    }
                } catch (error) {
                    addLog(`状态检查失败: ${error.message}`);
                    setTimeout(pollProcessStatus, 5000);
                }
            }

        // 根据错误信息确定当前步骤
        function getCurrentStepFromError(errorMessage) {
            const error = errorMessage.toLowerCase();
            
            if (currentFileType === 'pdf') {
                if (error.includes('pdf') || error.includes('convert')) return 2;
                if (error.includes('ocr')) return 3;
                if (error.includes('llm') || error.includes('knowledge')) return 4;
                if (error.includes('relation')) return 5;
                if (error.includes('graph')) return 6;
                if (error.includes('neo4j')) return 7;
            } else if (currentFileType === 'images') {
                if (error.includes('ocr')) return 2;
                if (error.includes('llm') || error.includes('knowledge')) return 3;
                if (error.includes('relation')) return 4;
                if (error.includes('graph')) return 5;
                if (error.includes('neo4j')) return 6;
            } else if (currentFileType === 'json') {
                if (error.includes('llm') || error.includes('knowledge')) return 2;
                if (error.includes('relation')) return 3;
                if (error.includes('graph')) return 4;
                if (error.includes('neo4j')) return 5;
            }
            
            return 1; // 默认返回第一步
        }

        // 测试Neo4j连接
        async function testNeo4jConnection() {
            const uri = document.getElementById('neo4j-uri').value;
            const user = document.getElementById('neo4j-user').value;
            const password = document.getElementById('neo4j-password').value;

            try {
                showStatus('测试Neo4j连接...', 'info');

                const response = await fetch('/api/test_neo4j', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ uri, user, password })
                });

                const result = await response.json();

                if (result.success) {
                    showStatus('Neo4j连接成功！', 'success');
                    addLog('Neo4j数据库连接测试成功');
                } else {
                    showStatus(`Neo4j连接失败: ${result.error}`, 'error');
                    addLog(`Neo4j连接失败: ${result.error}`);
                }
            } catch (error) {
                showStatus(`连接测试失败: ${error.message}`, 'error');
                addLog(`连接测试失败: ${error.message}`);
            }
        }

        // 更新系统状态
        async function updateSystemStats() {
            try {
                const response = await fetch('/api/system_stats');
                const stats = await response.json();

                document.getElementById('cpu-usage').textContent = `${stats.cpu_percent.toFixed(1)}%`;
                document.getElementById('memory-usage').textContent = `${stats.memory_percent.toFixed(1)}%`;
                document.getElementById('disk-usage').textContent = `${stats.disk_usage.toFixed(1)}%`;

                // 更新GPU信息
                const gpuContainer = document.getElementById('gpu-info');
                if (stats.gpu_info && stats.gpu_info.length > 0) {
                    let gpuHTML = '<h3 style="margin-top: 15px; color: #4a5568;">GPU状态</h3>';
                    stats.gpu_info.forEach(gpu => {
                        gpuHTML += `
                            <div class="stat-card" style="margin-top: 10px;">
                                <div class="stat-title">${gpu.name}</div>
                                <div style="font-size: 0.9em; color: #718096;">
                                    使用率: ${gpu.utilization}% | 显存: ${gpu.memory_percent}%
                                </div>
                            </div>
                        `;
                    });
                    gpuContainer.innerHTML = gpuHTML;
                }
            } catch (error) {
                console.error('更新系统状态失败:', error);
            }
        }

        // 页面加载时启动
        document.addEventListener('DOMContentLoaded', function() {
            updateSystemStats();
            setInterval(updateSystemStats, 5000); // 每5秒更新一次
        });

        // 拖拽上传功能
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.upload-area').forEach(area => {
                area.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    this.classList.add('dragover');
                });
    
                area.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    this.classList.remove('dragover');
                });
    
                area.addEventListener('drop', function(e) {
                    e.preventDefault();
                    this.classList.remove('dragover');
    
                    const files = e.dataTransfer.files;
                    const tabId = this.closest('.tab-content').id;
    
                    if (tabId === 'pdf-tab' && files.length > 0) {
                        document.getElementById('pdf-file').files = files;
                        handleFileSelect({files: files}, 'pdf');
                    } else if (tabId === 'json-tab' && files.length > 0) {
                        document.getElementById('json-file').files = files;
                        handleFileSelect({files: files}, 'json');
                    } else if (tabId === 'images-tab' && files.length > 0) {
                        document.getElementById('images-folder').files = files;
                        handleFileSelect({files: files}, 'images');
                    }
                });
            });
        });
    </script>
</body>
</html>
"""

# 智能对话页面HTML模板
CHAT_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能对话系统 - 多模态智能教学系统</title>
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
            text-align: center;
        }

        .header h1 {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
        }

        .nav-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            font-weight: bold;
            transition: transform 0.3s ease;
        }

        .nav-btn:hover {
            transform: translateY(-2px);
        }

        .nav-btn.active {
            background: linear-gradient(135deg, #764ba2, #667eea);
        }

        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
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

        .chat-container {
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

        .input-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .voice-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .voice-btn {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .voice-btn:hover {
            transform: scale(1.05);
        }

        .voice-btn.recording {
            background: linear-gradient(135deg, #e53e3e, #c53030);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }

        .voice-status {
            font-size: 0.9em;
            color: #718096;
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

        .emotion-display {
            background: #f8fafc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }

        .emotion-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .emotion-item {
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 5px;
            border: 1px solid #e2e8f0;
        }

        .emotion-name {
            font-size: 0.9em;
            font-weight: bold;
            color: #4a5568;
        }

        .emotion-value {
            font-size: 1.1em;
            color: #667eea;
        }

        .learning-states {
            margin-top: 15px;
        }

        .state-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
        }

        .progress-bar {
            width: 100px;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
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

        .recording-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(229, 62, 62, 0.1);
            border-radius: 5px;
            margin-bottom: 10px;
        }

        .recording-dot {
            width: 10px;
            height: 10px;
            background: #e53e3e;
            border-radius: 50%;
            animation: pulse 1s infinite;
        }

        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2em;
            }

            .nav-buttons {
                flex-direction: column;
                align-items: center;
            }

            .input-container {
                flex-direction: column;
                gap: 15px;
            }

            .voice-controls {
                width: 100%;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 页面头部 -->
        <div class="header">
            <h1>🎤 智能对话系统</h1>
            <p>语音输入转文字，情感识别，与AI助教智能对话</p>
            <div class="nav-buttons">
                <a href="/" class="nav-btn">知识图谱</a>
                <a href="/chat" class="nav-btn active">智能对话</a>
            </div>
        </div>

        <div class="main-grid">
            <!-- 对话界面 -->
            <div class="panel chat-container">
                <h2>💬 智能对话</h2>

                <div class="connection-status">
                    <div class="status-dot" id="statusDot"></div>
                    <span id="statusText">未连接到AI服务器</span>
                    <button class="btn btn-secondary" onclick="connectToServer()" style="margin-left: auto;">连接服务器</button>
                </div>

                <div class="recording-indicator" id="recordingIndicator">
                    <div class="recording-dot"></div>
                    <span>正在录音... 点击停止</span>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <div class="message system">
                        <div>欢迎使用智能对话系统！</div>
                        <div>您可以使用语音输入或文字输入与AI助教对话</div>
                        <div class="timestamp">系统启动时间: <span id="startTime"></span></div>
                    </div>
                </div>

                <div class="input-container">
                    <div class="voice-controls">
                        <button class="voice-btn" id="voiceBtn" onclick="toggleRecording()" title="语音输入">
                            🎤
                        </button>
                        <div class="voice-status" id="voiceStatus">点击开始录音</div>
                    </div>

                    <input type="text" id="messageInput" placeholder="输入您的问题或使用语音输入..." 
                           onkeypress="handleKeyPress(event)">

                    <button class="btn btn-primary" onclick="sendMessage()" id="sendButton">
                        发送
                    </button>
                </div>
            </div>

            <!-- 情感分析面板 -->
            <div class="panel">
                <h2>😊 情感分析</h2>

                <div class="emotion-display">
                    <h3 style="margin-bottom: 10px;">当前情感状态</h3>
                    <div id="currentEmotion" style="font-size: 1.5em; text-align: center; color: #667eea;">
                        中性 😐
                    </div>
                    <div id="emotionConfidence" style="text-align: center; color: #718096; margin-top: 5px;">
                        置信度: 0%
                    </div>
                </div>

                <div class="emotion-grid" id="emotionGrid">
                    <!-- 情感概率分布将在这里动态生成 -->
                </div>

                <div class="learning-states">
                    <h3 style="margin-bottom: 10px;">学习状态评估</h3>
                    <div class="state-item">
                        <span>注意力:</span>
                        <div>
                            <span id="attentionValue">50%</span>
                            <div class="progress-bar">
                                <div class="progress-fill" id="attentionBar" style="width: 50%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="state-item">
                        <span>参与度:</span>
                        <div>
                            <span id="engagementValue">50%</span>
                            <div class="progress-bar">
                                <div class="progress-fill" id="engagementBar" style="width: 50%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="state-item">
                        <span>理解度:</span>
                        <div>
                            <span id="understandingValue">50%</span>
                            <div class="progress-bar">
                                <div class="progress-fill" id="understandingBar" style="width: 50%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        
        class VoiceChatSystem {
            constructor() {
                this.websocket = null;
                this.isConnected = false;
                this.isRecording = false;
                this.mediaRecorder = null;
                this.audioChunks = [];
                this.recognition = null;
                this.startTime = new Date();

                this.initializeUI();
                this.initializeSpeechRecognition();
            }

            initializeUI() {
                document.getElementById('startTime').textContent = this.startTime.toLocaleString();
                this.updateConnectionStatus();
            }

            initializeSpeechRecognition() {
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    this.recognition = new SpeechRecognition();

                    this.recognition.continuous = false;
                    this.recognition.interimResults = false;
                    this.recognition.lang = 'zh-CN';

                    this.recognition.onstart = () => {
                        document.getElementById('voiceStatus').textContent = '正在识别语音...';
                    };

                    this.recognition.onresult = (event) => {
                        const transcript = event.results[0][0].transcript;
                        document.getElementById('messageInput').value = transcript;
                        document.getElementById('voiceStatus').textContent = `识别结果: ${transcript}`;

                        // 自动发送识别的文字
                        setTimeout(() => {
                            this.sendMessage();
                        }, 1000);
                    };

                    this.recognition.onerror = (event) => {
                        document.getElementById('voiceStatus').textContent = `语音识别错误: ${event.error}`;
                        this.stopRecording();
                    };

                    this.recognition.onend = () => {
                        this.stopRecording();
                    };
                } else {
                    document.getElementById('voiceStatus').textContent = '浏览器不支持语音识别';
                }
            }

            async connectToServer() {
                const serverUrl = 'ws://localhost:8765';

                try {
                    this.addSystemMessage('正在连接到AI服务器...');

                    this.websocket = new WebSocket(serverUrl);

                    this.websocket.onopen = () => {
                        this.isConnected = true;
                        this.updateConnectionStatus();
                        this.addSystemMessage('已成功连接到AI服务器！');
                    };

                    this.websocket.onmessage = (event) => {
                        this.handleServerMessage(JSON.parse(event.data));
                    };

                    this.websocket.onclose = () => {
                        this.isConnected = false;
                        this.updateConnectionStatus();
                        this.addSystemMessage('与AI服务器的连接已断开');
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

            toggleRecording() {
                if (!this.isRecording) {
                    this.startRecording();
                } else {
                    this.stopRecording();
                }
            }

            startRecording() {
                if (!this.recognition) {
                    document.getElementById('voiceStatus').textContent = '语音识别不可用';
                    return;
                }

                this.isRecording = true;
                document.getElementById('voiceBtn').classList.add('recording');
                document.getElementById('voiceBtn').textContent = '⏹️';
                document.getElementById('recordingIndicator').style.display = 'flex';
                document.getElementById('voiceStatus').textContent = '正在录音...';

                try {
                    this.recognition.start();
                } catch (error) {
                    console.error('录音启动失败:', error);
                    this.stopRecording();
                }
            }

            stopRecording() {
                this.isRecording = false;
                document.getElementById('voiceBtn').classList.remove('recording');
                document.getElementById('voiceBtn').textContent = '🎤';
                document.getElementById('recordingIndicator').style.display = 'none';

                if (this.recognition) {
                    this.recognition.stop();
                }

                if (document.getElementById('voiceStatus').textContent === '正在录音...') {
                    document.getElementById('voiceStatus').textContent = '点击开始录音';
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

                // 发送到服务器进行情感分析和对话生成
                const messageData = {
                    type: 'text',
                    id: this.generateMessageId(),
                    data: {
                        text: message,
                        context: {
                            timestamp: Date.now(),
                            session_id: 'voice_chat_session',
                            analyze_emotion: true
                        }
                    }
                };

                this.websocket.send(JSON.stringify(messageData));
            }

            handleServerMessage(data) {
                console.log('收到服务器消息:', data);

                if (data.type === 'text_result') {
                    // AI回复消息
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

            updateEmotionDisplay(emotionData) {
                if (!emotionData) return;

                // 更新主要情感显示
                if (emotionData.emotion) {
                    const emotionIcons = {
                        '愤怒': '😠', '厌恶': '🤢', '恐惧': '😨',
                        '喜悦': '😊', '中性': '😐', '悲伤': '😢', '惊讶': '😲'
                    };

                    const icon = emotionIcons[emotionData.emotion] || '😐';
                    document.getElementById('currentEmotion').textContent = `${emotionData.emotion} ${icon}`;
                }

                if (emotionData.confidence) {
                    document.getElementById('emotionConfidence').textContent = 
                        `置信度: ${(emotionData.confidence * 100).toFixed(1)}%`;
                }

                // 更新情感概率分布
                if (emotionData.emotions) {
                    const emotionGrid = document.getElementById('emotionGrid');
                    emotionGrid.innerHTML = '';

                    Object.entries(emotionData.emotions).forEach(([emotion, value]) => {
                        const percentage = (value * 100).toFixed(1);
                        const emotionItem = document.createElement('div');
                        emotionItem.className = 'emotion-item';
                        emotionItem.innerHTML = `
                            <div class="emotion-name">${emotion}</div>
                            <div class="emotion-value">${percentage}%</div>
                        `;
                        emotionGrid.appendChild(emotionItem);
                    });
                }

                // 更新学习状态
                if (emotionData.learning_states) {
                    Object.entries(emotionData.learning_states).forEach(([state, value]) => {
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

                if (this.isConnected) {
                    statusDot.className = 'status-dot connected';
                    statusText.textContent = '已连接到AI服务器';
                } else {
                    statusDot.className = 'status-dot disconnected';
                    statusText.textContent = '未连接到AI服务器';
                }
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
        let voiceChatSystem;
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            voiceChatSystem = new VoiceChatSystem();
            
            // 绑定全局函数
            window.connectToServer = () => voiceChatSystem.connectToServer();
            window.toggleRecording = () => voiceChatSystem.toggleRecording();
            window.sendMessage = () => voiceChatSystem.sendMessage();
            window.handleKeyPress = (event) => voiceChatSystem.handleKeyPress(event);
        });
            
        // 页面卸载时清理连接
        window.addEventListener('beforeunload', function() {
            if (voiceChatSystem && voiceChatSystem.websocket) {
                voiceChatSystem.websocket.close();
            }
        });
    </script>
</body>
</html>
"""

# 处理状态存储
processing_status = {}


@app.route('/')
def knowledge_graph_page():
    """知识图谱生成页面"""
    return render_template_string(KNOWLEDGE_GRAPH_TEMPLATE)


@app.route('/chat')
def chat_page():
    """智能对话页面"""
    return render_template_string(CHAT_TEMPLATE)


@app.route('/api/process_pdf', methods=['POST'])
def process_pdf():
    """处理PDF文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, ALLOWED_PDF_EXTENSIONS):
            return jsonify({'error': 'Invalid file type'}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 获取参数
        domain = request.form.get('domain', '计算机科学')
        batch_size = int(request.form.get('batch_size', 10))
        import_neo4j = request.form.get('import_neo4j') == 'true'

        # Neo4j配置
        neo4j_config = {
            'uri': request.form.get('neo4j_uri', 'bolt://localhost:7687'),
            'user': request.form.get('neo4j_user', 'neo4j'),
            'password': request.form.get('neo4j_password', 'admin123')
        }

        # 生成处理ID
        process_id = f"pdf_{int(time.time())}"
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '开始处理PDF文件...'
        }

        # 启动后台处理
        thread = threading.Thread(
            target=process_pdf_background,
            args=(process_id, filepath, domain, batch_size, import_neo4j, neo4j_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'process_id': process_id, 'message': 'PDF processing started'})

    except Exception as e:
        logger.error(f"处理PDF时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_images', methods=['POST'])
def process_images():
    """处理图片文件"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        # 创建临时图片文件夹
        timestamp = int(time.time())
        images_folder = UPLOAD_DIR / f"images_{timestamp}"
        images_folder.mkdir(exist_ok=True)

        # 保存所有图片文件
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = secure_filename(file.filename)
                filepath = images_folder / filename
                file.save(str(filepath))
                saved_files.append(str(filepath))

        if not saved_files:
            return jsonify({'error': 'No valid image files'}), 400

        # 获取参数
        domain = request.form.get('domain', '计算机科学')
        import_neo4j = request.form.get('import_neo4j') == 'true'

        neo4j_config = {
            'uri': request.form.get('neo4j_uri', 'bolt://localhost:7687'),
            'user': request.form.get('neo4j_user', 'neo4j'),
            'password': request.form.get('neo4j_password', 'admin123')
        }

        # 生成处理ID
        process_id = f"images_{timestamp}"
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': f'开始处理 {len(saved_files)} 个图片文件...'
        }

        # 启动后台处理
        thread = threading.Thread(
            target=process_images_background,
            args=(process_id, str(images_folder), domain, import_neo4j, neo4j_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'process_id': process_id, 'message': 'Images processing started'})

    except Exception as e:
        logger.error(f"处理图片时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_json', methods=['POST'])
def process_json():
    """处理JSON文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, ALLOWED_JSON_EXTENSIONS):
            return jsonify({'error': 'Invalid file type'}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 获取参数
        domain = request.form.get('domain', '计算机科学')
        import_neo4j = request.form.get('import_neo4j') == 'true'

        neo4j_config = {
            'uri': request.form.get('neo4j_uri', 'bolt://localhost:7687'),
            'user': request.form.get('neo4j_user', 'neo4j'),
            'password': request.form.get('neo4j_password', 'admin123')
        }

        # 生成处理ID
        process_id = f"json_{int(time.time())}"
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': '开始处理JSON文件...'
        }

        # 启动后台处理
        thread = threading.Thread(
            target=process_json_background,
            args=(process_id, filepath, domain, import_neo4j, neo4j_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({'process_id': process_id, 'message': 'JSON processing started'})

    except Exception as e:
        logger.error(f"处理JSON时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/process_status/<process_id>')
def get_process_status(process_id):
    """获取处理状态"""
    if process_id in processing_status:
        return jsonify(processing_status[process_id])
    else:
        return jsonify({'error': 'Process not found'}), 404


@app.route('/api/test_neo4j', methods=['POST'])
def test_neo4j():
    """测试Neo4j连接"""
    try:
        data = request.get_json()
        uri = data.get('uri', 'bolt://localhost:7687')
        user = data.get('user', 'neo4j')
        password = data.get('password', 'admin123')

        # 这里应该导入并测试Neo4j连接
        # 由于依赖问题，这里简化处理
        try:
            from py2neo import Graph
            graph = Graph(uri, auth=(user, password))
            # 执行一个简单的查询来测试连接
            graph.run("RETURN 1")
            return jsonify({'success': True, 'message': 'Neo4j连接成功'})
        except ImportError:
            return jsonify({'success': False, 'error': 'py2neo未安装'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/system_stats')
def get_system_stats():
    """获取系统统计信息API"""
    return jsonify(system_monitor.get_stats())


# 后台处理函数
# 在 ai_server/web_monitor_integrated.py 文件中修改后台处理函数

def process_pdf_background(process_id, filepath, domain, batch_size, import_neo4j, neo4j_config):
    """后台处理PDF - 支持控制台实时显示"""
    try:
        processing_status[process_id].update({
            'progress': 20,
            'message': '正在执行OCR文本提取...'
        })

        # 构建命令
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "knowledge_extractor" / "knowledge_extractor_integrated.py"),
            "--pdf", filepath,
            "--domain", domain,
            "--batch-size", str(batch_size),
            "--output", f"knowledge_graph_{process_id}.json"
        ]

        if import_neo4j:
            cmd.extend([
                "--import-neo4j",
                "--neo4j-uri", neo4j_config['uri'],
                "--neo4j-user", neo4j_config['user'],
                "--neo4j-password", neo4j_config['password']
            ])

        processing_status[process_id].update({
            'progress': 40,
            'message': '正在进行知识提取...'
        })

        print(f"\n🚀 开始处理PDF: {filepath}")
        print(f"📋 处理参数: 领域={domain}, 批次={batch_size}")

        # 关键修改：使用实时输出模式
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # 行缓冲
        )

        # 实时读取并显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # 直接打印到控制台
                print(output.strip())

                # 根据输出内容更新进度
                if "批次" in output and "完成" in output:
                    # 提取批次信息更新进度
                    try:
                        if "批次" in output:
                            current_progress = min(80, processing_status[process_id]['progress'] + 2)
                            processing_status[process_id].update({
                                'progress': current_progress,
                                'message': output.strip()
                            })
                    except:
                        pass

        # 等待进程完成
        return_code = process.poll()

        if return_code == 0:
            print(f"✅ PDF处理完成: {filepath}")
            processing_status[process_id].update({
                'status': 'completed',
                'progress': 100,
                'message': '处理完成！',
                'result': {
                    'concepts_count': 'N/A',
                    'relations_count': 'N/A',
                    'neo4j_imported': import_neo4j
                }
            })
        else:
            print(f"❌ PDF处理失败，返回码: {return_code}")
            processing_status[process_id].update({
                'status': 'error',
                'error': f'Processing failed with return code {return_code}'
            })

    except Exception as e:
        print(f"❌ 处理过程中出现异常: {e}")
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })

def process_images_background(process_id, images_folder, domain, import_neo4j, neo4j_config):
    """后台处理图片 - 支持控制台实时显示"""
    try:
        processing_status[process_id].update({
            'progress': 20,
            'message': '正在执行图片OCR提取...'
        })

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "knowledge_extractor" / "knowledge_extractor_integrated.py"),
            "--images", images_folder,
            "--domain", domain,
            "--output", f"knowledge_graph_{process_id}.json"
        ]

        if import_neo4j:
            cmd.extend([
                "--import-neo4j",
                "--neo4j-uri", neo4j_config['uri'],
                "--neo4j-user", neo4j_config['user'],
                "--neo4j-password", neo4j_config['password']
            ])

        processing_status[process_id].update({
            'progress': 60,
            'message': '正在进行知识提取...'
        })

        print(f"\n🚀 开始处理图片文件夹: {images_folder}")
        print(f"📋 处理参数: 领域={domain}")

        # 同样使用实时输出模式
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时读取并显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

                # 更新进度
                if "图片OCR" in output or "知识提取" in output:
                    try:
                        current_progress = min(90, processing_status[process_id]['progress'] + 3)
                        processing_status[process_id].update({
                            'progress': current_progress,
                            'message': output.strip()
                        })
                    except:
                        pass

        return_code = process.poll()

        if return_code == 0:
            print(f"✅ 图片处理完成: {images_folder}")
            processing_status[process_id].update({
                'status': 'completed',
                'progress': 100,
                'message': '处理完成！',
                'result': {
                    'concepts_count': 'N/A',
                    'relations_count': 'N/A',
                    'neo4j_imported': import_neo4j
                }
            })
        else:
            print(f"❌ 图片处理失败，返回码: {return_code}")
            processing_status[process_id].update({
                'status': 'error',
                'error': f'Processing failed with return code {return_code}'
            })

    except Exception as e:
        print(f"❌ 处理过程中出现异常: {e}")
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })

def process_json_background(process_id, filepath, domain, import_neo4j, neo4j_config):
    """后台处理JSON - 支持控制台实时显示"""
    try:
        processing_status[process_id].update({
            'progress': 30,
            'message': '正在进行知识提取...'
        })

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "knowledge_extractor" / "knowledge_extractor_integrated.py"),
            "--json", filepath,
            "--domain", domain,
            "--output", f"knowledge_graph_{process_id}.json"
        ]

        if import_neo4j:
            cmd.extend([
                "--import-neo4j",
                "--neo4j-uri", neo4j_config['uri'],
                "--neo4j-user", neo4j_config['user'],
                "--neo4j-password", neo4j_config['password']
            ])

        processing_status[process_id].update({
            'progress': 70,
            'message': '正在生成知识图谱...'
        })

        print(f"\n🚀 开始处理JSON文件: {filepath}")
        print(f"📋 处理参数: 领域={domain}")

        # 同样使用实时输出模式
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时读取并显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

                # 更新进度
                if "知识提取" in output or "图谱生成" in output:
                    try:
                        current_progress = min(95, processing_status[process_id]['progress'] + 5)
                        processing_status[process_id].update({
                            'progress': current_progress,
                            'message': output.strip()
                        })
                    except:
                        pass

        return_code = process.poll()

        if return_code == 0:
            print(f"✅ JSON处理完成: {filepath}")
            processing_status[process_id].update({
                'status': 'completed',
                'progress': 100,
                'message': '处理完成！',
                'result': {
                    'concepts_count': 'N/A',
                    'relations_count': 'N/A',
                    'neo4j_imported': import_neo4j
                }
            })
        else:
            print(f"❌ JSON处理失败，返回码: {return_code}")
            processing_status[process_id].update({
                'status': 'error',
                'error': f'Processing failed with return code {return_code}'
            })

    except Exception as e:
        print(f"❌ 处理过程中出现异常: {e}")
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })

# 如果你想要更详细的进度解析，可以添加这个辅助函数
def parse_and_update_progress(output_line, process_id):
    """解析输出行并更新详细进度"""
    try:
        line = output_line.strip()

        # 解析进度条信息
        if "[进度]" in line and "%" in line:
            # 提取百分比
            import re
            percentage_match = re.search(r'(\d+)%', line)
            if percentage_match:
                percentage = int(percentage_match.group(1))
                # 将OCR进度映射到总体进度 (OCR占40-80%)
                total_progress = 40 + int(percentage * 0.4)
                processing_status[process_id].update({
                    'progress': min(total_progress, 80),
                    'message': f'OCR进度: {percentage}%'
                })

        # 解析批次完成信息
        elif "批次" in line and "完成" in line:
            # 提取批次信息
            batch_match = re.search(r'批次(\d+)', line)
            if batch_match:
                batch_num = int(batch_match.group(1))
                processing_status[process_id].update({
                    'message': f'已完成批次 {batch_num}'
                })

        # 解析知识提取进度
        elif "知识提取" in line:
            processing_status[process_id].update({
                'progress': 85,
                'message': '正在提取知识点...'
            })

        elif "Neo4j" in line:
            processing_status[process_id].update({
                'progress': 95,
                'message': '正在导入Neo4j数据库...'
            })

    except Exception as e:
        pass  # 忽略解析错误

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - 增强版Web界面")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="Web服务器端口号")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    logger.info(f"启动增强版Web界面: {args.host}:{args.port}")

    # 启动系统监控
    system_monitor.start_monitoring()

    try:
        logger.info("正在启动Flask Web服务器...")
        print(f"\n🌐 多模态智能教学系统 - 增强版Web界面")
        print(f"📍 知识图谱页面: http://{args.host}:{args.port}")
        print(f"📍 智能对话页面: http://{args.host}:{args.port}/chat")
        print(f"🔧 系统监控已启动")
        print(f"💡 使用说明:")
        print(f"   1. 访问知识图谱页面上传文件生成知识图谱")
        print(f"   2. 访问智能对话页面进行语音对话")
        print(f"   3. 按 Ctrl+C 退出\n")

        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False
        )

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务...")
    except Exception as e:
        logger.error(f"启动Web服务器时出错: {e}")
    finally:
        logger.info("增强版Web界面已关闭")


if __name__ == "__main__":
    main()