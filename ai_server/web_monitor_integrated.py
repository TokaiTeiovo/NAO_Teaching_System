#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态智能教学系统 - Web API后端
提供RESTful API接口，支持前后端分离架构
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
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 定义路径常量
LOG_DIR = PROJECT_ROOT / "logs"
UPLOAD_DIR = PROJECT_ROOT / "uploads"
SHARED_DIR = PROJECT_ROOT / "shared"
STATIC_DIR = PROJECT_ROOT / "static"

# 确保目录存在
for directory in [LOG_DIR, UPLOAD_DIR, SHARED_DIR, STATIC_DIR]:
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
CORS(app)  # 启用跨域支持
app.config['SECRET_KEY'] = 'multimodal_teaching_system_2025'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 全局变量
system_monitor = SystemMonitor()
logger = setup_logger('web_monitor', log_file='web_monitor.log')
processing_status = {}
processing_processes = {}  # 存储正在运行的进程


# ==================== 静态文件服务 ====================
@app.route('/')
def serve_index():
    """提供首页静态文件"""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/chat')
def serve_chat():
    """提供聊天页面静态文件"""
    return send_from_directory(STATIC_DIR, 'chat.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """提供其他静态文件"""
    return send_from_directory(STATIC_DIR, filename)


# ==================== API 路由 ====================
@app.route('/api/system_stats')
def get_system_stats():
    """获取系统统计信息API"""
    return jsonify(system_monitor.get_stats())


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
        max_pages = request.form.get('max_pages')
        if max_pages:
            max_pages = int(max_pages)
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
            'message': '开始处理PDF文件...',
            'file_type': 'pdf'
        }

        # 启动后台处理
        thread = threading.Thread(
            target=process_pdf_background,
            args=(process_id, filepath, domain, batch_size, max_pages, import_neo4j, neo4j_config)
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
            'message': f'开始处理 {len(saved_files)} 个图片文件...',
            'file_type': 'images'
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
            'message': '开始处理JSON文件...',
            'file_type': 'json'
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


@app.route('/api/stop_process/<process_id>', methods=['POST'])
def stop_process(process_id):
    """停止处理任务"""
    try:
        if process_id not in processing_status:
            return jsonify({'error': 'Process not found'}), 404

        # 更新状态为已停止
        processing_status[process_id].update({
            'status': 'stopped',
            'message': '用户手动停止处理'
        })

        # 如果有正在运行的进程，终止它
        if process_id in processing_processes:
            process = processing_processes[process_id]
            try:
                if process.poll() is None:  # 进程还在运行
                    process.terminate()
                    logger.info(f"已终止处理进程: {process_id} (PID: {process.pid})")

                    # 等待进程终止，如果超时则强制杀死
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        logger.warning(f"强制杀死进程: {process_id} (PID: {process.pid})")

                del processing_processes[process_id]
            except Exception as e:
                logger.error(f"终止进程时出错: {e}")

        logger.info(f"处理任务已停止: {process_id}")
        return jsonify({'success': True, 'message': '处理任务已停止'})

    except Exception as e:
        logger.error(f"停止处理任务时出错: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/validate_image_path', methods=['POST'])
def validate_image_path():
    """验证图片文件夹路径"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '').strip()

        if not folder_path:
            return jsonify({'success': False, 'error': '请提供文件夹路径'}), 400

        folder_path = Path(folder_path)

        if not folder_path.exists():
            return jsonify({'success': False, 'error': '文件夹路径不存在'}), 400

        if not folder_path.is_dir():
            return jsonify({'success': False, 'error': '路径不是一个文件夹'}), 400

        # 支持的图片格式
        supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

        # 获取所有图片文件
        image_files = []
        format_stats = {}
        total_size = 0

        for ext in supported_formats:
            files = list(folder_path.glob(f"**/*{ext}"))
            files.extend(list(folder_path.glob(f"**/*{ext.upper()}")))
            image_files.extend(files)

        # 去重并统计
        image_files = list(set(image_files))

        for img_file in image_files:
            ext = img_file.suffix.lower()
            format_stats[ext] = format_stats.get(ext, 0) + 1
            try:
                total_size += img_file.stat().st_size
            except:
                pass

        if len(image_files) == 0:
            return jsonify({'success': False, 'error': '文件夹中未找到支持的图片文件'}), 400

        info = {
            'total_images': len(image_files),
            'format_stats': format_stats,
            'total_size_mb': total_size / (1024 * 1024),
            'folder_path': str(folder_path)
        }

        return jsonify({'success': True, 'info': info})

    except Exception as e:
        logger.error(f"验证图片路径时出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/process_images_folder', methods=['POST'])
def process_images_folder():
    """处理图片文件夹（通过路径）"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path', '').strip()

        if not folder_path:
            return jsonify({'error': '请提供文件夹路径'}), 400

        folder_path = Path(folder_path)

        if not folder_path.exists() or not folder_path.is_dir():
            return jsonify({'error': '文件夹路径无效'}), 400

        # 获取参数
        domain = data.get('domain', '计算机科学')
        batch_size = data.get('batch_size', 10)
        max_count = data.get('max_count')
        use_gpu = data.get('use_gpu', True)
        import_neo4j = data.get('import_neo4j', True)
        neo4j_config = data.get('neo4j_config', {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'admin123'
        })

        # 验证文件夹并统计图片数量
        supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
        image_files = []

        for ext in supported_formats:
            files = list(folder_path.glob(f"**/*{ext}"))
            files.extend(list(folder_path.glob(f"**/*{ext.upper()}")))
            image_files.extend(files)

        image_files = list(set(image_files))  # 去重

        if not image_files:
            return jsonify({'error': '文件夹中未找到支持的图片文件'}), 400

        # 生成处理ID
        process_id = f"images_folder_{int(time.time())}"
        processing_status[process_id] = {
            'status': 'processing',
            'progress': 0,
            'message': f'开始处理文件夹中的 {len(image_files)} 张图片...',
            'file_type': 'images'
        }

        # 启动后台处理
        thread = threading.Thread(
            target=process_images_folder_background,
            args=(process_id, str(folder_path), domain, batch_size, max_count, import_neo4j, neo4j_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'process_id': process_id,
            'message': 'Images folder processing started',
            'total_images': len(image_files)
        })

    except Exception as e:
        logger.error(f"处理图片文件夹时出错: {e}")
        return jsonify({'error': str(e)}), 500


def process_images_folder_background(process_id, folder_path, domain, batch_size, use_gpu, max_count, import_neo4j,
                                     neo4j_config):
    """后台处理图片文件夹"""
    try:
        update_progress(process_id, 10, '图片文件夹验证完成')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        # 构建命令
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "knowledge_extractor" / "knowledge_extractor_integrated.py"),
            "--images", folder_path,
            "--domain", domain,
            "--batch-size", str(batch_size),
            "--output", f"knowledge_graph_{process_id}.json"
        ]

        if max_count:
            cmd.extend(["--max-items", str(max_count)])

        if import_neo4j:
            cmd.extend([
                "--import-neo4j",
                "--neo4j-uri", neo4j_config['uri'],
                "--neo4j-user", neo4j_config['user'],
                "--neo4j-password", neo4j_config['password']
            ])

        update_progress(process_id, 30, '开始图片OCR文本提取...')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'  # 强制使用UTF-8编码

        if use_gpu:
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一个GPU
            update_progress(process_id, 35, '启用GPU加速模式...')

        # 执行处理
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # 明确指定UTF-8编码
            errors='ignore',   # 忽略编码错误
            env=env           # 传递环境变量
        )

        # 存储进程引用
        processing_processes[process_id] = process

        # 等待进程完成
        try:
            stdout, stderr = process.communicate(timeout=3600)

            # 检查是否被手动停止
            if processing_status.get(process_id, {}).get('status') == 'stopped':
                return

            if process.returncode == 0:
                # 尝试从输出中解析统计信息
                concepts_count = 'N/A'
                relations_count = 'N/A'

                # 简单的正则表达式匹配输出中的统计信息
                import re
                if stdout:
                    concept_match = re.search(r'提取概念.*?(\d+).*?个', stdout)
                    relation_match = re.search(r'提取关系.*?(\d+).*?个', stdout)
                    if concept_match:
                        concepts_count = concept_match.group(1)
                    if relation_match:
                        relations_count = relation_match.group(1)

                processing_status[process_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'message': '处理完成！',
                    'result': {
                        'concepts_count': concepts_count,
                        'relations_count': relations_count,
                        'neo4j_imported': import_neo4j
                    }
                })
            else:
                error_msg = stderr if stderr else 'Processing failed'
                try:
                    if isinstance(error_msg, bytes):
                        error_msg = error_msg.decode('utf-8', errors='ignore')
                except:
                    error_msg = 'Processing failed with encoding error'

                processing_status[process_id].update({
                    'status': 'error',
                    'error': error_msg[:500]  # 限制错误消息长度
                })
        except subprocess.TimeoutExpired:
            process.kill()
            processing_status[process_id].update({
                'status': 'error',
                'error': '处理超时'
            })
        finally:
            # 清理进程引用
            if process_id in processing_processes:
                del processing_processes[process_id]

    except Exception as e:
        error_msg = str(e)
        # 处理异常中的编码问题
        try:
            if isinstance(e, UnicodeDecodeError):
                error_msg = '文件路径或文件名包含不支持的字符，请使用英文路径'
        except:
            error_msg = '处理过程中出现未知错误'

        processing_status[process_id].update({
            'status': 'error',
            'error': error_msg[:500]
        })
        # 清理进程引用
        if process_id in processing_processes:
            del processing_processes[process_id]

def test_neo4j():
    """测试Neo4j连接"""
    try:
        data = request.get_json()
        uri = data.get('uri', 'bolt://localhost:7687')
        user = data.get('user', 'neo4j')
        password = data.get('password', 'admin123')

        try:
            from py2neo import Graph
            graph = Graph(uri, auth=(user, password))
            graph.run("RETURN 1")
            return jsonify({'success': True, 'message': 'Neo4j连接成功'})
        except ImportError:
            return jsonify({'success': False, 'error': 'py2neo未安装'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 后台处理函数 ====================
def update_progress(process_id, progress, message):
    """更新处理进度"""
    if process_id in processing_status:
        processing_status[process_id].update({
            'progress': progress,
            'message': message
        })


def process_pdf_background(process_id, filepath, domain, batch_size, max_pages, import_neo4j, neo4j_config):
    """后台处理PDF"""
    try:
        update_progress(process_id, 10, '文件验证完成')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        # 构建命令
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "knowledge_extractor" / "knowledge_extractor_integrated.py"),
            "--pdf", filepath,
            "--domain", domain,
            "--batch-size", str(batch_size),
            "--output", f"knowledge_graph_{process_id}.json"
        ]

        if max_pages:
            cmd.extend(["--max-items", str(max_pages)])

        if import_neo4j:
            cmd.extend([
                "--import-neo4j",
                "--neo4j-uri", neo4j_config['uri'],
                "--neo4j-user", neo4j_config['user'],
                "--neo4j-password", neo4j_config['password']
            ])

        update_progress(process_id, 20, 'PDF转图像处理中...')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        # 执行处理
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            text=True
        )

        # 存储进程引用
        processing_processes[process_id] = process

        # 等待进程完成
        try:
            stdout, stderr = process.communicate(timeout=3600)

            # 检查是否被手动停止
            if processing_status.get(process_id, {}).get('status') == 'stopped':
                return

            if process.returncode == 0:
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
                error_msg = stderr if stderr else 'Processing failed'
                processing_status[process_id].update({
                    'status': 'error',
                    'error': error_msg
                })
        except subprocess.TimeoutExpired:
            process.kill()
            processing_status[process_id].update({
                'status': 'error',
                'error': '处理超时'
            })
        finally:
            # 清理进程引用
            if process_id in processing_processes:
                del processing_processes[process_id]

    except Exception as e:
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })
        # 清理进程引用
        if process_id in processing_processes:
            del processing_processes[process_id]


def process_images_background(process_id, images_folder, domain, import_neo4j, neo4j_config):
    """后台处理图片"""
    try:
        update_progress(process_id, 10, '图片文件验证完成')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

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

        update_progress(process_id, 30, '图片OCR文本提取中...')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        # 执行处理
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            text=True
        )

        # 存储进程引用
        processing_processes[process_id] = process

        # 等待进程完成
        try:
            stdout, stderr = process.communicate(timeout=3600)

            # 检查是否被手动停止
            if processing_status.get(process_id, {}).get('status') == 'stopped':
                return

            if process.returncode == 0:
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
                error_msg = stderr if stderr else 'Processing failed'
                processing_status[process_id].update({
                    'status': 'error',
                    'error': error_msg
                })
        except subprocess.TimeoutExpired:
            process.kill()
            processing_status[process_id].update({
                'status': 'error',
                'error': '处理超时'
            })
        finally:
            # 清理进程引用
            if process_id in processing_processes:
                del processing_processes[process_id]

    except Exception as e:
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })
        # 清理进程引用
        if process_id in processing_processes:
            del processing_processes[process_id]


def process_json_background(process_id, filepath, domain, import_neo4j, neo4j_config):
    """后台处理JSON"""
    try:
        update_progress(process_id, 15, 'JSON文件验证完成')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

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

        update_progress(process_id, 40, 'LLM知识提取中...')

        # 检查是否被停止
        if processing_status.get(process_id, {}).get('status') == 'stopped':
            return

        # 执行处理
        process = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            text=True
        )

        # 存储进程引用
        processing_processes[process_id] = process

        # 等待进程完成
        try:
            stdout, stderr = process.communicate(timeout=1800)

            # 检查是否被手动停止
            if processing_status.get(process_id, {}).get('status') == 'stopped':
                return

            if process.returncode == 0:
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
                error_msg = stderr if stderr else 'Processing failed'
                processing_status[process_id].update({
                    'status': 'error',
                    'error': error_msg
                })
        except subprocess.TimeoutExpired:
            process.kill()
            processing_status[process_id].update({
                'status': 'error',
                'error': '处理超时'
            })
        finally:
            # 清理进程引用
            if process_id in processing_processes:
                del processing_processes[process_id]

    except Exception as e:
        processing_status[process_id].update({
            'status': 'error',
            'error': str(e)
        })
        # 清理进程引用
        if process_id in processing_processes:
            del processing_processes[process_id]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模态智能教学系统 - Web API后端")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口号")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")

    args = parser.parse_args()

    logger.info(f"启动Web API后端: {args.host}:{args.port}")

    # 启动系统监控
    system_monitor.start_monitoring()

    try:
        logger.info("正在启动Flask API服务器...")
        print(f"\n🌐 多模态智能教学系统 - Web API后端")
        print(f"📍 API服务器: http://{args.host}:{args.port}")
        print(f"📍 静态文件: {STATIC_DIR}")
        print(f"🔧 系统监控已启动")
        print(f"💡 使用说明:")
        print(f"   1. 将前端文件放置在 {STATIC_DIR} 目录")
        print(f"   2. 访问 http://{args.host}:{args.port} 查看应用")
        print(f"   3. API接口前缀: /api/")
        print(f"   4. 按 Ctrl+C 退出\n")

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
        logger.error(f"启动API服务器时出错: {e}")
    finally:
        logger.info("Web API后端已关闭")


if __name__ == "__main__":
    main()