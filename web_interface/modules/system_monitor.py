#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import random
import threading
import time


class SystemMonitor:
    """
    系统监控模块，用于监控服务器和NAO机器人的系统状态
    """

    def __init__(self, config):
        self.config = config

        # 系统信息
        self.system_info = self._get_system_info()

        # 性能指标
        self.metrics = {
            "cpu": [],
            "memory": [],
            "response_time": [],
            "nao_battery": [],
            "nao_temp": [],
            "timestamp": []
        }

        # 最大历史记录数
        self.max_history = 100

        # NAO机器人连接状态
        self.nao_connected = False

        # 上次更新时间
        self.last_update_time = time.time()

        # 更新间隔（秒）
        self.update_interval = 2

        # AI服务器连接状态
        self.ai_server_connected = False
        self.ai_server_start_time = None

        # 启动单独的监控线程
        self.monitor_thread = None
        self.monitoring_active = False

    def _get_system_info(self):
        """获取系统信息"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "hostname": platform.node(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return info

    def _get_cpu_usage(self):
        """获取CPU使用率"""
        # 实际应用中应使用psutil等库获取真实CPU使用率
        # 这里使用模拟数据
        return random.uniform(20, 60)

    def _get_memory_usage(self):
        """获取内存使用率"""
        # 实际应用中应使用psutil等库获取真实内存使用率
        # 这里使用模拟数据
        return random.uniform(30, 70)

    def _get_nao_battery(self):
        """获取NAO机器人电池电量"""
        # 实际应用中应与NAO机器人通信获取真实电量
        # 这里使用模拟数据
        if not self.nao_connected:
            return None

        return max(0, min(100, 70 + random.uniform(-5, 5)))

    def _get_nao_temperature(self):
        """获取NAO机器人温度"""
        # 实际应用中应与NAO机器人通信获取真实温度
        # 这里使用模拟数据
        if not self.nao_connected:
            return None

        return 37 + random.uniform(-2, 2)

    def _get_response_time(self):
        """获取AI服务器响应时间（毫秒）"""
        # 实际应用中应向AI服务器发送请求测试响应时间
        # 这里使用模拟数据
        if not self.ai_server_connected:
            return None

        # 生成100-500ms之间的响应时间，有10%的概率产生较大延迟
        base_time = random.uniform(100, 300)
        if random.random() < 0.1:
            return base_time + random.uniform(200, 700)
        else:
            return base_time

    def start_monitoring(self):
        """
        启动监控线程
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            # 已经在监控中
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_task)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """
        停止监控线程
        """
        self.monitoring_active = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None

    def _monitoring_task(self):
        """
        监控任务线程
        """
        while self.monitoring_active:
            try:
                # 更新性能指标
                self._update_metrics()

                # 预先检测问题
                self._check_for_issues()

                # 控制更新频率
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"监控任务出错: {e}")
                time.sleep(5)  # 出错后等待时间

    def _update_metrics(self):
        """
        更新性能指标
        """
        current_time = time.time()

        # 获取当前指标
        cpu_usage = self._get_cpu_usage()
        memory_usage = self._get_memory_usage()
        response_time = self._get_response_time()
        nao_battery = self._get_nao_battery()
        nao_temp = self._get_nao_temperature()

        # 更新数据
        self.metrics["timestamp"].append(current_time)
        self.metrics["cpu"].append(cpu_usage)
        self.metrics["memory"].append(memory_usage)

        if response_time is not None:
            self.metrics["response_time"].append(response_time)
        elif len(self.metrics["response_time"]) > 0:
            # 复制上一个值
            self.metrics["response_time"].append(self.metrics["response_time"][-1])
        else:
            self.metrics["response_time"].append(0)

        if nao_battery is not None:
            self.metrics["nao_battery"].append(nao_battery)
        elif len(self.metrics["nao_battery"]) > 0:
            self.metrics["nao_battery"].append(self.metrics["nao_battery"][-1])
        else:
            self.metrics["nao_battery"].append(0)

        if nao_temp is not None:
            self.metrics["nao_temp"].append(nao_temp)
        elif len(self.metrics["nao_temp"]) > 0:
            self.metrics["nao_temp"].append(self.metrics["nao_temp"][-1])
        else:
            self.metrics["nao_temp"].append(0)

        # 保持历史记录大小
        if len(self.metrics["timestamp"]) > self.max_history:
            for key in self.metrics:
                self.metrics[key] = self.metrics[key][-self.max_history:]

    def _check_for_issues(self):
        """
        检查系统问题
        """
        issues = []

        # 检查CPU使用率
        if len(self.metrics["cpu"]) > 0 and self.metrics["cpu"][-1] > 80:
            issues.append({
                "type": "warning",
                "component": "cpu",
                "message": f"CPU使用率过高: {self.metrics['cpu'][-1]:.1f}%"
            })

        # 检查内存使用率
        if len(self.metrics["memory"]) > 0 and self.metrics["memory"][-1] > 80:
            issues.append({
                "type": "warning",
                "component": "memory",
                "message": f"内存使用率过高: {self.metrics['memory'][-1]:.1f}%"
            })

        # 检查响应时间
        if len(self.metrics["response_time"]) > 5:
            avg_response = sum(self.metrics["response_time"][-5:]) / 5
            if avg_response > 500:
                issues.append({
                    "type": "warning",
                    "component": "response_time",
                    "message": f"AI服务器响应时间过长: {avg_response:.1f}ms"
                })

        # 检查NAO电池电量
        if len(self.metrics["nao_battery"]) > 0 and self.metrics["nao_battery"][-1] < 20:
            issues.append({
                "type": "warning",
                "component": "nao_battery",
                "message": f"NAO电池电量低: {self.metrics['nao_battery'][-1]:.1f}%"
            })

        # 检查NAO温度
        if len(self.metrics["nao_temp"]) > 0 and self.metrics["nao_temp"][-1] > 45:
            issues.append({
                "type": "warning",
                "component": "nao_temp",
                "message": f"NAO温度过高: {self.metrics['nao_temp'][-1]:.1f}°C"
            })

        # 返回检测到的问题
        return issues

    def get_metrics(self):
        """
        获取当前指标
        """
        # 如果没有数据，更新一次
        if len(self.metrics["timestamp"]) == 0:
            self._update_metrics()

        current_metrics = {
            "cpu": self.metrics["cpu"][-1] if self.metrics["cpu"] else 0,
            "memory": self.metrics["memory"][-1] if self.metrics["memory"] else 0,
            "response_time": self.metrics["response_time"][-1] if self.metrics["response_time"] else 0,
            "nao_battery": self.metrics["nao_battery"][-1] if self.metrics["nao_battery"] else 0,
            "nao_temp": self.metrics["nao_temp"][-1] if self.metrics["nao_temp"] else 0,
            "timestamp": time.time(),
            "uptime": self._get_uptime(),
            "nao_connected": self.nao_connected,
            "ai_server_connected": self.ai_server_connected,
            "issues": self._check_for_issues()
        }

        return current_metrics

    def get_metrics_history(self, period=60):
        """
        获取历史指标

        参数:
            period: 历史数据的时间范围（秒）
        """
        if len(self.metrics["timestamp"]) == 0:
            return self.metrics

        current_time = time.time()
        start_time = current_time - period

        # 找到起始位置
        start_idx = 0
        for i, timestamp in enumerate(self.metrics["timestamp"]):
            if timestamp >= start_time:
                start_idx = i
                break

        # 提取历史数据
        history = {}
        for key in self.metrics:
            history[key] = self.metrics[key][start_idx:]

        return history

    def _get_uptime(self):
        """
        获取系统运行时间
        """
        if not self.ai_server_start_time:
            # 模拟服务器已经运行了一段时间
            self.ai_server_start_time = time.time() - random.uniform(3600, 86400)  # 1小时到1天之间

        uptime_seconds = time.time() - self.ai_server_start_time

        # 格式化为可读时间
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        return {
            "seconds": uptime_seconds,
            "formatted": f"{int(days)}天 {int(hours)}小时 {int(minutes)}分 {int(seconds)}秒"
        }

    def set_nao_connection_status(self, status):
        """
        设置NAO机器人连接状态

        参数:
            status: 连接状态 (True/False)
        """
        self.nao_connected = status

    def set_ai_server_connection_status(self, status):
        """
        设置AI服务器连接状态

        参数:
            status: 连接状态 (True/False)
        """
        self.ai_server_connected = status

        # 如果连接成功且没有开始时间，设置开始时间
        if status and not self.ai_server_start_time:
            self.ai_server_start_time = time.time()

    def process_status(self, process_name):
        """
        检查进程状态

        参数:
            process_name: 进程名称
        """
        try:
            # 在实际应用中，应使用适合的命令检查进程
            # 这里模拟常见进程的状态
            if process_name == "ai_server":
                return {"status": "running", "pid": 12345, "memory": "256MB", "cpu": "2.3%"}
            elif process_name == "websocket_server":
                return {"status": "running", "pid": 12346, "memory": "128MB", "cpu": "1.1%"}
            elif process_name == "nao_client":
                return {"status": "stopped", "error": "Process not found"}
            else:
                return {"status": "unknown", "message": f"未知进程: {process_name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_logs(self, service="all", lines=50):
        """
        获取服务日志

        参数:
            service: 服务名称 (ai_server/nao_client/web/all)
            lines: 返回的日志行数
        """
        # 在实际应用中，应读取真实的日志文件
        # 这里返回模拟数据

        logs = []
        timestamps = [time.time() - i * 10 for i in range(lines)]

        log_levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
        log_weights = [0.7, 0.2, 0.08, 0.02]  # 不同级别日志的权重

        services = []
        if service == "all":
            services = ["ai_server", "nao_client", "web_interface"]
        else:
            services = [service]

        for i in range(lines):
            # 随机选择日志级别
            level = random.choices(log_levels, weights=log_weights)[0]

            # 随机选择服务
            log_service = random.choice(services)

            # 生成模拟日志内容
            if log_service == "ai_server":
                if level == "INFO":
                    content = random.choice([
                        "处理用户查询完成",
                        "模型加载成功",
                        "会话已初始化",
                        "成功连接到Neo4j数据库",
                        "情感分析完成"
                    ])
                elif level == "WARNING":
                    content = random.choice([
                        "模型响应延迟超过阈值",
                        "知识图谱查询超时，使用缓存结果",
                        "情感分析不确定性高"
                    ])
                elif level == "ERROR":
                    content = random.choice([
                        "模型推理出错",
                        "数据库连接失败",
                        "会话初始化失败"
                    ])
                else:
                    content = "模型处理中..."

            elif log_service == "nao_client":
                if level == "INFO":
                    content = random.choice([
                        "动作执行完成",
                        "语音合成完成",
                        "相机初始化成功",
                        "已连接到AI服务器"
                    ])
                elif level == "WARNING":
                    content = random.choice([
                        "电池电量低",
                        "音频质量不佳",
                        "动作执行超时"
                    ])
                elif level == "ERROR":
                    content = random.choice([
                        "动作执行失败",
                        "连接AI服务器失败",
                        "相机初始化失败"
                    ])
                else:
                    content = "处理音频数据..."

            elif log_service == "web_interface":
                if level == "INFO":
                    content = random.choice([
                        "页面访问: /dashboard",
                        "WebSocket连接建立",
                        "用户登录: admin",
                        "数据导出完成"
                    ])
                elif level == "WARNING":
                    content = random.choice([
                        "页面加载超时",
                        "WebSocket重连",
                        "会话过期"
                    ])
                elif level == "ERROR":
                    content = random.choice([
                        "API请求失败",
                        "模板渲染错误",
                        "数据库查询失败"
                    ])
                else:
                    content = "处理HTTP请求..."

            # 构造日志条目
            log_entry = {
                "timestamp": timestamps[i],
                "formatted_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamps[i])),
                "service": log_service,
                "level": level,
                "content": content
            }

            logs.append(log_entry)

        # 按时间倒序排序
        logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return logs

    def start_process(self, process_name):
        """
        启动进程

        参数:
            process_name: 进程名称
        """
        # 在实际应用中，应使用subprocess等启动进程
        # 这里仅返回模拟结果

        if process_name == "ai_server":
            self.ai_server_connected = True
            self.ai_server_start_time = time.time()
            return {"status": "success", "message": "AI服务器已启动"}
        elif process_name == "nao_client":
            self.nao_connected = True
            return {"status": "success", "message": "NAO客户端已启动"}
        else:
            return {"status": "error", "message": f"未知进程: {process_name}"}

    def stop_process(self, process_name):
        """
        停止进程

        参数:
            process_name: 进程名称
        """
        # 在实际应用中，应使用适当方法停止进程
        # 这里仅返回模拟结果

        if process_name == "ai_server":
            self.ai_server_connected = False
            return {"status": "success", "message": "AI服务器已停止"}
        elif process_name == "nao_client":
            self.nao_connected = False
            return {"status": "success", "message": "NAO客户端已停止"}
        else:
            return {"status": "error", "message": f"未知进程: {process_name}"}

    def restart_process(self, process_name):
        """
        重启进程

        参数:
            process_name: 进程名称
        """
        stop_result = self.stop_process(process_name)
        if stop_result["status"] == "error":
            return stop_result

        # 模拟重启延迟
        time.sleep(1)

        return self.start_process(process_name)