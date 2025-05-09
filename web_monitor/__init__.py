# -*- coding: utf-8 -*-
"""
NAO教学系统Web监控模块初始化文件
"""

from flask import Blueprint

# 创建蓝图
web_monitor_bp = Blueprint(
    'web_monitor',
    __name__,
    template_folder='templates',
    static_folder='static'
)

# 导入视图函数，避免循环导入
from . import views