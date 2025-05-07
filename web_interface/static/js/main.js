// main.js - NAO教学系统Web界面核心JavaScript功能

// 全局变量
let socket;
let serverConnected = false;
let naoConnected = false;
let currentSession = null;

// DOM元素缓存
const elements = {};

// 初始化函数
document.addEventListener('DOMContentLoaded', function() {
    // 缓存常用DOM元素
    cacheElements();

    // 初始化Socket.IO连接
    initSocketConnection();

    // 初始化事件监听器
    initEventListeners();

    // 初始化UI组件
    initUI();

    // 检查系统状态
    checkSystemStatus();
});

// 缓存常用DOM元素
function cacheElements() {
    elements.serverStatus = document.getElementById('server-status');
    elements.naoStatus = document.getElementById('nao-status');
    elements.startButton = document.getElementById('start-server-button');
    elements.stopButton = document.getElementById('stop-server-button');
    elements.connectButton = document.getElementById('connect-nao-button');
}

// 初始化Socket.IO连接
function initSocketConnection() {
    socket = io();

    socket.on('connect', function() {
        console.log('已连接到WebSocket服务器');
        showToast('成功连接到服务器', 'success');
    });

    socket.on('disconnect', function() {
        console.log('与WebSocket服务器断开连接');
        showToast('与服务器断开连接', 'danger');
        updateServerStatus(false);
    });

    // 系统状态更新
    socket.on('system_update', function(data) {
        console.log('系统状态更新:', data);
        updateServerStatus(data.ai_server_connected);
        updateNaoStatus(data.nao_connected);

        // 更新服务器指标
        if (data.cpu !== undefined) {
            updateSystemMetrics(data);
        }

        // 检查系统警告
        if (data.issues && data.issues.length > 0) {
            showSystemIssues(data.issues);
        }
    });

    // 情感状态更新
    socket.on('emotion_update', function(data) {
        console.log('情感状态更新:', data);
        updateEmotionDisplay(data);
    });

    // 消息处理
    socket.on('message', function(data) {
        console.log('收到消息:', data);
        if (data.type === 'notification') {
            showToast(data.message, data.level || 'info');
        }
    });
}

// 初始化事件监听器
function initEventListeners() {
    // 启动服务器按钮
    if (elements.startButton) {
        elements.startButton.addEventListener('click', function() {
            startService('ai_server');
        });
    }

    // 停止服务器按钮
    if (elements.stopButton) {
        elements.stopButton.addEventListener('click', function() {
            stopService('ai_server');
        });
    }

    // 连接NAO按钮
    if (elements.connectButton) {
        elements.connectButton.addEventListener('click', function() {
            if (naoConnected) {
                disconnectNao();
            } else {
                connectNao();
            }
        });
    }

    // 全局通知关闭按钮
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('toast-close-button')) {
            e.target.parentElement.remove();
        }
    });
}

// 初始化UI组件
function initUI() {
    // 初始化提示框
    initTooltips();

    // 初始化下拉菜单
    initDropdowns();
}

// 初始化提示框
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// 初始化下拉菜单
function initDropdowns() {
    const dropdownTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="dropdown"]'));
    dropdownTriggerList.map(function (dropdownTriggerEl) {
        return new bootstrap.Dropdown(dropdownTriggerEl);
    });
}

// 检查系统状态
function checkSystemStatus() {
    // 发送请求获取当前系统状态
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateServerStatus(data.ai_server_connected);
            updateNaoStatus(data.nao_connected);

            if (data.metrics) {
                updateSystemMetrics(data.metrics);
            }
        })
        .catch(error => {
            console.error('获取系统状态时出错:', error);
            updateServerStatus(false);
            updateNaoStatus(false);
        });
}

// 更新服务器状态
function updateServerStatus(connected) {
    serverConnected = connected;

    if (elements.serverStatus) {
        if (connected) {
            elements.serverStatus.innerHTML = '<i class="fas fa-circle text-success"></i> 服务器状态: 已连接';
        } else {
            elements.serverStatus.innerHTML = '<i class="fas fa-circle text-danger"></i> 服务器状态: 未连接';
        }
    }

    // 更新控制按钮状态
    if (elements.startButton) {
        elements.startButton.disabled = connected;
    }

    if (elements.stopButton) {
        elements.stopButton.disabled = !connected;
    }
}

// 更新NAO状态
function updateNaoStatus(connected) {
    naoConnected = connected;

    if (elements.naoStatus) {
        if (connected) {
            elements.naoStatus.innerHTML = '<i class="fas fa-circle text-success"></i> NAO状态: 已连接';
        } else {
            elements.naoStatus.innerHTML = '<i class="fas fa-circle text-secondary"></i> NAO状态: 未连接';
        }
    }

    // 更新连接按钮文本
    if (elements.connectButton) {
        if (connected) {
            elements.connectButton.textContent = '断开连接';
            elements.connectButton.classList.remove('btn-primary');
            elements.connectButton.classList.add('btn-danger');
        } else {
            elements.connectButton.textContent = '连接NAO';
            elements.connectButton.classList.remove('btn-danger');
            elements.connectButton.classList.add('btn-primary');
        }
    }
}

// 更新系统指标
function updateSystemMetrics(metrics) {
    const cpuElement = document.getElementById('cpu-usage');
    const memoryElement = document.getElementById('memory-usage');
    const responseTimeElement = document.getElementById('response-time');

    if (cpuElement && metrics.cpu !== undefined) {
        cpuElement.textContent = `${metrics.cpu.toFixed(1)}%`;

        const cpuProgress = document.getElementById('cpu-progress');
        if (cpuProgress) {
            cpuProgress.style.width = `${metrics.cpu}%`;

            // 根据使用率设置颜色
            if (metrics.cpu > 80) {
                cpuProgress.className = 'progress-bar bg-danger';
            } else if (metrics.cpu > 60) {
                cpuProgress.className = 'progress-bar bg-warning';
            } else {
                cpuProgress.className = 'progress-bar bg-success';
            }
        }
    }

    if (memoryElement && metrics.memory !== undefined) {
        memoryElement.textContent = `${metrics.memory.toFixed(1)}%`;

        const memoryProgress = document.getElementById('memory-progress');
        if (memoryProgress) {
            memoryProgress.style.width = `${metrics.memory}%`;

            // 根据使用率设置颜色
            if (metrics.memory > 80) {
                memoryProgress.className = 'progress-bar bg-danger';
            } else if (metrics.memory > 60) {
                memoryProgress.className = 'progress-bar bg-warning';
            } else {
                memoryProgress.className = 'progress-bar bg-success';
            }
        }
    }

    if (responseTimeElement && metrics.response_time !== undefined) {
        responseTimeElement.textContent = `${metrics.response_time.toFixed(0)}ms`;

        // 设置响应时间颜色
        if (metrics.response_time > 500) {
            responseTimeElement.classList.remove('text-success', 'text-warning');
            responseTimeElement.classList.add('text-danger');
        } else if (metrics.response_time > 200) {
            responseTimeElement.classList.remove('text-success', 'text-danger');
            responseTimeElement.classList.add('text-warning');
        } else {
            responseTimeElement.classList.remove('text-warning', 'text-danger');
            responseTimeElement.classList.add('text-success');
        }
    }

    // 更新NAO电池电量
    const batteryElement = document.getElementById('nao-battery');
    if (batteryElement && metrics.nao_battery !== undefined) {
        const battery = metrics.nao_battery;

        const batteryProgress = document.getElementById('battery-progress');
        if (batteryProgress) {
            batteryProgress.style.width = `${battery}%`;
            batteryProgress.textContent = `${battery.toFixed(0)}%`;

            // 根据电量设置颜色
            if (battery < 20) {
                batteryProgress.className = 'progress-bar bg-danger';
            } else if (battery < 40) {
                batteryProgress.className = 'progress-bar bg-warning';
            } else {
                batteryProgress.className = 'progress-bar bg-success';
            }
        }
    }

    // 更新NAO温度
    const tempElement = document.getElementById('nao-temp');
    if (tempElement && metrics.nao_temp !== undefined) {
        tempElement.textContent = `${metrics.nao_temp.toFixed(1)}°C`;

        // 设置温度颜色
        if (metrics.nao_temp > 45) {
            tempElement.classList.remove('text-success', 'text-warning');
            tempElement.classList.add('text-danger');
        } else if (metrics.nao_temp > 40) {
            tempElement.classList.remove('text-success', 'text-danger');
            tempElement.classList.add('text-warning');
        } else {
            tempElement.classList.remove('text-warning', 'text-danger');
            tempElement.classList.add('text-success');
        }
    }
}

// 更新情感显示
function updateEmotionDisplay(data) {
    const emotionElement = document.getElementById('current-emotion');
    if (emotionElement && data.emotion) {
        emotionElement.textContent = data.emotion;

        // 设置情感颜色
        const emotionColor = getEmotionColor(data.emotion);
        emotionElement.style.color = emotionColor;
    }

    // 更新情感强度
    const emotions = data.emotions || {};
    for (const [emotion, value] of Object.entries(emotions)) {
        const emotionBar = document.getElementById(`emotion-${emotion}-bar`);
        if (emotionBar) {
            emotionBar.style.width = `${value * 100}%`;
        }
    }

    // 更新学习状态
    const learningStates = data.learning_states || {};
    for (const [state, value] of Object.entries(learningStates)) {
        const stateElement = document.getElementById(`${state}-meter`);
        if (stateElement) {
            const stateBar = stateElement.querySelector('.progress-bar');
            if (stateBar) {
                stateBar.style.width = `${value * 100}%`;
                stateBar.textContent = `${Math.round(value * 100)}%`;

                // 根据值设置颜色
                if (value < 0.3) {
                    stateBar.className = 'progress-bar bg-danger';
                } else if (value < 0.6) {
                    stateBar.className = 'progress-bar bg-warning';
                } else {
                    stateBar.className = 'progress-bar bg-success';
                }
            }
        }
    }
}

// 显示系统问题
function showSystemIssues(issues) {
    issues.forEach(issue => {
        if (issue.type === 'warning' || issue.type === 'error') {
            showToast(issue.message, issue.type);
        }
    });
}

// 显示通知提示
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        // 创建容器
        const container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }

    // 创建提示框
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast show bg-${type} text-white`;
    toast.id = toastId;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    toast.innerHTML = `
        <div class="toast-header bg-${type} text-white">
            <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
            <small>${new Date().toLocaleTimeString()}</small>
            <button type="button" class="btn-close btn-close-white toast-close-button" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    // 添加到容器
    document.getElementById('toast-container').appendChild(toast);

    // 5秒后自动关闭
    setTimeout(() => {
        const toastElement = document.getElementById(toastId);
        if (toastElement) {
            toastElement.remove();
        }
    }, 5000);
}

// 启动服务
function startService(service) {
    fetch('/api/service/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ service: service })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(`${service} 启动成功`, 'success');
            if (service === 'ai_server') {
                updateServerStatus(true);
            }
        } else {
            showToast(`启动 ${service} 失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('启动服务时出错:', error);
        showToast(`启动 ${service} 时出错`, 'danger');
    });
}

// 停止服务
function stopService(service) {
    fetch('/api/service/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ service: service })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(`${service} 已停止`, 'info');
            if (service === 'ai_server') {
                updateServerStatus(false);
            }
        } else {
            showToast(`停止 ${service} 失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('停止服务时出错:', error);
        showToast(`停止 ${service} 时出错`, 'danger');
    });
}

// 连接NAO
function connectNao() {
    fetch('/api/nao/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ip: document.getElementById('nao-ip').value || '127.0.0.1' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast('NAO连接成功', 'success');
            updateNaoStatus(true);
        } else {
            showToast(`连接NAO失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('连接NAO时出错:', error);
        showToast('连接NAO时出错', 'danger');
    });
}

// 断开NAO连接
function disconnectNao() {
    fetch('/api/nao/disconnect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast('NAO已断开连接', 'info');
            updateNaoStatus(false);
        } else {
            showToast(`断开NAO连接失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('断开NAO连接时出错:', error);
        showToast('断开NAO连接时出错', 'danger');
    });
}

// 获取情感颜色
function getEmotionColor(emotion) {
    const colorMap = {
        "喜悦": "#4CAF50",  // 绿色
        "悲伤": "#2196F3",  // 蓝色
        "愤怒": "#F44336",  // 红色
        "恐惧": "#9C27B0",  // 紫色
        "惊讶": "#FFC107",  // 黄色
        "厌恶": "#795548",  // 棕色
        "中性": "#9E9E9E"   // 灰色
    };

    return colorMap[emotion] || "#9E9E9E";
}

// 格式化时间
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
}

// 格式化日期
function formatDate(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString();
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes < 1024) {
        return bytes + " B";
    } else if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(2) + " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    } else {
        return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
    }
}