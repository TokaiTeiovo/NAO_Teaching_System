// dashboard.js - NAO教学系统仪表盘JavaScript功能

// 图表对象
let systemChart = null;
let emotionChart = null;
let sessionChart = null;

// 初始化仪表盘
document.addEventListener('DOMContentLoaded', function() {
    // 初始化图表
    initCharts();

    // 加载最近会话
    loadRecentSessions();

    // 加载系统状态
    loadSystemStatus();

    // 绑定事件监听器
    bindEventListeners();

    // 设置定时刷新
    setInterval(updateDashboard, 5000);
});

// 初始化图表
function initCharts() {
    // 系统资源图表
    const systemCtx = document.getElementById('system-chart').getContext('2d');
    systemChart = new Chart(systemCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'CPU使用率',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.4
                },
                {
                    label: '内存使用率',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'minute',
                        displayFormats: {
                            minute: 'HH:mm:ss'
                        }
                    },
                    title: {
                        display: true,
                        text: '时间'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '百分比 (%)'
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            }
        }
    });

    // 情感分布图表
    const emotionCtx = document.getElementById('emotion-chart').getContext('2d');
    emotionChart = new Chart(emotionCtx, {
        type: 'pie',
        data: {
            labels: ['喜悦', '悲伤', '愤怒', '恐惧', '惊讶', '厌恶', '中性'],
            datasets: [{
                data: [10, 5, 3, 4, 7, 2, 15],
                backgroundColor: [
                    '#4CAF50', // 喜悦
                    '#2196F3', // 悲伤
                    '#F44336', // 愤怒
                    '#9C27B0', // 恐惧
                    '#FFC107', // 惊讶
                    '#795548', // 厌恶
                    '#9E9E9E'  // 中性
                ],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                }
            }
        }
    });

    // 会话统计图表
    const sessionCtx = document.getElementById('session-chart').getContext('2d');
    sessionChart = new Chart(sessionCtx, {
        type: 'bar',
        data: {
            labels: ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
            datasets: [{
                label: '会话数量',
                data: [3, 5, 2, 4, 6, 1, 0],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    stepSize: 1
                }
            }
        }
    });
}

// 加载最近会话
function loadRecentSessions() {
    fetch('/api/sessions?limit=5')
        .then(response => response.json())
        .then(data => {
            const sessionsList = document.getElementById('recent-sessions');
            if (!sessionsList) return;

            sessionsList.innerHTML = '';

            if (data.length === 0) {
                sessionsList.innerHTML = '<div class="list-group-item text-center">无会话记录</div>';
                return;
            }

            data.forEach(session => {
                const listItem = document.createElement('a');
                listItem.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                listItem.href = `/sessions/${session.id}`;

                const duration = session.end_time ? Math.round((session.end_time - session.start_time) / 60) : '进行中';

                listItem.innerHTML = `
                    <div>
                        <div class="fw-bold">${session.id}</div>
                        <small class="text-muted">${session.formatted_time || new Date(session.start_time * 1000).toLocaleString()}</small>
                    </div>
                    <span class="badge bg-primary rounded-pill">${typeof duration === 'number' ? `${duration}分钟` : duration}</span>
                `;

                sessionsList.appendChild(listItem);
            });
        })
        .catch(error => {
            console.error('获取会话列表出错:', error);
            document.getElementById('recent-sessions').innerHTML =
                '<div class="list-group-item text-center text-danger">加载失败</div>';
        });
}

// 加载系统状态
function loadSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateSystemStatus(data);
        })
        .catch(error => {
            console.error('获取系统状态出错:', error);
        });
}

// 更新系统状态
function updateSystemStatus(data) {
    // 更新服务器状态
    const serverStatus = document.getElementById('ai-server-status');
    if (serverStatus) {
        if (data.ai_server_connected) {
            serverStatus.innerHTML = '<span class="badge bg-success">已连接</span>';
        } else {
            serverStatus.innerHTML = '<span class="badge bg-danger">未连接</span>';
        }
    }

    // 更新NAO状态
    const naoStatus = document.getElementById('nao-status');
    if (naoStatus) {
        if (data.nao_connected) {
            naoStatus.innerHTML = '<span class="badge bg-success">已连接</span>';
        } else {
            naoStatus.innerHTML = '<span class="badge bg-secondary">未连接</span>';
        }
    }

    // 更新NAO电池电量
    const naoBattery = document.getElementById('nao-battery');
    if (naoBattery && data.nao_battery !== undefined) {
        const batteryLevel = Math.round(data.nao_battery);
        let batteryColor = 'bg-success';

        if (batteryLevel < 20) {
            batteryColor = 'bg-danger';
        } else if (batteryLevel < 40) {
            batteryColor = 'bg-warning';
        }

        naoBattery.innerHTML = `
            <div class="progress">
                <div class="progress-bar ${batteryColor}" role="progressbar" style="width: ${batteryLevel}%"
                    aria-valuenow="${batteryLevel}" aria-valuemin="0" aria-valuemax="100">${batteryLevel}%</div>
            </div>
        `;
    }

    // 更新响应时间
    const responseTime = document.getElementById('response-time');
    if (responseTime && data.response_time !== undefined) {
        const time = Math.round(data.response_time);
        let timeColor = 'bg-info';

        if (time > 500) {
            timeColor = 'text-danger';
        } else if (time > 200) {
            timeColor = 'text-warning';
        } else {
            timeColor = 'text-info';
        }

        responseTime.innerHTML = `<span class="${timeColor}">${time}ms</span>`;
    }

    // 更新系统图表数据
    if (systemChart && data.timestamp) {
        updateSystemChart(data);
    }

    // 检查系统警告
    if (data.issues && data.issues.length > 0) {
        showSystemIssues(data.issues);
    }
}

// 更新系统图表
function updateSystemChart(data) {
    const timestamp = new Date(data.timestamp * 1000);

    // 添加数据点
    systemChart.data.labels.push(timestamp);
    systemChart.data.datasets[0].data.push(data.cpu);
    systemChart.data.datasets[1].data.push(data.memory);

    // 保持固定数量的数据点
    const maxDataPoints = 20;
    if (systemChart.data.labels.length > maxDataPoints) {
        systemChart.data.labels.shift();
        systemChart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }

    // 更新图表
    systemChart.update();
}

// 显示系统问题
function showSystemIssues(issues) {
    const issuesContainer = document.getElementById('system-issues');
    if (!issuesContainer) return;

    // 清空容器
    issuesContainer.innerHTML = '';

    // 如果没有问题，显示一条成功消息
    if (issues.length === 0) {
        issuesContainer.innerHTML = `
            <div class="alert alert-success" role="alert">
                <i class="fas fa-check-circle"></i> 系统运行正常，无警告或错误
            </div>
        `;
        return;
    }

    // 添加每个问题
    issues.forEach(issue => {
        const alertType = issue.type === 'warning' ? 'warning' : 'danger';
        const icon = issue.type === 'warning' ? 'exclamation-triangle' : 'exclamation-circle';

        const alert = document.createElement('div');
        alert.className = `alert alert-${alertType} mb-2`;
        alert.role = 'alert';

        alert.innerHTML = `
            <i class="fas fa-${icon}"></i> ${issue.message}
        `;

        issuesContainer.appendChild(alert);
    });
}

// 绑定事件监听器
function bindEventListeners() {
    // 启动服务器按钮
    const startServerBtn = document.getElementById('start-server-btn');
    if (startServerBtn) {
        startServerBtn.addEventListener('click', () => {
            startService('ai_server');
        });
    }

    // 停止服务器按钮
    const stopServerBtn = document.getElementById('stop-server-btn');
    if (stopServerBtn) {
        stopServerBtn.addEventListener('click', () => {
            stopService('ai_server');
        });
    }

    // 连接NAO按钮
    const connectNaoBtn = document.getElementById('connect-nao-btn');
    if (connectNaoBtn) {
        connectNaoBtn.addEventListener('click', () => {
            connectNao();
        });
    }

    // 断开NAO连接按钮
    const disconnectNaoBtn = document.getElementById('disconnect-nao-btn');
    if (disconnectNaoBtn) {
        disconnectNaoBtn.addEventListener('click', () => {
            disconnectNao();
        });
    }

    // 刷新状态按钮
    const refreshStatusBtn = document.getElementById('refresh-status-btn');
    if (refreshStatusBtn) {
        refreshStatusBtn.addEventListener('click', () => {
            loadSystemStatus();
        });
    }
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
            loadSystemStatus();
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
            loadSystemStatus();
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
    // 获取IP地址
    const ipInput = document.getElementById('nao-ip-input');
    const ip = ipInput ? ipInput.value : '127.0.0.1';

    fetch('/api/nao/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ip: ip })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            showToast('NAO连接成功', 'success');
            loadSystemStatus();
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
            loadSystemStatus();
        } else {
            showToast(`断开NAO连接失败: ${data.message}`, 'danger');
        }
    })
    .catch(error => {
        console.error('断开NAO连接时出错:', error);
        showToast('断开NAO连接时出错', 'danger');
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
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
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

// 更新仪表盘
function updateDashboard() {
    // 加载系统状态
    loadSystemStatus();

    // 更新情感分布图表
    updateEmotionChart();

    // 更新会话图表数据
    // 这里通常应该从服务器获取数据，这里使用模拟数据
    updateSessionChart();
}

// 更新情感分布图表
function updateEmotionChart() {
    // 获取情感分布数据
    fetch('/api/emotion/history')
        .then(response => response.json())
        .then(data => {
            // 计算情感分布
            const emotionCounts = {};

            // 初始化所有情感计数为0
            const emotions = ['喜悦', '悲伤', '愤怒', '恐惧', '惊讶', '厌恶', '中性'];
            emotions.forEach(emotion => {
                emotionCounts[emotion] = 0;
            });

            // 统计每种情感出现次数
            if (data.emotion && data.emotion.length > 0) {
                data.emotion.forEach(emotion => {
                    if (emotionCounts[emotion] !== undefined) {
                        emotionCounts[emotion]++;
                    }
                });
            }

            // 更新图表数据
            emotionChart.data.datasets[0].data = emotions.map(emotion => emotionCounts[emotion]);
            emotionChart.update();
        })
        .catch(error => {
            console.error('获取情感历史数据出错:', error);
        });
}

// 更新会话图表数据
function updateSessionChart() {
    // 这里应该从服务器获取数据，这里使用模拟数据
    const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    const sessionCounts = [
        Math.floor(Math.random() * 5) + 1,
        Math.floor(Math.random() * 5) + 1,
        Math.floor(Math.random() * 5) + 1,
        Math.floor(Math.random() * 5) + 1,
        Math.floor(Math.random() * 5) + 1,
        Math.floor(Math.random() * 3),
        Math.floor(Math.random() * 3)
    ];

    sessionChart.data.labels = days;
    sessionChart.data.datasets[0].data = sessionCounts;
    sessionChart.update();
}