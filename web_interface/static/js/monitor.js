// monitor.js - 监控页面JavaScript功能

// 图表对象
let systemChart;
let emotionChart;
let learningChart;

// 初始化图表
function initCharts() {
    // 系统资源图表
    const systemCtx = document.getElementById('systemChart').getContext('2d');
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
                },
                {
                    label: '响应时间(ms)',
                    data: [],
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.4,
                    yAxisID: 'responseTime'
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
                },
                responseTime: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '响应时间 (ms)'
                    },
                    grid: {
                        drawOnChartArea: false
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

    // 情感状态图表
    const emotionCtx = document.getElementById('emotionChart').getContext('2d');
    emotionChart = new Chart(emotionCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '喜悦',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4
                },
                {
                    label: '悲伤',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4
                },
                {
                    label: '愤怒',
                    data: [],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.4
                },
                {
                    label: '中性',
                    data: [],
                    borderColor: 'rgba(201, 203, 207, 1)',
                    backgroundColor: 'rgba(201, 203, 207, 0.2)',
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
                    max: 1,
                    title: {
                        display: true,
                        text: '强度'
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

    // 学习状态图表
    const learningCtx = document.getElementById('learningChart').getContext('2d');
    learningChart = new Chart(learningCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: '注意力',
                    data: [],
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.4
                },
                {
                    label: '参与度',
                    data: [],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4
                },
                {
                    label: '理解度',
                    data: [],
                    borderColor: 'rgba(255, 206, 86, 1)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
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
                    max: 1,
                    title: {
                        display: true,
                        text: '指数'
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
}

// 更新图表
function updateCharts(data) {
    // 更新系统图表
    if (systemChart) {
        const timestamp = new Date(data.timestamp * 1000);

        systemChart.data.labels.push(timestamp);
        systemChart.data.datasets[0].data.push(data.cpu);
        systemChart.data.datasets[1].data.push(data.memory);
        systemChart.data.datasets[2].data.push(data.response_time);

        // 保持固定数量的数据点，避免图表过长
        const maxDataPoints = 50;
        if (systemChart.data.labels.length > maxDataPoints) {
            systemChart.data.labels.shift();
            systemChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        systemChart.update();
    }
}

// 更新情感图表
function updateEmotionChart(data) {
    if (emotionChart) {
        const timestamp = new Date(data.timestamp * 1000);

        emotionChart.data.labels.push(timestamp);

        // 更新各情感强度
        const emotions = data.emotions || {};
        emotionChart.data.datasets.forEach(dataset => {
            const emotion = dataset.label;
            dataset.data.push(emotions[emotion] || 0);
        });

        // 保持固定数量的数据点
        const maxDataPoints = 50;
        if (emotionChart.data.labels.length > maxDataPoints) {
            emotionChart.data.labels.shift();
            emotionChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        emotionChart.update();
    }
}

// 更新学习状态图表
function updateLearningChart(data) {
    if (learningChart) {
        const timestamp = new Date(data.timestamp * 1000);
        const learningStates = data.learning_states || {};

        learningChart.data.labels.push(timestamp);

        // 更新各学习状态指数
        learningChart.data.datasets[0].data.push(learningStates['注意力'] || 0);
        learningChart.data.datasets[1].data.push(learningStates['参与度'] || 0);
        learningChart.data.datasets[2].data.push(learningStates['理解度'] || 0);

        // 保持固定数量的数据点
        const maxDataPoints = 50;
        if (learningChart.data.labels.length > maxDataPoints) {
            learningChart.data.labels.shift();
            learningChart.data.datasets.forEach(dataset => {
                learningChart.data.labels.shift();
            learningChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }

        learningChart.update();
    }
}

// 更新情感状态面板
function updateEmotionPanel(data) {
    // 更新主要情感显示
    const primaryEmotion = document.getElementById('primary-emotion');
    const emotionConfidence = document.getElementById('emotion-confidence').querySelector('.progress-bar');
    const attention = document.getElementById('attention-meter').querySelector('.progress-bar');
    const engagement = document.getElementById('engagement-meter').querySelector('.progress-bar');
    const understanding = document.getElementById('understanding-meter').querySelector('.progress-bar');

    if (primaryEmotion && data.emotion) {
        primaryEmotion.textContent = data.emotion;

        // 根据情感类型设置颜色
        let emotionColor = '#6c757d'; // 默认灰色
        switch(data.emotion) {
            case '喜悦':
                emotionColor = '#4CAF50'; // 绿色
                break;
            case '悲伤':
                emotionColor = '#2196F3'; // 蓝色
                break;
            case '愤怒':
                emotionColor = '#F44336'; // 红色
                break;
            case '恐惧':
                emotionColor = '#9C27B0'; // 紫色
                break;
            case '惊讶':
                emotionColor = '#FFD700'; // 金色
                break;
            case '厌恶':
                emotionColor = '#795548'; // 棕色
                break;
        }
        primaryEmotion.style.color = emotionColor;
    }

    // 更新情感置信度
    if (emotionConfidence && data.confidence) {
        const confidence = Math.round(data.confidence * 100);
        emotionConfidence.style.width = `${confidence}%`;
        emotionConfidence.textContent = `${confidence}%`;
    }

    // 更新学习状态指标
    const learningStates = data.learning_states || {};

    if (attention) {
        const attentionValue = Math.round((learningStates['注意力'] || 0) * 100);
        attention.style.width = `${attentionValue}%`;
        attention.textContent = `${attentionValue}%`;

        // 根据值设置颜色
        if (attentionValue < 30) {
            attention.className = 'progress-bar bg-danger';
        } else if (attentionValue < 60) {
            attention.className = 'progress-bar bg-warning';
        } else {
            attention.className = 'progress-bar bg-info';
        }
    }

    if (engagement) {
        const engagementValue = Math.round((learningStates['参与度'] || 0) * 100);
        engagement.style.width = `${engagementValue}%`;
        engagement.textContent = `${engagementValue}%`;

        if (engagementValue < 30) {
            engagement.className = 'progress-bar bg-danger';
        } else if (engagementValue < 60) {
            engagement.className = 'progress-bar bg-warning';
        } else {
            engagement.className = 'progress-bar bg-success';
        }
    }

    if (understanding) {
        const understandingValue = Math.round((learningStates['理解度'] || 0) * 100);
        understanding.style.width = `${understandingValue}%`;
        understanding.textContent = `${understandingValue}%`;

        if (understandingValue < 30) {
            understanding.className = 'progress-bar bg-danger';
        } else if (understandingValue < 60) {
            understanding.className = 'progress-bar bg-warning';
        } else {
            understanding.className = 'progress-bar bg-success';
        }
    }
}

// 更新系统日志
function updateSystemLogs(logs) {
    const logContainer = document.getElementById('system-logs');
    if (!logContainer) return;

    // 清空当前日志
    logContainer.innerHTML = '';

    // 添加新日志
    logs.forEach(log => {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${log.level.toLowerCase()}`;

        const timestamp = new Date(log.timestamp * 1000).toLocaleTimeString();
        logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> <span class="log-service">[${log.service}]</span> <span class="log-level">[${log.level}]</span> ${log.content}`;

        logContainer.appendChild(logEntry);
    });

    // 滚动到底部
    logContainer.scrollTop = logContainer.scrollHeight;
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化图表
    initCharts();

    // 请求系统日志
    document.getElementById('log-refresh').addEventListener('click', function() {
        fetchSystemLogs();
    });

    // 日志过滤
    document.querySelectorAll('[data-filter]').forEach(element => {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            const filter = this.getAttribute('data-filter');
            fetchSystemLogs(filter);
        });
    });

    // 开始获取数据
    fetchInitialData();

    // 设置Socket.IO事件监听
    setupSocketListeners();
});

// 获取初始数据
function fetchInitialData() {
    // 获取系统指标历史数据
    fetch('/api/system/metrics')
        .then(response => response.json())
        .then(data => {
            // 更新图表
            updateSystemChartWithHistory(data);
        })
        .catch(error => console.error('获取系统指标数据时出错:', error));

    // 获取情感历史数据
    fetch('/api/emotion/history')
        .then(response => response.json())
        .then(data => {
            // 更新情感图表
            updateEmotionChartWithHistory(data);
        })
        .catch(error => console.error('获取情感历史数据时出错:', error));

    // 获取系统日志
    fetchSystemLogs();
}

// 获取系统日志
function fetchSystemLogs(filter = 'all') {
    fetch(`/api/logs?service=${filter}&lines=50`)
        .then(response => response.json())
        .then(data => {
            updateSystemLogs(data);
        })
        .catch(error => console.error('获取系统日志时出错:', error));
}

// 使用历史数据更新系统图表
function updateSystemChartWithHistory(data) {
    if (!systemChart) return;

    // 清空现有数据
    systemChart.data.labels = [];
    systemChart.data.datasets.forEach(dataset => {
        dataset.data = [];
    });

    // 添加历史数据
    for (let i = 0; i < data.timestamp.length; i++) {
        const timestamp = new Date(data.timestamp[i] * 1000);
        systemChart.data.labels.push(timestamp);
        systemChart.data.datasets[0].data.push(data.cpu[i]);
        systemChart.data.datasets[1].data.push(data.memory[i]);
        systemChart.data.datasets[2].data.push(data.response_time[i]);
    }

    systemChart.update();
}

// 使用历史数据更新情感图表
function updateEmotionChartWithHistory(data) {
    if (!emotionChart || !learningChart) return;

    // 清空现有数据
    emotionChart.data.labels = [];
    emotionChart.data.datasets.forEach(dataset => {
        dataset.data = [];
    });

    learningChart.data.labels = [];
    learningChart.data.datasets.forEach(dataset => {
        dataset.data = [];
    });

    // 添加情感历史数据
    for (let i = 0; i < data.timestamp.length; i++) {
        const timestamp = new Date(data.timestamp[i] * 1000);

        // 更新情感图表
        emotionChart.data.labels.push(timestamp);
        emotionChart.data.datasets.forEach(dataset => {
            const emotion = dataset.label;
            const value = data.emotions[emotion] ? data.emotions[emotion][i] || 0 : 0;
            dataset.data.push(value);
        });

        // 更新学习状态图表
        learningChart.data.labels.push(timestamp);
        learningChart.data.datasets[0].data.push(data.learning_states['注意力'] ? data.learning_states['注意力'][i] || 0 : 0);
        learningChart.data.datasets[1].data.push(data.learning_states['参与度'] ? data.learning_states['参与度'][i] || 0 : 0);
        learningChart.data.datasets[2].data.push(data.learning_states['理解度'] ? data.learning_states['理解度'][i] || 0 : 0);
    }

    emotionChart.update();
    learningChart.update();

    // 更新情感面板 - 使用最新数据
    if (data.timestamp.length > 0) {
        const latestIndex = data.timestamp.length - 1;
        const latestData = {
            timestamp: data.timestamp[latestIndex],
            emotion: data.emotion[latestIndex],
            confidence: data.emotions[data.emotion[latestIndex]] ?
                       data.emotions[data.emotion[latestIndex]][latestIndex] || 0.5 : 0.5,
            emotions: {},
            learning_states: {}
        };

        // 构建最新的情感数据
        Object.keys(data.emotions).forEach(emotion => {
            latestData.emotions[emotion] = data.emotions[emotion][latestIndex] || 0;
        });

        // 构建最新的学习状态数据
        Object.keys(data.learning_states).forEach(state => {
            latestData.learning_states[state] = data.learning_states[state][latestIndex] || 0;
        });

        updateEmotionPanel(latestData);
    }
}

// 设置Socket.IO事件监听
function setupSocketListeners() {
    // 系统更新事件
    socket.on('system_update', function(data) {
        updateCharts(data);
    });

    // 情感更新事件
    socket.on('emotion_update', function(data) {
        updateEmotionChart(data);
        updateLearningChart(data);
        updateEmotionPanel(data);
    });

    // 日志更新事件
    socket.on('log_update', function(data) {
        updateSystemLogs(data);
    });
}