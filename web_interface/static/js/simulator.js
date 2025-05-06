// simulator.js - NAO教学系统模拟器交互脚本

// 全局变量
let socket;
let isConnected = false;
let messageHistory = [];
let currentEmotion = "中性";
let learningStates = {
    "注意力": 0.7,
    "参与度": 0.7,
    "理解度": 0.5
};

// DOM元素引用
let robotView, messageLog, emotionDisplay, attentionMeter, engagementMeter, understandingMeter;

// 初始化函数
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    robotView = document.getElementById('robot-view');
    messageLog = document.getElementById('message-log');
    emotionDisplay = document.getElementById('emotion-display');
    attentionMeter = document.getElementById('attention-meter');
    engagementMeter = document.getElementById('engagement-meter');
    understandingMeter = document.getElementById('understanding-meter');

    // 初始化Socket.IO连接
    initSocketConnection();

    // 初始化用户界面
    initUI();

    // 初始化事件监听器
    initEventListeners();

    // 显示初始机器人状态
    updateRobotView(currentEmotion);
    updateLearningStateMeters(learningStates);
});

// 初始化Socket.IO连接
function initSocketConnection() {
    socket = io();

    socket.on('connect', function() {
        console.log('已连接到服务器');
        isConnected = true;
        updateConnectionStatus(true);
    });

    socket.on('disconnect', function() {
        console.log('与服务器断开连接');
        isConnected = false;
        updateConnectionStatus(false);
    });

    socket.on('emotion_update', function(data) {
        console.log('收到情感更新:', data);
        currentEmotion = data.emotion;
        updateRobotView(currentEmotion);
        updateEmotionDisplay(data);
        updateLearningStateMeters(data.learning_states);
    });

    socket.on('system_response', function(data) {
        console.log('收到系统响应:', data);
        addMessage('NAO', data.text);

        // 如果有动作，更新机器人状态
        if (data.actions && data.actions.length > 0) {
            performGesture(data.actions[0]);
        }
    });
}

// 初始化用户界面
function initUI() {
    // 初始化机器人视图
    updateRobotView(currentEmotion);

    // 初始化消息日志
    addMessage('系统', '欢迎使用NAO教学系统模拟器！');

    // 初始化情感显示
    updateEmotionDisplay({
        emotion: currentEmotion,
        emotions: {
            "喜悦": 0.1,
            "悲伤": 0.1,
            "愤怒": 0.1,
            "恐惧": 0.1,
            "惊讶": 0.1,
            "厌恶": 0.1,
            "中性": 0.4
        }
    });

    // 初始化学习状态仪表
    updateLearningStateMeters(learningStates);
}

// 初始化事件监听器
function initEventListeners() {
    // 发送按钮点击事件
    document.getElementById('send-button').addEventListener('click', sendMessage);

    // 输入框回车事件
    document.getElementById('message-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // 连接按钮点击事件
    document.getElementById('connect-button').addEventListener('click', toggleConnection);

    // 情感模拟按钮事件
    document.querySelectorAll('.emotion-button').forEach(button => {
        button.addEventListener('click', function() {
            simulateEmotion(this.dataset.emotion);
        });
    });

    // 演示场景按钮事件
    document.getElementById('demo-button').addEventListener('click', runDemoScenario);

    // 清除日志按钮事件
    document.getElementById('clear-log-button').addEventListener('click', clearMessageLog);
}

// 发送消息
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    if (!message) return;

    // 添加消息到日志
    addMessage('用户', message);

    // 清空输入框
    input.value = '';

    // 发送消息到服务器
    if (isConnected) {
        socket.emit('student_message', { text: message });
    } else {
        // 本地模拟响应
        simulateResponse(message);
    }
}

// 添加消息到日志
function addMessage(sender, content) {
    const messageItem = document.createElement('div');
    messageItem.className = `message ${sender.toLowerCase()}-message`;

    const timestamp = new Date().toLocaleTimeString();
    messageItem.innerHTML = `
        <span class="message-timestamp">[${timestamp}]</span>
        <span class="message-sender">${sender}:</span>
        <span class="message-content">${content}</span>
    `;

    messageLog.appendChild(messageItem);
    messageLog.scrollTop = messageLog.scrollHeight;

    // 保存到历史记录
    messageHistory.push({
        timestamp: new Date(),
        sender: sender,
        content: content
    });
}

// 更新连接状态
function updateConnectionStatus(connected) {
    const connectButton = document.getElementById('connect-button');

    if (connected) {
        connectButton.textContent = '断开连接';
        connectButton.classList.remove('btn-primary');
        connectButton.classList.add('btn-danger');
    } else {
        connectButton.textContent = '连接服务器';
        connectButton.classList.remove('btn-danger');
        connectButton.classList.add('btn-primary');
    }
}

// 切换连接状态
function toggleConnection() {
    if (isConnected) {
        // 断开连接
        socket.disconnect();
    } else {
        // 重新连接
        socket.connect();
    }
}

// 更新机器人视图
function updateRobotView(emotion) {
    // 根据情感设置背景颜色
    let bgColor = 'gray';

    switch (emotion) {
        case '喜悦':
            bgColor = '#4CAF50'; // 绿色
            break;
        case '悲伤':
            bgColor = '#2196F3'; // 蓝色
            break;
        case '愤怒':
            bgColor = '#F44336'; // 红色
            break;
        case '恐惧':
            bgColor = '#9C27B0'; // 紫色
            break;
        case '惊讶':
            bgColor = '#FFC107'; // 黄色
            break;
        case '厌恶':
            bgColor = '#795548'; // 棕色
            break;
        case '中性':
        default:
            bgColor = '#9E9E9E'; // 灰色
            break;
    }

    // 更新机器人视图
    robotView.innerHTML = `
        <div class="robot-face" style="background-color: ${bgColor};">
            <div class="robot-head">
                <div class="robot-eyes">
                    <div class="robot-eye left"></div>
                    <div class="robot-eye right"></div>
                </div>
                <div class="robot-mouth emotion-${emotion.toLowerCase()}"></div>
            </div>
        </div>
        <div class="robot-name">NAO</div>
        <div class="robot-emotion">当前情绪: ${emotion}</div>
    `;
}

// 更新情感显示
function updateEmotionDisplay(emotionData) {
    // 更新情感标签
    emotionDisplay.querySelector('.current-emotion').textContent = emotionData.emotion;

    // 更新情感条形图
    const emotionsDiv = emotionDisplay.querySelector('.emotion-bars');
    emotionsDiv.innerHTML = '';

    for (const [emotion, value] of Object.entries(emotionData.emotions)) {
        const barContainer = document.createElement('div');
        barContainer.className = 'emotion-bar-container';

        const label = document.createElement('div');
        label.className = 'emotion-label';
        label.textContent = emotion;

        const barWrapper = document.createElement('div');
        barWrapper.className = 'emotion-bar-wrapper';

        const bar = document.createElement('div');
        bar.className = 'emotion-bar';
        bar.style.width = `${value * 100}%`;
        bar.style.backgroundColor = getEmotionColor(emotion);

        const percentage = document.createElement('span');
        percentage.className = 'emotion-percentage';
        percentage.textContent = `${Math.round(value * 100)}%`;

        barWrapper.appendChild(bar);
        barContainer.appendChild(label);
        barContainer.appendChild(barWrapper);
        barContainer.appendChild(percentage);
        emotionsDiv.appendChild(barContainer);
    }
}

// 更新学习状态仪表
function updateLearningStateMeters(states) {
    if (!states) return;

    // 更新注意力
    const attention = states["注意力"] || 0;
    attentionMeter.querySelector('.meter-fill').style.width = `${attention * 100}%`;
    attentionMeter.querySelector('.meter-value').textContent = `${Math.round(attention * 100)}%`;

    // 更新参与度
    const engagement = states["参与度"] || 0;
    engagementMeter.querySelector('.meter-fill').style.width = `${engagement * 100}%`;
    engagementMeter.querySelector('.meter-value').textContent = `${Math.round(engagement * 100)}%`;

    // 更新理解度
    const understanding = states["理解度"] || 0;
    understandingMeter.querySelector('.meter-fill').style.width = `${understanding * 100}%`;
    understandingMeter.querySelector('.meter-value').textContent = `${Math.round(understanding * 100)}%`;

    // 保存当前学习状态
    learningStates = states;
}

// 模拟情感
function simulateEmotion(emotion) {
    if (!emotion) return;

    console.log(`模拟情感: ${emotion}`);

    // 更新当前情感
    currentEmotion = emotion;

    // 更新机器人视图
    updateRobotView(emotion);

    // 生成模拟的情感数据
    const emotionStrengths = {
        "喜悦": 0.1,
        "悲伤": 0.1,
        "愤怒": 0.1,
        "恐惧": 0.1,
        "惊讶": 0.1,
        "厌恶": 0.1,
        "中性": 0.1
    };

    // 设置当前情感强度更高
    emotionStrengths[emotion] = 0.7;

    // 生成学习状态
    const stateValues = {};
    if (emotion === '喜悦') {
        stateValues["注意力"] = 0.8;
        stateValues["参与度"] = 0.9;
        stateValues["理解度"] = 0.7;
    } else if (emotion === '悲伤' || emotion === '厌恶') {
        stateValues["注意力"] = 0.4;
        stateValues["参与度"] = 0.3;
        stateValues["理解度"] = 0.4;
    } else if (emotion === '惊讶') {
        stateValues["注意力"] = 0.9;
        stateValues["参与度"] = 0.7;
        stateValues["理解度"] = 0.5;
    } else if (emotion === '愤怒') {
        stateValues["注意力"] = 0.6;
        stateValues["参与度"] = 0.3;
        stateValues["理解度"] = 0.3;
    } else if (emotion === '恐惧') {
        stateValues["注意力"] = 0.7;
        stateValues["参与度"] = 0.2;
        stateValues["理解度"] = 0.2;
    } else {
        // 中性
        stateValues["注意力"] = 0.6;
        stateValues["参与度"] = 0.6;
        stateValues["理解度"] = 0.5;
    }

    // 添加随机波动
    for (const state in stateValues) {
        stateValues[state] = Math.max(0.1, Math.min(0.9, stateValues[state] + (Math.random() - 0.5) * 0.1));
    }

    // 创建完整的情感数据
    const emotionData = {
        emotion: emotion,
        emotions: emotionStrengths,
        learning_states: stateValues
    };

    // 更新情感显示
    updateEmotionDisplay(emotionData);

    // 更新学习状态仪表
    updateLearningStateMeters(stateValues);

    // 如果已连接到服务器，发送情感数据
    if (isConnected) {
        socket.emit('simulate_emotion', { emotion: emotion });
    }

    // 添加消息到日志
    addMessage('系统', `检测到情感变化: ${emotion}`);
}

// 模拟响应
function simulateResponse(message) {
    // 简单的模拟响应逻辑
    setTimeout(() => {
        let response, gesture;

        if (message.includes('你好') || message.includes('hello')) {
            response = "你好！我是NAO机器人助教，很高兴为你提供学习帮助。";
            gesture = "greeting";
        } else if (message.includes('再见')) {
            response = "再见！希望今天的学习对你有帮助。";
            gesture = null;
        } else if (message.match(/什么是(.*?)[?？]/)) {
            const concept = message.match(/什么是(.*?)[?？]/)[1].trim();
            response = `${concept}是计算机科学中的一个重要概念，它通常指...具体来说，${concept}可以用于解决...`;
            gesture = "explaining";
        } else if (message.includes('谢谢') || message.includes('感谢')) {
            response = "不用谢！很高兴能帮到你。有任何问题随时问我。";
            gesture = "greeting";
        } else if (message.includes('不明白') || message.includes('不懂')) {
            response = "没关系，我可以用另一种方式解释。让我换个角度来说明这个概念...";
            gesture = "explaining";
        } else {
            response = "这是个很好的问题。让我思考一下...";
            gesture = "thinking";
        }

        // 添加响应到日志
        addMessage('NAO', response);

        // 如果有手势，执行它
        if (gesture) {
            performGesture(gesture);
        }

    }, 1000); // 模拟思考时间
}

// 执行手势
function performGesture(gesture) {
    // 根据手势更新机器人状态
    const statusText = document.getElementById('robot-status');

    switch (gesture) {
        case 'greeting':
            statusText.textContent = '执行手势: 问候';
            // 这里可以添加手势动画
            break;
        case 'explaining':
            statusText.textContent = '执行手势: 解释';
            break;
        case 'pointing':
            statusText.textContent = '执行手势: 指向';
            break;
        case 'thinking':
            statusText.textContent = '执行手势: 思考';
            break;
        default:
            statusText.textContent = '标准姿势';
    }

    // 添加消息到日志
    addMessage('系统', `NAO执行手势: ${gesture}`);

    // 3秒后恢复默认状态
    setTimeout(() => {
        statusText.textContent = '标准姿势';
    }, 3000);
}

// 运行教学演示场景
function runDemoScenario() {
    // 禁用演示按钮
    const demoButton = document.getElementById('demo-button');
    demoButton.disabled = true;

    // 清空消息日志
    clearMessageLog();

    // 添加标题
    addMessage('系统', '=== 开始教学演示场景 ===');

    // 演示场景步骤
    const demoSteps = [
        { action: 'nao_say', content: '欢迎来到编程基础课。今天我将为大家讲解C语言的基本概念。', delay: 0 },
        { action: 'gesture', content: 'greeting', delay: 1000 },
        { action: 'nao_say', content: '我们将学习三个主要概念：变量、函数和条件语句。', delay: 3000 },
        { action: 'gesture', content: 'explaining', delay: 1000 },
        { action: 'nao_say', content: '首先，让我们了解什么是变量。变量是计算机内存中存储数据的命名空间。', delay: 3000 },
        { action: 'gesture', content: 'pointing', delay: 1000 },
        { action: 'emotion', content: '惊讶', delay: 2000 },
        { action: 'nao_say', content: '我注意到有些同学可能对变量的概念还不太清楚。让我用另一种方式解释。', delay: 3000 },
        { action: 'nao_say', content: '变量就像是一个带标签的盒子，你可以在里面放东西，也可以随时查看或改变里面的内容。', delay: 3000 },
        { action: 'gesture', content: 'explaining', delay: 1000 },
        { action: 'emotion', content: '喜悦', delay: 2000 },
        { action: 'nao_say', content: '很好！看来大家已经理解了变量的概念。接下来我们来看看函数...', delay: 3000 },
        { action: 'nao_say', content: '函数是一段可重复使用的代码块，它可以接收输入参数，并返回处理结果。', delay: 3000 },
        { action: 'student_ask', content: '老师，这个函数和数学中的函数有什么区别？', delay: 2000 },
        { action: 'nao_say', content: '很好的问题！编程中的函数与数学中的函数有相似之处，都是接收输入并产生输出。不过编程中的函数更加灵活，除了计算值外，还可以执行各种操作，如修改数据、显示信息等。', delay: 3000 },
        { action: 'emotion', content: '惊讶', delay: 1000 },
        { action: 'nao_say', content: '现在，让我们继续学习条件语句。条件语句允许程序根据不同条件执行不同的代码分支。', delay: 3000 },
        { action: 'student_ask', content: '能举个例子吗？', delay: 2000 },
        { action: 'nao_say', content: '当然！例如，我们可以用条件语句判断一个数是否为偶数：if (number % 2 == 0) { printf("这是偶数"); } else { printf("这是奇数"); }', delay: 3000 },
        { action: 'gesture', content: 'explaining', delay: 1000 },
        { action: 'emotion', content: '喜悦', delay: 2000 },
        { action: 'nao_say', content: '今天的课程到此结束。下次我们将学习循环结构。有什么问题可以随时问我！', delay: 3000 },
        { action: 'gesture', content: 'greeting', delay: 1000 },
        { action: 'system', content: '=== 教学演示场景结束 ===', delay: 2000 }
    ];

    // 执行演示步骤
    let currentStep = 0;

    function executeNextStep() {
        if (currentStep >= demoSteps.length) {
            // 演示结束，启用演示按钮
            demoButton.disabled = false;
            return;
        }

        const step = demoSteps[currentStep];

        setTimeout(() => {
            switch (step.action) {
                case 'nao_say':
                    addMessage('NAO', step.content);
                    break;
                case 'student_ask':
                    addMessage('学生', step.content);
                    break;
                case 'gesture':
                    performGesture(step.content);
                    break;
                case 'emotion':
                    simulateEmotion(step.content);
                    break;
                case 'system':
                    addMessage('系统', step.content);
                    break;
            }

            currentStep++;
            executeNextStep();
        }, step.delay);
    }

    // 开始执行演示
    executeNextStep();
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

// 清除消息日志
function clearMessageLog() {
    messageLog.innerHTML = '';
    messageHistory = [];
}