
# NAO机器人智能辅助教学系统

本项目整合自然语言处理、情感识别、知识推荐等技术，构建了一个多功能的NAO机器人智能辅教系统，具有多模态交互、情绪感知和个性化教学能力。

---

## 📁 项目总体结构

项目主要分为以下几个模块：

- `ai_server/`：AI服务器端
- `nao_control/`：NAO机器人控制
- `knowledge_extraction/`：知识提取模块
- 其他模块：配置文件、启动脚本、测试工具等

---

## 🔧 主要文件说明

### 1. 启动脚本

#### `start_ai_server.py`

AI服务器启动脚本，主要功能：

- 解析命令行参数（主机地址、端口、配置文件）
- 创建并启动AI服务器实例
- 使用 `asyncio` 运行服务

#### `start_nao_client.py`

NAO客户端启动脚本：

- 连接并控制NAO机器人
- 解析命令行参数（IP地址、端口）
- 调用NAO控制函数

---

### 2. AI服务器模块（`ai_server/`）

#### `server.py`

- WebSocket接口核心实现
- 处理音频、图像、文本和命令消息
- 管理任务队列和处理线程

#### `emotion/face_emotion.py`

- 面部表情情感识别
- 支持情感类别：愤怒、厌恶、恐惧、喜悦、中性、悲伤、惊讶

#### `emotion/audio_emotion.py`

- 音频情感识别（基于LSTM）
- 特征提取：MFCC、能量等

#### `emotion/fusion.py`

- 多模态情绪融合（音频 + 图像）
- 输出注意力、参与度、理解度状态

#### `nlp/llm_model.py`

- LLM封装类，使用 DeepSeek-7B-Chat
- 支持 4-bit 和 8-bit 量化加载

#### `nlp/conversation.py`

- 对话管理
- 意图识别：概念解释、问题解决、鼓励等

#### `knowledge/knowledge_graph.py`

- 使用 Neo4j 构建知识图谱
- 节点、关系管理与查询

#### `knowledge/recommender.py`

- 个性化知识推荐
- 考虑掌握度、重要性、难度

#### `knowledge/learning_path.py`

- 学习路径规划器（Q-learning）

#### `knowledge/pdf_knowledge_adapter.py`

- PDF知识导入适配器
- 映射节点与属性，去重冲突处理

#### `utils/config.py`

- 系统配置加载和管理

#### `utils/logger.py`

- 全局日志配置（控制台 + 文件）

---

### 3. NAO机器人控制模块（`nao_control/`）

#### `main.py`

- NAO主控逻辑
- 初始化代理、执行基本测试

#### `motion_control.py`

- 教学相关动作控制
- 动作如：思考、解释、指向、问候

#### `video_capture.py`

- 图像采集（NAO摄像头）
- 图像转numpy数组

#### `audio_capture.py`

- 录音控制（麦克风）
- 保存音频并支持事件触发

#### `websocket_client.py`

- 与AI服务器WebSocket通信
- 发送多模态数据并接收命令

---

### 4. 知识提取模块（`knowledge_extraction/`）

#### `ocr_pdf_extractor.py`

- OCR方式提取PDF文本
- 提取结构化章节内容

#### `llm_knowledge_extractor.py`

- LLM分析文本，提取知识点与关系
- 构建知识图谱JSON结构

#### `pure_paddle_ocr_extractor.py`

- 使用 PaddleOCR 的轻量提取器
- 支持多语言，优化教学类文档识别

#### `import_to_neo4j.py`

- 将知识图谱导入Neo4j
- 自动建图并设定约束

---

### 5. 测试与辅助工具

| 文件 | 功能 |
|------|------|
| `test_nao_connection.py` | 测试NAO连接与基本功能 |
| `test_llm_model.py` | 测试LLM加载与问答效果 |
| `test.py` | Paddle环境配置测试 |
| `ocr_llm_extraction.py` | OCR+LLM联合提取知识图谱 |

---

### 6. 配置文件

#### `config.json`

包含以下配置项：

- AI服务器地址、端口
- LLM模型名称、路径、量化方式
- 情感融合权重
- Neo4j URI、用户名、密码

---

## 🧩 系统架构图（逻辑结构）

```
客户端 (NAO 端)
├── 视频采集
├── 音频采集
├── 动作控制
└── WebSocket通信
        ⇅
AI服务器
├── 大语言模型（LLM）
├── 情感分析（音频+面部）
├── 知识图谱查询与推荐
└── 对话管理器
        ⇅
知识提取模块
├── PDF OCR
├── LLM知识点提取
└── Neo4j导入工具
```

---

## ✅ 系统能力总结

- **多模态输入**：音频 + 视频 + 文本
- **情感识别融合**：分析学生情绪状态
- **智能对话管理**：识别意图，生成回复
- **知识图谱与推荐**：精准推荐学习路径
- **NAO机器人交互**：肢体+语言教学配合
