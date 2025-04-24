NAO_Teaching_System/
├── nao_control/               # NAO机器人控制(Python 2.7)
│   ├── main.py                # 主程序入口
│   ├── audio_capture.py       # 音频采集模块
│   ├── video_capture.py       # 视频采集模块
│   ├── motion_control.py      # 动作控制模块
│   └── websocket_client.py    # 与服务器通信模块
│
├── ai_server/                 # AI服务器(Python 3.10)
│   ├── server.py              # 服务器主程序
│   ├── nlp/                   # 自然语言处理模块
│   │   ├── llm_model.py       # 大语言模型集成
│   │   └── conversation.py    # 对话管理
│   ├── emotion/               # 情感识别模块
│   │   ├── audio_emotion.py   # 语音情感分析
│   │   ├── face_emotion.py    # 面部表情分析
│   │   └── fusion.py          # 多模态融合
│   ├── knowledge/             # 知识推荐模块
│   │   ├── knowledge_graph.py # 知识图谱
│   │   ├── recommender.py     # 推荐算法
│   │   ├── learning_path.py   # 学习路径规划
│   │   └── pdf_knowledge_adapter.py # PDF知识图谱适配器
│   └── utils/                 # 工具函数
│       ├── config.py          # 配置文件
│       └── logger.py          # 日志工具
│
├── knowledge_extraction/      # PDF知识提取模块(Python 3.10)
│   ├── pdf_extractor.py       # PDF文本提取
│   ├── knowledge_extractor.py # 知识点提取
│   └── knowledge_graph_builder.py # 知识图谱构建
│
├── extract_knowledge_graph.py # 知识图谱提取主程序
├── test_pdf_extraction.py     # PDF提取测试脚本
├── start_nao_client.py        # NAO客户端启动脚本
├── start_ai_server.py         # AI服务器启动脚本
├── test_nao_connection.py     # NAO连接测试脚本
├── test_llm_model.py          # 大语言模型测试脚本
│
├── config.json                # 系统配置文件
├── output/                    # 输出文件夹
│   └── knowledge_graph.json   # 输出的知识图谱文件
└── temp/                      # 临时文件夹
    └── 编译原理_测试.json     # 测试生成的知识图谱