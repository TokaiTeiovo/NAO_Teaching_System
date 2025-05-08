@echo off
chcp 936 >nul
echo ============================================
echo NAO机器人智能辅助教学系统启动脚本
echo ============================================

REM 确保Neo4j数据库运行
echo 检查Neo4j数据库服务...
sc query "Neo4j" | find "RUNNING" > nul
if %ERRORLEVEL% NEQ 0 (
    echo Neo4j数据库服务未运行！请确保已启动Neo4j数据库。
    echo 提示：您可以通过Neo4j Desktop或服务管理器启动Neo4j数据库。
    pause
    exit /b 1
)

REM 检查配置文件
if not exist config.json (
    echo 错误：配置文件不存在！
    echo 正在创建默认配置文件...
    
    echo {> config.json
    echo   "server": {>> config.json
    echo     "host": "localhost",>> config.json
    echo     "port": 8765>> config.json
    echo   },>> config.json
    echo   "llm": {>> config.json
    echo     "model_name": "deepseek-ai/deepseek-llm-7b-chat",>> config.json
    echo     "model_path": "models/deepseek-llm-7b-chat",>> config.json
    echo     "use_lora": false>> config.json
    echo   },>> config.json
    echo   "emotion": {>> config.json
    echo     "fusion_weights": {>> config.json
    echo       "audio": 0.4,>> config.json
    echo       "face": 0.6>> config.json
    echo     }>> config.json
    echo   },>> config.json
    echo   "knowledge": {>> config.json
    echo     "neo4j": {>> config.json
    echo       "uri": "bolt://localhost:7687",>> config.json
    echo       "user": "neo4j",>> config.json
    echo       "password": "admin123">> config.json
    echo     },>> config.json
    echo     "domain": "计算机科学",>> config.json
    echo     "default_importance": 3,>> config.json
    echo     "default_difficulty": 3>> config.json
    echo   }>> config.json
    echo }>> config.json
    
    echo 默认配置文件已创建，请检查并根据需要修改。
)

REM 启动AI服务器
echo.
echo [1/3] 启动AI服务器...
echo --------------------------------------------
start cmd /k "title NAO教学系统-AI服务器 && call ai_server\.venv\Scripts\activate.bat && python start_ai_server.py --host 0.0.0.0 --port 8765 --web-monitor --web-host 0.0.0.0 --web-port 5000"

REM 等待AI服务器启动
echo 等待AI服务器启动...
timeout /t 5 /nobreak > nul

REM 启动Web监控界面 - 可选
echo.
echo [2/3] 启动Web监控界面...
echo --------------------------------------------
start http://localhost:5000/monitor

REM 启动NAO客户端 - 真实环境使用真实IP
echo.
echo [3/3] 启动NAO客户端...
echo --------------------------------------------
set /p NAO_IP="请输入NAO机器人IP地址 [默认为模拟器127.0.0.1]: "
if "%NAO_IP%"=="" set NAO_IP=127.0.0.1

set /p NAO_PORT="请输入NAO机器人端口 [默认9559]: "
if "%NAO_PORT%"=="" set NAO_PORT=9559

REM 模拟器模式或真实模式
if "%NAO_IP%"=="127.0.0.1" (
    echo 启动NAO模拟器...
    start cmd /k "title NAO教学系统-模拟器 && call nao_control\.venv\Scripts\activate.bat && python nao_simulator_client.py --server-url ws://localhost:8765 --mode demo"
) else (
    echo 连接到真实NAO机器人: %NAO_IP%:%NAO_PORT%
    start cmd /k "title NAO教学系统-客户端 && call nao_control\.venv\Scripts\activate.bat && python start_nao_client.py --ip %NAO_IP% --port %NAO_PORT%"
)

echo.
echo ============================================
echo NAO机器人智能辅助教学系统已启动！
echo ============================================
echo.
echo Web监控地址: http://localhost:5000/monitor
echo AI服务器地址: ws://localhost:8765
echo.
echo 提示：
echo   - 可通过Web监控界面查看系统状态
echo   - 使用Ctrl+C终止各个命令窗口以关闭系统
echo.
pause