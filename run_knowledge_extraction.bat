@echo off
echo NAO教学系统知识提取流程启动...

REM 设置路径
set PDF_PATH="C程序设计第5版.pdf"
set OCR_OUTPUT=temp\all_ocr_text.json
set PROCESSED_OCR=temp\all_ocr_text_processed.json
set KG_OUTPUT=output\knowledge_graph.json
set MODEL_PATH=models\deepseek-llm-7b-chat

REM 确保输出目录存在
if not exist temp mkdir temp
if not exist temp_ocr mkdir temp_ocr
if not exist output mkdir output

REM 步骤1: 激活knowledge_extraction环境并运行OCR
echo 步骤1: 启动OCR处理 - 使用PaddleOCR CPU模式进行文本识别
call knowledge_extraction\.venv\Scripts\activate.bat
echo 当前Python环境: %VIRTUAL_ENV%

echo 使用PaddleOCR CPU模式处理PDF: %PDF_PATH%
python ocr_paddle_cpu_extraction.py --pdf "%PDF_PATH%" --ocr_lang ch --batch_size 5 --dpi 400

REM 检查OCR结果
echo 检查OCR结果...
if exist %OCR_OUTPUT% (
    echo OCR结果文件存在: %OCR_OUTPUT%
) else (
    echo 错误: OCR结果文件不存在!
    exit /b 1
)

REM 运行OCR后处理
echo 进行OCR结果后处理...
python preprocess_book_ocr.py %OCR_OUTPUT%

REM 查看处理后的OCR结果大小
for %%F in (%PROCESSED_OCR%) do (
    echo 处理后OCR结果文件大小: %%~zF 字节
    if %%~zF LSS 1000 (
        echo 警告: 处理后OCR结果文件可能为空或内容不足!
        echo 将使用原始OCR结果
        copy %OCR_OUTPUT% %PROCESSED_OCR%
    )
)

REM 退出knowledge_extraction虚拟环境
call deactivate
echo OCR处理完成，结果保存至: %PROCESSED_OCR%

REM 步骤2: 激活ai_server环境并运行知识图谱提取
echo 步骤2: 启动知识图谱提取 - 使用DeepSeek LLM
call ai_server\.venv\Scripts\activate.bat
echo 当前Python环境: %VIRTUAL_ENV%

echo 从OCR结果提取知识图谱...
python direct_extract_knowledge.py --json_file %PROCESSED_OCR% --output %KG_OUTPUT% --model %MODEL_PATH% --use_gpu --retry --domain "计算机科学"

REM 退出ai_server虚拟环境
call deactivate

echo 知识提取流程完成！
echo 知识图谱已保存至: %KG_OUTPUT%
echo.
echo 您可以使用知识图谱导入到Neo4j数据库：
echo python knowledge_extraction\import_to_neo4j.py --json %KG_OUTPUT% --uri bolt://localhost:7687 --user neo4j --password [您的密码] --clear