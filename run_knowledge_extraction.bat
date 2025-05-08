@echo off
chcp 936 >nul
echo ============================================
echo NAO教学系统知识提取流程自动化脚本
echo ============================================

REM 设置路径和变量
set PDF_PATH="教材.pdf"
set OCR_OUTPUT=temp\all_ocr_text.json
set PROCESSED_OCR=temp\all_ocr_text_processed.json
set KG_OUTPUT=output\knowledge_graph.json
set MODEL_PATH=models\deepseek-llm-7b-chat
set DOMAIN="计算机科学"

REM 确保输出目录存在
if not exist temp mkdir temp
if not exist temp_ocr mkdir temp_ocr
if not exist output mkdir output

echo.
echo [1/4] 激活知识提取环境并运行OCR处理...
echo --------------------------------------------
call knowledge_extraction\.venv\Scripts\activate.bat
echo 当前Python环境: %VIRTUAL_ENV%

echo.
echo [2/4] 使用PaddleOCR处理PDF: %PDF_PATH%
python ocr_paddle_cpu_extraction.py --pdf %PDF_PATH% --ocr_lang ch --batch_size 10 --dpi 400

REM 检查OCR结果
if exist %OCR_OUTPUT% (
    echo OCR结果文件生成成功: %OCR_OUTPUT%
) else (
    echo 错误: OCR结果文件生成失败!
    exit /b 1
)

echo.
echo [3/4] 处理OCR结果并生成结构化数据...
python preprocess_book_ocr.py %OCR_OUTPUT%

REM 查看处理后的OCR结果
for %%F in (%PROCESSED_OCR%) do (
    echo 处理后OCR结果文件大小: %%~zF 字节
    if %%~zF LSS 1000 (
        echo 警告: 处理后OCR结果文件内容不足，将使用原始OCR文件
        copy %OCR_OUTPUT% %PROCESSED_OCR%
    )
)

REM 退出知识提取环境
call deactivate
echo OCR处理完成，结果保存至: %PROCESSED_OCR%

echo.
echo [4/4] 激活AI服务器环境并提取知识图谱...
echo --------------------------------------------
call ai_server\.venv\Scripts\activate.bat
echo 当前Python环境: %VIRTUAL_ENV%

echo 开始从OCR结果提取知识图谱...
python direct_extract_knowledge.py --json_file %PROCESSED_OCR% --output %KG_OUTPUT% --model %MODEL_PATH% --use_gpu --retry --domain %DOMAIN% --start_page 25 --max_pages 400

REM 导入知识图谱到Neo4j
echo.
echo 导入知识图谱到Neo4j数据库...
python knowledge_extraction\import_to_neo4j.py --json %KG_OUTPUT% --uri bolt://localhost:7687 --user neo4j --password admin123 --clear

REM 退出AI服务器环境
call deactivate

echo.
echo ============================================
echo 知识提取流程完成!
echo 知识图谱已保存至: %KG_OUTPUT%
echo 知识图谱已导入Neo4j数据库
echo ============================================
echo.
echo 提示：您可以修改脚本中的PDF_PATH变量来处理其他PDF文件
echo 提示：您可以修改DOMAIN变量来设置不同的知识领域
echo.
pause