@echo off
cd ..  REM 回根目录
set MODEL_DIR=model/
set PORT=8080

REM 可选：从 config/ 读取（简化，硬编码）
echo 启动服务，模型目录: %MODEL_DIR%，端口: %PORT%

REM 运行 exe（需先构建）
build\Release\ocr_server.exe %MODEL_DIR% %PORT%

pause