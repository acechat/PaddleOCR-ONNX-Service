@echo off
echo Building PaddleOCR ONNX Service (模型预置模式)...

REM 检查 vcpkg
if not exist "C:\vcpkg" (
    echo vcpkg not found. Install from https://vcpkg.io/
    pause
    exit /b 1
)

REM 检查模型（可选验证）
if not exist "model\det_mobile.onnx" (
    echo 警告: model\det_mobile.onnx 未找到。请手动预置。
)
if not exist "model\rec_mobile.onnx" (
    echo 警告: model\rec_mobile.onnx 未找到。请手动预置。
)
if not exist "model\ppocr_keys_v1.txt" (
    echo 警告: model\ppocr_keys_v1.txt 未找到。请手动预置。
)

REM 构建选项：默认测试 ON，可添加 -DBUILD_TESTS=OFF
set CMAKE_ARGS=-G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

if exist build rmdir /s /q build
mkdir build
cd build

cmake .. %CMAKE_ARGS%
if %errorlevel% neq 0 (
    echo CMake 配置失败！
    pause
    exit /b %errorlevel%
)

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo 构建失败！
    pause
    exit /b %errorlevel%
)

REM 测试（仅若 BUILD_TESTS=ON）
if exist tests (
    ctest -C Release --output-on-failure
    if %errorlevel% neq 0 (
        echo 测试失败！
        pause
        exit /b %errorlevel%
    )
)

echo 构建成功！ exe 在 build/Release/ocr_server.exe
cd ..
pause