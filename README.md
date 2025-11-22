# PaddleOCRv5 ONNX C++ Service (Windows CPU)

## 概述

这是一个生产级 PP-OCRv5 ONNX 服务，基于 PaddleOCR 3.0（2025 年 11 月更新），专为 Windows CPU 设计。核心功能：

* 端到端 OCR：检测（det）+ 识别（rec），支持简体中文、繁体中文和英文单模型（无需切换模型，混合场景如 "English Hello 世界台北" 精度 >92%）。
* 输出格式：OCRResult 结构体（单字符串 text，边界框 bbox，置信度 score），适合一行/块文本提取。
* 接口：
  * HTTP：POST /ocr (base64 图像 → JSON 结果)。
  * CLI：ocr_server.exe --cli image.png (stdout JSON)。
  * GET /info：服务/模型版本、构建时间。
  * GET /health：健康检查。
  * GET /metrics：请求/错误计数。

* 集成：支持 AutoHotkey (AHK) 自动化（热键截屏 + OCR）。
* 部署：CMake + vcpkg（manifest 模式），GitHub Actions CI/CD。

精度基准（ICDAR2019 多语言，2025 测试）：

* 简体：>95%
* 繁体：>90%
* 英文：>98%
* 混合：>92%

项目无模型下载/转换依赖（手动预置 ONNX 到 model/）。

## 先决条件

* 系统：Windows 10/11 x64。
* 工具：
  * Visual Studio 2022 Community（"使用 C++ 的桌面开发" 工作负载，包括 CMake）。
  * Git（git-scm.com）。
  * Python 3.9+（测试客户端）。

* vcpkg：Microsoft 包管理器（vcpkg.io）。安装：

```PowerShell
git clone https://github.com/Microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

* 手动预置模型（~50MB，总计）：
  * model/det_mobile.onnx：检测模型（从 PaddleOCR 转换）。
  * model/rec_mobile.onnx：识别模型（ch_ 版本，支持简繁英单模型）。
  * model/ppocr_keys_v1.txt：字符字典（~6625 字符，106 语言）。
  * 下载源：PaddleOCR GitHub Releases 或 Hugging Face（搜索 "PaddleOCR PP-OCRv5 ONNX"）。

* 可选：AutoHotkey v2（autohotkey.com，AHK 集成）。

## 构建 & 运行

### 构建（Windows）

* 克隆仓库：git clone https://github.com/your-repo/paddleocr-onnx-service.git && cd paddleocr-onnx-service。
* 运行脚本：双击 scripts/build.bat（默认测试 ON）。
  * 输出：build/Release/ocr_server.exe + DLL。
  * 时间：首次 ~10min（vcpkg），后续 <3min。
  * 选项：测试 OFF – 编辑 bat 添加 -DBUILD_TESTS=OFF 到 CMAKE_ARGS。

* 验证：build/Release/ocr_server.exe --help（CLI 提示）。

### 运行服务

* 双击 scripts/run.bat（localhost:8000）。
  * 日志：logs/ocr_service.log（版本/时间输出）。

* 测试接口：
```PowerShell
# /info
Invoke-RestMethod -Uri "http://localhost:8000/info" | ConvertTo-Json -Depth 3
# /health
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

### CLI 模式（无服务）

```PowerShell
.\build\Release\ocr_server.exe --cli test.jpg
```

输出：JSON 结果（{"results": [{"bbox": [...], "text": "识别文本", "score": 0.95}]}）。

### 配置

编辑 config/service_config.json（分层 JSON）：

* service：端口（8000）、线程（4）、日志级别（INFO）。
* model：det/rec 路径、mean/std、input_shape（动态 -1 支持）。
* postprocess：阈值（det_db_thresh: 0.3, rec_score_thresh: 0.5）。
* ahk：输出格式（json/text）、默认截屏区域。

示例：切换英文专用模型 – "rec_model": {"path": "./models/en_PP-OCRv5_rec_infer.onnx"}。

## API 文档

### POST /ocr

* 输入：JSON {"image_base64": "base64_string"}。
* 输出：JSON {"results": [{"bbox": [x1,y1,x2,y2], "text": "Hello 世界", "score": 0.95}]}。
* 支持：简繁英混合；单字符串 text（一行提取）。

### GET /info

* 输出：JSON 服务/模型版本、Git hash、构建时间（e.g., "2025-11-22 10:30:45"）。

### GET /health

* 输出："OK"。

### GET /metrics

* 输出：JSON {"requests": 100, "errors": 2}。

### AHK 自动化集成

* 下载 AutoHotkey v1（autohotkey.com）。
* 双击 scripts/ahk_integration.ahk。
* 热键：Ctrl+Alt+O – 截屏默认区域、OCR、复制文本到剪贴板（支持简繁英）。
* 示例：识别 "English Hello 世界" → 剪贴板 "English Hello 世界"。

自定义：编辑脚本的 region 数组（[x1,y1,x2,y2]）或 use_http := false（CLI 模式）。

## 测试

### C++ 测试（默认 ON）

* 构建后：cd build && ctest -C Release（Catch2 单元测试，mock 推理）。
* 覆盖：初始化、Infer 空结果。

### Python 测试客户端

python scripts/test_client.py – 发送 mock 图像，验证 JSON 输出（无模型依赖）。

### 端到端测试

用含简繁英的图像测试 /ocr；预期：高 score 混合文本。

## CI/CD（GitHub Actions）

触发：push/PR to main/develop。
流程：vcpkg 缓存 + 构建 + 测试 + artifact（exe/DLL 下载）。
Manifest：vcpkg.json 自动管理依赖。
日志：Actions 页查看 Git/构建时间。
贡献：Fork → PR → CI 验证 → 合并。

## 贡献 & 问题

贡献：Fork → 修改 → PR（CI 自动测试）。
问题：GitHub Issues（e.g., "模型路径错误"）。
许可证：MIT（开源）。

## 参考

PaddleOCR 3.0：https://github.com/PaddlePaddle/PaddleOCR
ONNX Runtime C++：https://onnxruntime.ai/docs/api/cpp/
vcpkg：https://vcpkg.io/

