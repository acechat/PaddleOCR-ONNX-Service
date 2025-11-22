# MODEL_SETUP.md（模型下载、转换与预置指南）

## 概述
本项目假设 ONNX 模型和字典文件已手动预置到 model/ 目录（生产部署最佳实践）。本文件提供 PP-OCRv5 模型的下载、转换（PaddlePaddle → ONNX）和预置步骤。支持简体/繁体/英文单模型（ch_PP-OCRv5_rec_infer），适用于 Windows CPU。

注意：

* 模型大小：~50MB（mobile 版，轻量）。
* 转换工具：Paddle2ONNX（pip install paddle2onnx）。
* 环境：Python 3.9+ + PaddlePaddle CPU 版（pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/）。

## 1. 下载原始模型
使用 PowerShell（管理员模式）下载官方 tar 包（PaddleOCR 3.0 发布，2025 年 11 月）。

```PowerShell
# 创建目录
mkdir model
cd model

# 检测模型（det）
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/PP-OCRv5/ch_PP-OCRv5_mobile_det_infer.tar" -OutFile "det.tar"

# 识别模型（rec，支持简繁英单模型）
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/PP-OCRv5/ch_PP-OCRv5_mobile_rec_infer.tar" -OutFile "rec.tar"

# 方向分类（cls，可选）
Invoke-WebRequest -Uri "https://paddleocr.bj.bcebos.com/PP-OCRv5/ch_ppocr_mobile_v2.0_cls_infer.tar" -OutFile "cls.tar"

# 字符字典（ppocr_keys_v1.txt，支持 106 语言，包括简繁英）
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/3.0/ppocr/utils/ppocr_keys_v1.txt" -OutFile "ppocr_keys_v1.txt"
```

解压（使用 7-Zip 或 PowerShell tar）：

````PowerShell
# 解压（需 7-Zip 或 conda install tar）
tar -xf det.tar
tar -xf rec.tar
tar -xf cls.tar  # 可选

# 提取 inference.pdmodel / pdiparams
# 目录结构示例：
# model/
# ├── ch_PP-OCRv5_mobile_det_infer/
# │   ├── inference.pdmodel
# │   └── inference.pdiparams
# ├── ch_PP-OCRv5_mobile_rec_infer/  # 同上
# ├── ch_ppocr_mobile_v2.0_cls_infer/  # 可选
# └── ppocr_keys_v1.txt

````

替代：从 PaddleOCR GitHub Releases 或 Hugging Face 下载预转换 ONNX（搜索 "PP-OCRv5 ONNX"）。

## 2. 转换到 ONNX

使用 Paddle2ONNX 工具（pip install paddle2onnx）。在 Anaconda Prompt（激活 paddleocr 环境）运行：

```PowerShell
# 检测模型
paddle2onnx --model_dir model\ch_PP-OCRv5_mobile_det_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file model\det_mobile.onnx --opset_version 13 --enable_onnx_checker True --dynamic_shape x@1,3,*,*

# 识别模型
paddle2onnx --model_dir model\ch_PP-OCRv5_mobile_rec_infer --save_file model\rec_mobile.onnx --opset_version 13 --enable_onnx_checker True --dynamic_shape x@1,3,48,*

# 方向分类（可选）
paddle2onnx --model_dir model\ch_ppocr_mobile_v2.0_cls_infer --save_file model\cls_mobile.onnx --opset_version 13 --enable_onnx_checker True

# 清理原始 tar/dir（可选）
Remove-Item -Recurse -Force ch_PP-OCRv5_mobile_det_infer.tar, ch_PP-OCRv5_mobile_det_infer, ...  # 列出文件
```

参数说明：

--opset_version 13：ONNX 标准（兼容 ONNXRuntime 1.18+）。
--dynamic_shape：支持动态输入（* 为 H/W）。
输出：model/det_mobile.onnx 等（大小 ~10-20MB/文件）。
验证：python -c "import onnx; onnx.checker.check_model(onnx.load('model/det_mobile.onnx'))"。

常见问题：

算子不支持：升级 paddle2onnx (pip install --upgrade paddle2onnx)；添加 --enable_dev_version True。
路径含空格：用双引号包围。
英文专用：下载 en_PP-OCRv5_mobile_rec_infer.tar，转换到 rec_en_mobile.onnx（config 中切换路径）。

## 3. 预置与配置

* 放置文件：确保 model/ 包含：
  * det_mobile.onnx
  * rec_mobile.onnx（简繁英）
  * ppocr_keys_v1.txt

* 更新 config/service_config.json：
```JSON
{
  "service_config": {
    "model": {
      "det_model": {"path": "./models/det_mobile.onnx", ...},
      "rec_model": {"path": "./models/rec_mobile.onnx", ...},
      "character_dict": {"path": "./models/ppocr_keys_v1.txt"}
    }
  }
}
```

* 构建 & 测试：
  * scripts/build.bat（验证预置）。
  * scripts/run.bat 启动。
  * 测试：python scripts/test_client.py（需图像）。


## 故障排除


|问题|解决|
|--|--|
|下载慢|"用 VPN 或镜像（e.g., tsinghua.bcebos.com）。"|
|转换失败|检查 PaddlePaddle 版本（3.0.0 CPU）；报告 GitHub Issue。|
|字典不匹配|确认 ppocr_keys_v1.txt 行数 ~6625（head -n 10 检查）。|
|简繁英精度低|用 server 版模型（更大，精度 +5%）；调整 rec_threshold: 0.5。|


参考：PaddleOCR 文档（https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.0/doc/doc_en/deployment_en.md）。更



















问题解决下载慢用 VPN 或镜像（e.g., tsinghua.bcebos.com）。转换失败检查 PaddlePaddle 版本（3.0.0 CPU）；报告 GitHub Issue。字典不匹配确认 ppocr_keys_v1.txt 行数 ~6625（head -n 10 检查）。简繁英精度低用 server 版模型（更大，精度 +5%）；调整 rec_threshold: 0.5。
参考：PaddleOCR 文档（https://github.com/PaddlePaddle/PaddleOCR/blob/release/3.0/doc/doc_en/deployment_en.md）。更新日期：2025-11-22。