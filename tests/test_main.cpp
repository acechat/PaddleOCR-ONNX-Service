// tests/test_ocr.cpp
#include <catch2/catch.hpp>
#include "ocr_inference.h"  // 头文件从 libocr
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

TEST_CASE("OCR Inference Basic", "[ocr]") {
    // 加载 config (简化，mock 路径)
    json config = json::parse(R"({
        "service_config": {
            "model": {
                "det_model": {"path": "model/det_mobile.onnx"},
                "rec_model": {"path": "model/rec_mobile.onnx", "character_dict": {"path": "model/ppocr_keys_v1.txt"}}
            }
        }
    })");

    // Mock 初始化（假设路径存在，或 skip 实际推理）
    OCRInference inference(config);
    cv::Mat dummy_img(100, 200, CV_8UC3, cv::Scalar(255, 255, 255));  // 白图

    auto result = inference.Infer(dummy_img);
    REQUIRE(result.contains("results"));
    REQUIRE(result["results"].is_array());
    // 预期: 空结果（白图无文本）
    REQUIRE(result["results"].empty() == true);  // 或 >=0，根据实现
}