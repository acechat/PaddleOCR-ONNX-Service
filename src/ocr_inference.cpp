#include "ocr_inference.h"
#include <spdlog/spdlog.h>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>

OCRInference::OCRInference(const json& service_config) : service_config_(service_config) {
    try {
        auto model_layer = service_config.at("model");

        // Det 子层加载
        auto det_config = model_layer.at("det_model");
        detector_ = std::make_unique<OCRDetect>(det_config);

        // Rec 子层加载（合并 character_dict）
        auto rec_config = model_layer.at("rec_model");
        auto dict_config = model_layer.at("character_dict");
        rec_config["character_dict"] = dict_config;  // 注入 dict 到 rec
        auto postprocess_config = model_layer.at("postprocess");
        rec_config["postprocess"] = postprocess_config;  // 注入 postprocess
        recognizer_ = std::make_unique<OCRRecognize>(rec_config);

        // Cls 子层（可选）
        auto cls_config = model_layer.at("cls_model");
        std::string cls_path = cls_config.at("path").get<std::string>();
        if (!cls_path.empty() && std::filesystem::exists(cls_path)) {
            // cls_ = std::make_unique<OCRCls>(cls_config);  // 若实现 OCRCls
            spdlog::info("方向分类模块启用: {}", cls_path);
        } else {
            spdlog::info("方向分类模块禁用");
        }

        spdlog::info("OCR 推理管道初始化完成 (det: {}, rec: {}, dict: {})",
                     det_config.at("path"), rec_config.at("path"), dict_config.at("path"));
    } catch (const std::exception& e) {
        spdlog::error("OCR 管道初始化失败: {}", e.what());
        throw;
    }
}

json OCRInference::Infer(const cv::Mat& img) {
    std::lock_guard<std::mutex> lock(mutex_);  // 线程安全
    if (img.empty()) {
        spdlog::warn("输入图像为空");
        return json{{"results", json::array()}};
    }

    auto results = RunPipeline(img);

    json response;
    response["results"] = json::array();
    for (const auto& res : results) {
        json j_res;
        j_res["bbox"] = res.bbox;
        j_res["text"] = res.text;
        j_res["score"] = res.score;
        response["results"].push_back(j_res);
    }

    // 从 postprocess 层获取 max_text_length（已集成到 recognize）
    auto postprocess = service_config_.at("model").at("postprocess");
    int max_len = postprocess.value("max_text_length", 25);
    spdlog::info("OCR 推理完成: {} 结果 (max_len: {})", results.size(), max_len);
    return response;
}

std::vector<OCRResult> OCRInference::RunPipeline(const cv::Mat& img) {
    std::vector<OCRResult> results;

    // 1. 检测
    auto bboxes = detector_->Detect(img);
    if (bboxes.empty()) {
        spdlog::debug("未检测到文本框");
        return results;
    }

    // 2. 方向分类（可选，若启用 cls_）
    // for each bbox: float angle = cls_->Classify(crop); rotate if needed

    // 3. 识别
    for (const auto& bbox : bboxes) {
        if (bbox.size() != 5) continue;  // [x1,y1,x2,y2,score]
        cv::Rect roi(static_cast<int>(bbox[0]), static_cast<int>(bbox[1]),
                     static_cast<int>(bbox[2] - bbox[0]), static_cast<int>(bbox[3] - bbox[1]));
        if (roi.area() <= 0 || roi.x < 0 || roi.y < 0) continue;

        cv::Mat crop = img(roi);
        if (crop.empty()) continue;

        float rec_score = 0.0f;
        std::string text = recognizer_->Recognize(crop, rec_score);
        if (text.empty() || rec_score < 0.1f) continue;  // 最小阈值

        OCRResult res;
        res.bbox = {bbox[0], bbox[1], bbox[2], bbox[3]};
        res.text = std::move(text);
        res.score = std::max(bbox[4], rec_score);  // 取最大分数（det or rec）

        results.push_back(std::move(res));
    }

    // 4. 排序（可选，按 y 坐标）
    std::sort(results.begin(), results.end(), [](const OCRResult& a, const OCRResult& b) {
        return a.bbox[1] < b.bbox[1];  // top-to-bottom
    });

    return results;
}