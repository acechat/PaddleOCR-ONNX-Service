#include "ocr_recognize.h"
#include <spdlog/spdlog.h>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>

OCRRecognize::OCRRecognize(const json& rec_config) : rec_config_(rec_config) {
    std::string path = rec_config.at("path").get<std::string>();
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("识别模型路径不存在: " + path);
    }

    // 加载预处理参数
    mean_ = rec_config.at("mean").get<std::vector<float>>();
    std_ = rec_config.at("std").get<std::vector<float>>();
    if (mean_.size() != 3 || std_.size() != 3) {
        throw std::invalid_argument("mean/std 必须为 3 维");
    }
    is_bgr_ = rec_config.value("is_bgr", true);
    rec_image_height_ = rec_config.value("rec_image_height", 48);
    rec_batch_num_ = rec_config.value("rec_batch_num", 6);

    // 输入/输出名和形状
    input_names_ = rec_config.at("input_names").get<std::vector<std::string>>();
    output_names_ = rec_config.at("output_names").get<std::vector<std::string>>();
    input_shape_ = rec_config.at("input_shape").get<std::vector<int64_t>>();

    // 初始化 ONNX
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Recognize");
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_ = Ort::Session(env_, path.c_str(), session_options_);

    // 字典从 character_dict 层
    json dict_config = rec_config.at("character_dict");
    std::string dict_path = dict_config.at("path").get<std::string>();
    int expected_size = dict_config.value("dict_size", 6625);
    LoadDict(dict_path);
    if (static_cast<int>(dict_.size()) != expected_size) {
        spdlog::warn("字典大小不匹配: {} vs {}", dict_.size(), expected_size);
    }

    // 阈值从 postprocess 层
    json postprocess = rec_config.value("postprocess", json::object());
    rec_threshold_ = postprocess.value("rec_score_thresh", 0.5f);

    spdlog::info("识别模块加载: {} (高度: {}, 字典大小: {})", path, rec_image_height_, dict_.size());
}

OCRRecognize::~OCRRecognize() = default;

void OCRRecognize::LoadDict(const std::string& dict_path) {
    std::ifstream file(dict_path);
    if (!file.is_open()) {
        throw std::runtime_error("字典文件无法打开: " + dict_path);
    }
    std::string line;
    dict_.clear();
    while (std::getline(file, line)) {
        if (!line.empty()) dict_.emplace_back(std::move(line));
    }
    if (dict_.empty()) throw std::runtime_error("字典为空");
}

cv::Mat OCRRecognize::Preprocess(const cv::Mat& img) {
    if (img.empty()) throw std::invalid_argument("输入裁剪图像为空");

    // Resize to height, dynamic width
    double ratio = static_cast<double>(rec_image_height_) / img.rows;
    int target_w = static_cast<int>(img.cols * ratio);
    target_w = std::min(target_w, 320);  // max width
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, rec_image_height_), 0, 0, cv::INTER_LINEAR);

    // Pad width if needed (to multiple of 32? simplify)
    int pad_w = (32 - (target_w % 32)) % 32;
    cv::copyMakeBorder(resized, resized, 0, 0, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    // To grayscale if RGB
    if (resized.channels() == 3) cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);

    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Normalize (3 channels replicate gray)
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = resized.clone();
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }
    cv::merge(channels, resized);

    if (is_bgr_) cv::cvtColor(resized, resized, cv::COLOR_RGB2BGR);  // if needed

    // CHW blob
    cv::Mat chw;
    cv::dnn::blobFromImage(resized, chw, 1.0, resized.size(), cv::Scalar(), true, false, CV_32F);
    return chw;
}

std::string OCRRecognize::Recognize(const cv::Mat& img_crop, float& score) {
    std::lock_guard<std::mutex> lock(mutex_);
    cv::Mat input = Preprocess(img_crop);
    std::vector<int64_t> dynamic_shape = input_shape_;  // [1,3,48,W]
    dynamic_shape[3] = input.size[3];  // dynamic W
    size_t input_size = input.total();

    std::vector<float> input_data(input_size);
    memcpy(input_data.data(), input.ptr<float>(0), input_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_size,
                                                       dynamic_shape.data(), dynamic_shape.size());

    std::vector<Ort::Value> input_tensors{std::move(input_tensor)};
    std::vector<Ort::Value> output_tensors;
    try {
        session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(), input_names_.size(),
                     output_names_.data(), output_names_.size(), &output_tensors);
    } catch (const Ort::Exception& e) {
        spdlog::error("识别推理失败: {}", e.what());
        score = 0.0f;
        return "";
    }

    std::string text = Postprocess(output_tensors, score);
    if (score < rec_threshold_) {
        spdlog::debug("识别分数低: {:.3f} < {:.3f}, 过滤", score, rec_threshold_);
        return "";
    }
    return text;
}

std::string OCRRecognize::Postprocess(const std::vector<Ort::Value>& outputs, float& score) {
    if (outputs.empty()) {
        score = 0.0f;
        return "";
    }

    auto& output = outputs[0];
    float* output_data = output.GetTensorMutableData<float>();
    auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int T = shape[1];  // time steps
    int C = shape[2];  // classes (dict_size + blank)

    score = 0.0f;
    std::vector<int> pred(T);
    for (int t = 0; t < T; ++t) {
        float max_p = -1.0f;
        int max_idx = 0;
        for (int c = 0; c < C; ++c) {
            float p = output_data[t * C + c];
            if (p > max_p) {
                max_p = p;
                max_idx = c;
            }
        }
        pred[t] = max_idx;
        score += max_p / T;
    }

    // CTC decode: remove blanks (last class) and duplicates
    std::string text;
    int prev = -1;
    for (int p : pred) {
        if (p == C - 1) continue;  // blank (last)
        if (p == prev) continue;   // duplicate
        if (p > 0 && p <= static_cast<int>(dict_.size())) {  // valid char
            text += dict_[p - 1];
        }
        prev = p;
    }

    // Trim max length from postprocess
    json postprocess = rec_config_.value("postprocess", json::object());
    int max_len = postprocess.value("max_text_length", 25);
    if (static_cast<int>(text.length()) > max_len) text = text.substr(0, max_len);

    spdlog::debug("识别解码: '{}' (score: {:.3f})", text, score);
    return text;
}