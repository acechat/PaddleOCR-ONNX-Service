#include "ocr_detect.h"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <memory>
#include <algorithm>

OCRDetect::OCRDetect(const json& det_config) : det_config_(det_config) {
    std::string path = det_config.at("path").get<std::string>();
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("检测模型路径不存在: " + path);
    }

    // 加载预处理参数
    mean_ = det_config.at("mean").get<std::vector<float>>();
    std_ = det_config.at("std").get<std::vector<float>>();
    if (mean_.size() != 3 || std_.size() != 3) {
        throw std::invalid_argument("mean/std 必须为 3 维");
    }
    is_bgr_ = det_config.value("is_bgr", true);
    min_size_ = det_config.value("min_size", 32);
    max_size_ = det_config.value("max_size", 1536);

    // 输入/输出名和形状
    input_names_ = det_config.at("input_names").get<std::vector<std::string>>();
    output_names_ = det_config.at("output_names").get<std::vector<std::string>>();
    input_shape_ = det_config.at("input_shape").get<std::vector<int64_t>>();

    // 初始化 ONNX
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Detect");
    session_options_.SetIntraOpNumThreads(4);  // 从 service thread_pool_size 动态（假设固定）
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options_.DisableCpuMemArena();
    session_ = Ort::Session(env_, path.c_str(), session_options_);

    // 阈值从 postprocess 层
    json postprocess = det_config.value("postprocess", json::object());  // 若无，空
    det_threshold_ = postprocess.value("det_db_thresh", 0.3f);
    nms_threshold_ = postprocess.value("det_db_box_thresh", 0.6f);  // 示例使用 box_thresh

    spdlog::info("检测模块加载: {} (BGR: {}, min_size: {}, max_size: {})", path, is_bgr_, min_size_, max_size_);
}

OCRDetect::~OCRDetect() = default;

cv::Mat OCRDetect::Preprocess(const cv::Mat& img) {
    if (img.empty()) throw std::invalid_argument("输入图像为空");

    // 动态 resize（短边 min_size，长边 max_size）
    double scale = std::max(static_cast<double>(min_size_) / std::min(img.rows, img.cols),
                            static_cast<double>(max_size_) / std::max(img.rows, img.cols));
    cv::Size new_size(static_cast<int>(img.cols * scale), static_cast<int>(img.rows * scale));
    new_size.width = std::min(new_size.width, max_size_);
    new_size.height = std::min(new_size.height, max_size_);
    cv::Mat resized;
    cv::resize(img, resized, new_size, 0, 0, cv::INTER_LINEAR);

    // Pad to square-ish (for det)
    int pad_h = std::max(0, max_size_ - resized.rows);
    int pad_w = std::max(0, max_size_ - resized.cols);
    cv::copyMakeBorder(resized, resized, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Normalize
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    if (is_bgr_) cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }
    cv::merge(channels, resized);

    // CHW blob (dynamic shape)
    cv::Mat chw;
    cv::dnn::blobFromImage(resized, chw, 1.0, resized.size(), cv::Scalar(), true, false, CV_32F);
    return chw;
}

std::vector<std::vector<float>> OCRDetect::Detect(const cv::Mat& img) {
    std::lock_guard<std::mutex> lock(mutex_);
    cv::Mat input = Preprocess(img);
    std::vector<int64_t> dynamic_shape = input_shape_;  // [1,3,H,W] dynamic H/W
    dynamic_shape[2] = input.size[2];  // H
    dynamic_shape[3] = input.size[3];  // W
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
        spdlog::error("检测推理失败: {}", e.what());
        return {};
    }

    double ratio = static_cast<double>(input.size[3]) / img.cols;  // W ratio
    return Postprocess(output_tensors, img.cols, img.rows, ratio);
}

std::vector<std::vector<float>> OCRDetect::Postprocess(const std::vector<Ort::Value>& outputs, int orig_w, int orig_h, double ratio) {
    if (outputs.empty()) return {};

    auto& output = outputs[0];
    float* prob_map = output.GetTensorMutableData<float>();
    auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int out_h = shape[2], out_w = shape[3];

    // Binary map (DB thresh)
    cv::Mat binary(out_h, out_w, CV_8UC1);
    for (int i = 0; i < out_h * out_w; ++i) {
        binary.at<uchar>(i / out_w, i % out_w) = (prob_map[i] > det_threshold_) ? 255 : 0;
    }

    // Morphology close
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);

    // Contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> boxes;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 10) continue;
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (rect.size.width < min_size_ || rect.size.height < min_size_) continue;
        boxes.push_back(rect);
    }

    // NMS (use nms_threshold)
    std::vector<float> scores(boxes.size(), 0.9f);  // 简化 score
    std::vector<int> indices;
    cv::dnn::NMSBoxesRotated(boxes, scores, det_threshold_, nms_threshold_, indices);

    std::vector<std::vector<float>> bboxes;
    for (int idx : indices) {
        cv::RotatedRect& rect = boxes[idx];
        cv::Point2f pts[4];
        rect.points(pts);
        // 平均到 bbox (简化，实际用 unclip)
        float x1 = std::min({pts[0].x, pts[1].x, pts[2].x, pts[3].x}) / ratio;
        float y1 = std::min({pts[0].y, pts[1].y, pts[2].y, pts[3].y}) / ratio;
        float x2 = std::max({pts[0].x, pts[1].x, pts[2].x, pts[3].x}) / ratio;
        float y2 = std::max({pts[0].y, pts[1].y, pts[2].y, pts[3].y}) / ratio;
        bboxes.emplace_back(std::vector<float>{x1, y1, x2, y2, scores[idx]});
    }

    spdlog::debug("检测到 {} 个文本框 (阈值: {:.2f})", bboxes.size(), det_threshold_);
    return bboxes;
}