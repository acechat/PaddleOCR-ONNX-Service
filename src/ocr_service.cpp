#include "ocr_service.h"
#include <spdlog/spdlog.h>
#include <json.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <mutex>
#include <thread>
#include <fstream>
#include <filesystem>

OCRService::OCRService(const json& service_config) : service_config_(service_config) {
    auto service_layer = service_config.at("service");
    int thread_size = service_layer.value("thread_pool_size", 4);
    // 设置 ONNX threads (假设 inference_ 初始化时传递)
    max_size_ = service_layer.value("max_batch_size", 8) * 1024 * 1024;

    try {
        inference_ = std::make_unique<OCRInference>(service_config);
    } catch (const std::exception& e) {
        spdlog::error("OCR 管道初始化失败: {}", e.what());
        throw;
    }
    spdlog::info("服务配置加载完成 (模型: {})", service_config.at("model").at("rec_model").at("path"));
}

void OCRService::StartServer() {
    auto service_layer = service_config_.at("service");
    int port = service_layer.value("port", 8000);
    int timeout = service_layer.value("timeout_ms", 30000);

    httplib::Server svr;
    svr.set_timeout(timeout / 1000, 0);  // sec, usec

    // /ocr
    svr.Post("/ocr", [this](const httplib::Request& req, httplib::Response& res) {
        ocr_handler(req, res);
    });

    // /info
    svr.Get("/info", [this](const httplib::Request&, httplib::Response& res) {
        info_handler({}, res);
    });

    // /health
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("OK", "text/plain");
    });

    // /metrics
    svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        json metrics = {{"requests", request_count_}, {"errors", error_count_}};
        res.set_content(metrics.dump(), "application/json");
    });

    // 线程池
    if (service_layer.value("use_multithread", true)) {
        int thread_size = service_layer["thread_pool_size"];
        svr.new_task_queue = [thread_size]() { return new httplib::ThreadPool(thread_size); };
    }

    spdlog::info("服务器监听端口: {}", port);
    if (!svr.listen("0.0.0.0", port)) {
        spdlog::error("服务器启动失败");
    }
}

json OCRService::Infer(const cv::Mat& img) {
    return inference_->Infer(img);
}

void OCRService::ocr_handler(const httplib::Request& req, httplib::Response& res) {
    request_count_++;
    try {
        if (req.body.size() > max_size_) {
            res.status = 413;
            res.set_content("图像过大", "text/plain");
            return;
        }
        json j = json::parse(req.body);
        if (!j.contains("image_base64") || j["image_base64"].empty()) {
            res.status = 400;
            res.set_content("缺少 image_base64", "text/plain");
            return;
        }

        std::string base64_img = j["image_base64"];
        std::string decoded = base64_decode(base64_img);
        if (decoded.size() > max_size_) throw std::runtime_error("解码后过大");

        std::vector<uchar> img_data(decoded.begin(), decoded.end());
        cv::Mat img = cv::imdecode(img_data, cv::IMREAD_COLOR);
        if (img.empty()) {
            res.status = 400;
            res.set_content("无效图像", "text/plain");
            return;
        }

        auto results = inference_->Infer(img);
        json response {{"results", results["results"]}};
        res.set_content(response.dump(2), "application/json");
        spdlog::info("处理请求成功: {} 结果", results["results"].size());
    } catch (const std::exception& e) {
        error_count_++;
        spdlog::error("处理失败: {}", e.what());
        res.status = 500;
        res.set_content("内部错误: " + std::string(e.what()), "text/plain");
    }
}

void OCRService::info_handler(const httplib::Request&, httplib::Response& res) {
    try {
        json info = GetInfo();
        res.set_content(info.dump(2), "application/json");
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content("{\"error\": \"" + std::string(e.what()) + "\"}", "application/json");
    }
}

json OCRService::GetInfo() {
    json info;
    auto service_layer = service_config_.at("service");
    info["service"] = {
        {"name", service_layer.value("name", "ppocrv5_onnx_service")},
        {"version", service_layer.value("version", "1.0.0")},
        {"git_version", GIT_VERSION},
        {"build_time", BUILD_TIME}
    };

    // 模型信息
    auto model_layer = service_config_.at("model");
    json models;
    auto det_config = model_layer.at("det_model");
    std::string det_path = det_config.at("path");
    models["det"] = {{"path", det_path}};
    // 读取 ONNX metadata
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Info");
        Ort::SessionOptions opts;
        Ort::Session session(env, det_path.c_str(), opts);
        const auto& meta = session.GetModelMetadata();
        const char* producer_ver = nullptr;
        meta.LookupCustomMetadataMap("producer_version", &producer_ver);
        models["det"]["version"] = producer_ver ? producer_ver : "unknown";
        models["det"]["op_version"] = meta.GetOpsetVersion();
    } catch (...) {
        models["det"]["version"] = "unknown";
        models["det"]["op_version"] = -1;
    }

    // Rec 类似
    auto rec_config = model_layer.at("rec_model");
    std::string rec_path = rec_config.at("path");
    models["rec"] = {{"path", rec_path}};
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Info");
        Ort::SessionOptions opts;
        Ort::Session session(env, rec_path.c_str(), opts);
        const auto& meta = session.GetModelMetadata();
        const char* producer_ver = nullptr;
        meta.LookupCustomMetadataMap("producer_version", &producer_ver);
        models["rec"]["version"] = producer_ver ? producer_ver : "unknown";
        models["rec"]["op_version"] = meta.GetOpsetVersion();
    } catch (...) {
        models["rec"]["version"] = "unknown";
        models["rec"]["op_version"] = -1;
    }

    // Dict
    auto dict_config = model_layer.at("character_dict");
    models["dict"] = {{"path", dict_config.at("path")}, {"version", "v1"}};

    info["models"] = models;
    return info;
}

std::string OCRService::base64_decode(const std::string& encoded) {
    // 完整 Base64 解码（处理 padding）
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string decoded;
    int val = 0, valb = -8;
    for (char c : encoded) {
        if (c == '=') break;  // padding
        size_t pos = chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        valb += 6;
        if (valb >= 0) {
            decoded.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return decoded;
}