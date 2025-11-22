#include "ocr_service.h"
#include <spdlog/spdlog.h>
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

#ifndef GIT_VERSION
#define GIT_VERSION "unknown"
#endif
#ifndef BUILD_TIME
#define BUILD_TIME "unknown"
#endif

int main(int argc, char** argv) {
    spdlog::info("Git Version: {}", GIT_VERSION);
    spdlog::info("Build Time: {}", BUILD_TIME);

    try {
        std::string config_path = (argc > 1) ? argv[1] : "config/service_config.json";
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("无法加载分层配置: " + config_path);
        }
        auto root_config = nlohmann::json::parse(config_file);
        auto service_config = root_config.at("service_config");

        // 服务层加载
        auto service_layer = service_config.at("service");
        std::string service_name = service_layer.value("name", "ppocrv5_onnx_service");
        std::string version = service_layer.value("version", "1.0.0");
        spdlog::info("{} v{} (Git: {}, Build: {}) 启动", service_name, version, GIT_VERSION, BUILD_TIME);

        // 日志初始化（同前）
        auto log_level_str = service_layer.value("log_level", "INFO");
        spdlog::level::level_enum log_level = spdlog::level::from_str(log_level_str);
        spdlog::set_level(log_level);
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
        std::string log_file = service_layer.value("log_file", "logs/ocr_service.log");
        auto logger = spdlog::rotating_logger_mt("ocr_logger", log_file, 10485760, 5);
        spdlog::set_default_logger(logger);

        // CLI 模式支持（同前）
        if (argc > 1 && std::string(argv[1]) == "--cli") {
            if (argc < 3) {
                spdlog::error("CLI 用法: ocr_server.exe --cli <image_path>");
                return -1;
            }
            std::string img_path = argv[2];
            cv::Mat img = cv::imread(img_path);
            if (img.empty()) {
                spdlog::error("图像加载失败: {}", img_path);
                return -1;
            }
            OCRService service(service_config);
            auto results = service.Infer(img);  // 假设 public 或 friend
            std::cout << results.dump(2) << std::endl;
            return 0;
        }

        OCRService service(service_config);
        service.StartServer();

        spdlog::info("服务运行中 (端口: {}, 线程: {})", 
                     service_layer.value("port", 8000), service_layer.value("thread_pool_size", 4));
    } catch (const std::exception& e) {
        spdlog::error("初始化失败: {}", e.what());
        return -1;
    }
    return 0;
}