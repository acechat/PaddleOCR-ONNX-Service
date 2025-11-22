#ifndef OCR_RECOGNIZE_H
#define OCR_RECOGNIZE_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <json.hpp>
#include <vector>
#include <string>
#include <mutex>

using json = nlohmann::json;

class OCRRecognize {
public:
    OCRRecognize(const json& rec_config);  // 从分层 JSON 初始化
    ~OCRRecognize();
    std::string Recognize(const cv::Mat& img_crop, float& score);  // 返回文本 + score

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    json rec_config_;  // 存储子 config
    std::vector<float> mean_, std_;
    bool is_bgr_;
    int rec_image_height_;
    int rec_batch_num_;
    float rec_threshold_;  // 从 postprocess 层
    std::vector<std::string> dict_;  // 字符字典

    cv::Mat Preprocess(const cv::Mat& img);  // 动态预处理
    std::string Postprocess(const std::vector<Ort::Value>& outputs, float& score);  // CTC decode
    void LoadDict(const std::string& dict_path);
    std::mutex mutex_;  // 线程安全
};

#endif // OCR_RECOGNIZE_H