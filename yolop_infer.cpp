// yolop_infer.cpp
//
// Generic ONNX Runtime + OpenCV C++ inference example for a segmentation ONNX model.
// Loads an ONNX model, runs inference on all images in a directory, saves output masks.
//
// Build (example):
// g++ -std=c++17 yolop_infer.cpp \
//     -I/path/to/onnxruntime/include -L/path/to/onnxruntime/lib \
//     -lonnxruntime -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
//     -o yolop_infer

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono> // Add this include for timing

#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// Letterbox resize: keep aspect ratio, pad to (target_w, target_h) with color 114
cv::Mat letterbox(const cv::Mat& src,
                  int target_w, int target_h,
                  int& pad_w, int& pad_h,
                  float& scale) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)target_w / w, (float)target_h / h);
    int new_w = int(w * scale), new_h = int(h * scale);
    pad_w = (target_w - new_w) / 2;
    pad_h = (target_h - new_h) / 2;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    cv::Mat canvas(target_h, target_w, src.type(), cv::Scalar(114,114,114));
    resized.copyTo(canvas(cv::Rect(pad_w, pad_h, new_w, new_h)));
    return canvas;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <input_dir> <output_dir> <input_w> <input_h>\n";
        return 1;
    }

    const char* model_path = argv[1];
    fs::path input_dir  = argv[2];
    fs::path output_dir = argv[3];
    int input_w = std::stoi(argv[4]);
    int input_h = std::stoi(argv[5]);

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: input_dir does not exist or is not a directory\n";
        return 1;
    }
    fs::create_directories(output_dir);

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolop");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(2);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path, session_opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input names (keep AllocatedStringPtr alive)
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_names(num_input_nodes);
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    input_name_ptrs.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; i++) {
        input_name_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
        input_names[i] = input_name_ptrs.back().get();
    }

    // Get output names (keep AllocatedStringPtr alive)
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> output_names(num_output_nodes);
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    output_name_ptrs.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; i++) {
        output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
        output_names[i] = output_name_ptrs.back().get();
    }

    // Print input and output names
    std::cout << "=== Model I/O Names ===\n";
    std::cout << "Inputs:\n";
    for (size_t i = 0; i < input_names.size(); ++i)
        std::cout << "  [" << i << "] " << input_names[i] << "\n";
    std::cout << "Outputs:\n";
    for (size_t i = 0; i < output_names.size(); ++i)
        std::cout << "  [" << i << "] " << output_names[i] << "\n";

    // Prepare MemoryInfo for tensor creation
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault);

    // Iterate through input images
    for (auto& entry : fs::directory_iterator(input_dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

        cv::Mat img_bgr = cv::imread(entry.path().string());
        if (img_bgr.empty()) {
            std::cerr << "Failed to read image: " << entry.path() << "\n";
            continue;
        }
        int orig_w = img_bgr.cols, orig_h = img_bgr.rows;

        // Preprocess
        int pad_w, pad_h; float scale;
        cv::Mat canvas = letterbox(img_bgr, input_w, input_h, pad_w, pad_h, scale);
        cv::Mat img_rgb;
        cv::cvtColor(canvas, img_rgb, cv::COLOR_BGR2RGB);
        img_rgb.convertTo(img_rgb, CV_32F, 1.0f / 255.0f);
        static const std::array<float,3> mean = {0.485f, 0.456f, 0.406f};
        static const std::array<float,3> std  = {0.229f, 0.224f, 0.225f};
        std::vector<cv::Mat> ch(3);
        cv::split(img_rgb, ch);
        for (int c = 0; c < 3; c++)
            ch[c] = (ch[c] - mean[c]) / std[c];
        cv::merge(ch, img_rgb);

        // HWC -> CHW
        std::vector<float> input_tensor_values(3 * input_h * input_w);
        size_t idx = 0;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < input_h; y++)
                for (int x = 0; x < input_w; x++)
                    input_tensor_values[idx++] = img_rgb.at<cv::Vec3f>(y, x)[c];

        // Create input tensor
        std::array<int64_t,4> input_shape{1,3,input_h,input_w};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        // pick the drivable‐area segmentation head (index 1)
        const char* seg_name = output_names[1];

        // Measure inference time
        auto start_time = std::chrono::high_resolution_clock::now(); // Start timer

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(), &input_tensor, 1,
            &seg_name,                // pass pointer to the single desired output name
            1);                       // 1 output

        auto end_time = std::chrono::high_resolution_clock::now(); // End timer
        auto inference_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Inference time: " << inference_duration.count() << " ms\n";

        // Process output
        auto& out_tensor = output_tensors.front();
        auto info = out_tensor.GetTensorTypeAndShapeInfo();
        auto dims = info.GetShape();  // {1,C,H_out,W_out}
        int C     = (int)dims[1],
            H_out = (int)dims[2],
            W_out = (int)dims[3];
        const float* out_data = out_tensor.GetTensorData<float>();

        cv::Mat mask(H_out, W_out, CV_8UC1);
        if (C == 1) {
            for (int y = 0; y < H_out; y++)
                for (int x = 0; x < W_out; x++)
                    mask.at<uint8_t>(y,x) = (out_data[y*W_out + x] > 0.5f) ? 255 : 0;
        } else {
            for (int y = 0; y < H_out; y++) {
                for (int x = 0; x < W_out; x++) {
                    int best_c = 0;
                    float best_v = out_data[0 * H_out * W_out + y * W_out + x];
                    for (int c = 1; c < C; c++) {
                        float v = out_data[c * H_out * W_out + y * W_out + x];
                        if (v > best_v) { best_v = v; best_c = c; }
                    }
                    mask.at<uint8_t>(y,x) = (best_c > 0) ? 255 : 0;
                }
            }
        }

        // Resize & save
        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
        fs::path out_path = output_dir / entry.path().filename();
        out_path.replace_extension("_mask.png");
        cv::imwrite(out_path.string(), mask_resized);
        std::cout << "Saved mask to " << out_path << "\n";
    }

    return 0;
}
