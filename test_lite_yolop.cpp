#include "lite/lite.h"
#include <filesystem>
#include <chrono>
#include <vector>

// Set to true if you only want to process and time the drivable area segmentation (DA) output.
constexpr bool DA_ONLY = true;

void process_images_in_folder(const std::string &onnx_path,
                              const std::string &input_folder,
                              const std::string &output_folder) {
    // Create the YOLOP model object with 16 threads
    lite::cv::detection::YOLOP *yolop = new lite::cv::detection::YOLOP(onnx_path, 16);

    // Ensure the output folder exists
    std::filesystem::create_directories(output_folder);

    // Vector to store DA inference times (in milliseconds)
    std::vector<double> inference_times;

    // Loop through all images in the input folder
    for (const auto &entry : std::filesystem::directory_iterator(input_folder)) {
        if (!entry.is_regular_file())
            continue; // Skip non-file entries

        std::string input_path = entry.path().string();
        std::string filename   = entry.path().stem().string();
        std::cout << "Processing: " << input_path << std::endl;

        // Construct the output file path for drivable area segmentation results
        std::string save_da_path = output_folder + "/" + filename + "_da.jpg";

        // Read the input image
        cv::Mat img_bgr = cv::imread(input_path);
        if (img_bgr.empty()) {
            std::cerr << "Error: Unable to read image: " << input_path << std::endl;
            continue;
        }

        // Measure inference time for the DA segmentation output
        auto start_time = std::chrono::high_resolution_clock::now();

        lite::types::SegmentContent da_seg_content;
        if (DA_ONLY) {
            // Create dummy variables for detection boxes and lane segmentation,
            // since we only care about DA segmentation.
            std::vector<lite::types::Boxf> dummy_boxes;
            lite::types::SegmentContent dummy_ll_seg_content;
            yolop->detect(img_bgr, dummy_boxes, da_seg_content, dummy_ll_seg_content);
        } else {
            // Full inference: you may process all outputs if needed.
            std::vector<lite::types::Boxf> detected_boxes;
            lite::types::SegmentContent ll_seg_content;
            yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        inference_times.push_back(elapsed_time_ms);

        // Save the drivable area segmentation result if available
        if (da_seg_content.flag) {
            cv::imwrite(save_da_path, da_seg_content.class_mat);
            std::cout << "Saved " << save_da_path << " done!" << "\n";
        } else {
            std::cout << "DA segmentation failed for " << input_path << "\n";
        }
    }

    // Print the DA inference times
    std::cout << "\nDA Inference Times (ms):\n";
    for (size_t i = 0; i < inference_times.size(); ++i) {
        std::cout << "Image " << i + 1 << ": " << inference_times[i] << " ms\n";
    }

    delete yolop;
}

int main() {
    // Model and directory setup
    std::string onnx_path    = "yolop-640-640.onnx"; // Ensure this file is in the same directory as the binary
    std::string input_folder = "./Images";           // Folder containing input images
    std::string output_folder = "./Output";          // Folder to save output results

    std::cout << "Using model: " << onnx_path << "\n";
    std::cout << "Input folder: " << input_folder << "\n";
    std::cout << "Output folder: " << output_folder << "\n";

    // Process the images in the folder (DA-only inference mode)
    process_images_in_folder(onnx_path, input_folder, output_folder);

    return 0;
}
