#include "lite/lite.h"
#include <filesystem>

void process_images_in_folder(const std::string &onnx_path, const std::string &input_folder, const std::string &output_folder) {
    // Create the YOLOP model object
    lite::cv::detection::YOLOP *yolop = new lite::cv::detection::YOLOP(onnx_path, 16); // 16 threads

    // Ensure output folder exists
    std::filesystem::create_directories(output_folder);

    // Loop through all images in the input folder
    for (const auto &entry : std::filesystem::directory_iterator(input_folder)) {
        if (!entry.is_regular_file()) continue; // Skip non-file entries

        std::string input_path = entry.path().string();
        std::string filename = entry.path().stem().string();
        std::cout << "Processing: " << input_path << std::endl;

        // Construct output file paths
        std::string save_det_path = output_folder + "/" + filename + "_det.jpg";
        std::string save_da_path = output_folder + "/" + filename + "_da.jpg";
        std::string save_ll_path = output_folder + "/" + filename + "_ll.jpg";
        std::string save_merge_path = output_folder + "/" + filename + "_merge.jpg";

        // Read input image
        cv::Mat img_bgr = cv::imread(input_path);
        if (img_bgr.empty()) {
            std::cerr << "Error: Unable to read image: " << input_path << std::endl;
            continue;
        }

        // Perform inference
        lite::types::SegmentContent da_seg_content;
        lite::types::SegmentContent ll_seg_content;
        std::vector<lite::types::Boxf> detected_boxes;
        yolop->detect(img_bgr, detected_boxes, da_seg_content, ll_seg_content);

        // Save results if detection and segmentation succeed
        if (!detected_boxes.empty() && da_seg_content.flag && ll_seg_content.flag) {
            // Save detection results
            cv::Mat img_det = img_bgr.clone();
            lite::utils::draw_boxes_inplace(img_det, detected_boxes);
            cv::imwrite(save_det_path, img_det);
            std::cout << "Saved " << save_det_path << " done!" << "\n";

            // Save segmentation results
            cv::imwrite(save_da_path, da_seg_content.class_mat);
            cv::imwrite(save_ll_path, ll_seg_content.class_mat);
            std::cout << "Saved " << save_da_path << " done!" << "\n";
            std::cout << "Saved " << save_ll_path << " done!" << "\n";

            // Save merged results
            cv::Mat img_merge = img_bgr.clone();
            cv::Mat color_seg = da_seg_content.color_mat + ll_seg_content.color_mat;
            cv::addWeighted(img_merge, 0.5, color_seg, 0.5, 0., img_merge);
            lite::utils::draw_boxes_inplace(img_merge, detected_boxes);
            cv::imwrite(save_merge_path, img_merge);
            std::cout << "Saved " << save_merge_path << " done!" << "\n";
        } else {
            std::cout << "No detections or segmentation for " << input_path << "\n";
        }
    }

    delete yolop;
}

int main() {
    // Model and directory setup
    std::string onnx_path = "yolop-640-640.onnx";  // Ensure this file is in the same directory as the binary
    std::string input_folder = "./Images";         // Folder containing input images
    std::string output_folder = "./Output";        // Folder to save output results

    // Process the images in the folder
    std::cout << "Using model: " << onnx_path << "\n";
    std::cout << "Input folder: " << input_folder << "\n";
    std::cout << "Output folder: " << output_folder << "\n";

    process_images_in_folder(onnx_path, input_folder, output_folder);

    return 0;
}
