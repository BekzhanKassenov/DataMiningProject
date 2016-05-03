#include <iostream>
#include <cv.h>
#include <highgui.h>

namespace cardet {
    const std::string videoOutputName = "Video";
    const std::string detectedOutputName = "After detection";


    std::string filename = "data/video.avi";
    std::string trainedData = "trainedData/HaarFeatures.xml";
}

int main(int argc, char **argv) {
    // Parsing command line arguments
    switch (argc) {
        case 1:
            break;

        case 2:
            cardet::filename = argv[1];
            break;

        case 3:
            cardet::filename = argv[1];
            cardet::trainedData = argv[2];
            break;

        default:
            std::cout << "Usage: carrecognition [video] [config]" << std::endl;
            return 1;
    }

    std::cout << "Opening " << cardet::filename << std::endl;

    cv::VideoCapture input(cardet::filename);

    if (!input.isOpened()) {
        std::cerr << "Cannot open video!" << std::endl;
        return 1;
    }

    cv::Mat frame;
    cv::CascadeClassifier classifier(cardet::trainedData);
    while (true) {
        input >> frame;

        if (frame.empty()) {
            break;
        }

        std::vector <cv::Rect> detectedObjects;

        classifier.detectMultiScale(
                frame,
                detectedObjects, // Result of detection
                1.05,            // Scale at each iteration
                1,               // Min neighbours
                0,               // not used
                cv::Size(0, 0),  // minimal size of the object
                frame.size()     // maximal size of the object
        );

        cv::Mat detected(frame);

        for (const auto& rect: detectedObjects) {
            cv::rectangle(detected, rect, CV_RGB(255, 0, 0), 2);
        }

        cv::imshow(cardet::videoOutputName, frame);
        cv::imshow(cardet::detectedOutputName, detected);
        cv::waitKey(20);
    }

    return 0;
}