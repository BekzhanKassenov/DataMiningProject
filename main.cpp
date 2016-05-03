#include <iostream>
#include <cv.h>
#include <highgui.h>

namespace cardet {
    // Names of the windows with video
    const std::string videoWindowName = "Video";
    const std::string detectedWindowName = "After detection";

    // Output file
    const std::string videoOutputName = "/home/bekzhan/Programming/DataMiningProject/output.avi";

    // File with input data
    std::string filename = "data/video.avi";

    // File with Haar classifier data
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

    /* Initializing input stream */
    cv::VideoCapture input(cardet::filename);

    if (!input.isOpened()) {
        std::cerr << "Cannot open video!" << std::endl;
        return 1;
    }

    /* Opening output stream */
    int fourcc = (int) input.get(CV_CAP_PROP_FOURCC);

    double fps = input.get(CV_CAP_PROP_FPS);

    cv::Size videoSize((int) input.get(CV_CAP_PROP_FRAME_WIDTH),
                       (int) input.get(CV_CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter output(cardet::videoOutputName, fourcc, fps, videoSize, true);

    if (!output.isOpened()) {
        std::cout << "Cannot open video for writing" << std::endl;
        return 1;
    }

    // Frame to be read at each iteration
    cv::Mat frame;

    // Haar Cascades Classifier
    cv::CascadeClassifier classifier(cardet::trainedData);

    while (true) {
        input >> frame;

        if (frame.empty()) {
            break;
        }

        // List of objects that were detected by classifier
        std::vector <cv::Rect> detectedObjects;

        // Detection
        classifier.detectMultiScale(
                frame,
                detectedObjects, // Result of detection
                1.05,            // Scale at each iteration
                1,               // Min neighbours
                0,               // not used
                cv::Size(0, 0),  // minimal size of the object
                frame.size()     // maximal size of the object
        );

        cv::Mat detected;
        frame.copyTo(detected);

        for (const auto& rect: detectedObjects) {
            cv::rectangle(detected, rect, CV_RGB(255, 0, 0), 2);
        }

        cv::imshow(cardet::videoWindowName, frame);
        cv::imshow(cardet::detectedWindowName, detected);
        cv::waitKey(20);

        output << detected;
    }

    // Release captures
    output.release();
    input.release();

    return 0;
}