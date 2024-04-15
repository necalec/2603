#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace std::chrono;

int main() {
    auto start = high_resolution_clock::now();

    VideoCapture cap("video.mp4");

    if (!cap.isOpened()) {
        cerr << "eror" << endl;
        return -1;
    }

    CascadeClassifier faceCascade, eyeCascade, smileCascade;
    if (!faceCascade.load("C:/Users/necal/Downloads/haarcascades/haarcascade_frontalface_alt.xml") ||
        !eyeCascade.load("C:/Users/necal/Downloads/haarcascades/haarcascade_eye_tree_eyeglasses.xml") ||
        !smileCascade.load("C:/Users/necal/Downloads/haarcascades/haarcascade_smile.xml")) {
        cerr << "egor" << endl;
        return -1;
    }

    Mat frame;
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter video("out.mp4", VideoWriter::fourcc('H', '2', '6', '4'), 30, Size(frame_width / 2, frame_height / 2));

    if (!video.isOpened()) {
        cerr << "eror" << endl;
        return -1;
    }

#pragma omp parallel
    {
        while (true) {
#pragma omp critical
            {
                cap >> frame;
            }

            if (frame.empty())
                break;

            resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));

            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            vector<Rect> faces;
            faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

#pragma omp parallel for
            for (int i = 0; i < faces.size(); ++i) {
                rectangle(frame, faces[i], Scalar(255, 0, 0), 2);

                Mat faceROI = gray(faces[i]);

                vector<Rect> eyes;
                eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 3);

                int j = 0;
                for (const auto& eye : eyes) {
                    if (j < 2) {
                        Point eyeCenter(faces[i].x + eye.x + eye.width / 2, faces[i].y + eye.y + eye.height / 2);
                        int radius = cvRound((eye.width + eye.height) * 0.25);
                        circle(frame, eyeCenter, radius, Scalar(0, 0, 255), 2);
                        j++;
                    }
                }

                vector<Rect> smiles;
                smileCascade.detectMultiScale(faceROI, smiles, 1.2, 30, 0, Size(20, 20));

                j = 0;
                for (const auto& smile : smiles) {
                    if (j < 1) {
                        Rect smile_abs(smile.x + faces[i].x, smile.y + faces[i].y, smile.width, smile.height);
                        rectangle(frame, smile_abs, Scalar(0, 255, 0), 2);
                        j++;
                    }
                }
            }

#pragma omp critical
            {
                video.write(frame);
            }
        }
    }

    cap.release();
    video.release();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    cout << "time: " << duration.count() << endl;

    return 0;
}

