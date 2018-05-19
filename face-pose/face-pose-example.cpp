//
// face-pose-example.cpp
//
// Face detection program based on Dlib's original face pose example and on
// Satya Mallick "Speeding up Dlib’s Facial Landmark Detector" blog post.
//
// - Dlib's face pose file comments (shortened):
//
//   This example program shows how to find frontal human faces in an image and
//   estimate their pose. The pose takes the form of 68 landmarks. These are
//   points on the face such as the corners of the mouth, along the eyebrows, on
//   the eyes, and so forth.
//
//   https://github.com/davisking/dlib/blob/master/examples/webcam_face_pose_ex.cpp
//
// - Speeding up Dlib’s Facial Landmark Detector
//
//   https://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
//


#include <time.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;


const unsigned int image_detection_ratio = 2;
const unsigned int image_downsample_ratio = 2;


static dlib::rectangle openCVRectToDlib(cv::Rect r) {

  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);

}


static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r) {

  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));

}


void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false) {

    std::vector <cv::Point> points;

    for (int i = start; i <= end; ++i) {

        points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));

    }

    cv::polylines(img, points, isClosed, cv::Scalar(0, 255, 0), 1, 16);

}


int main() {

    try {

        cv::VideoCapture video_feed(0);

        if (video_feed.isOpened() == false) {

            printf("Unable to connect to the camera.");

            return 0;

        }

        // Load frontal face detector and the face landmarks pose model.

        shape_predictor pose_model;
        frontal_face_detector detector = get_frontal_face_detector();

        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Auxiliaries to calculate the output fps. We measure the time delta
        // every iteration and once it is above 1 second, the fps is calculated.

        time_t start;
        time(&start);

        // Grab and process frames until the esc key is pressed.

        while(cv::waitKey(10) != 27) {

            // Grab a frame and downsize it in order to speed up face detection.

            static double down_ratio = 1.0/image_downsample_ratio;
            cv::Mat video_sample;
            cv::Mat video_sample_small;

            if (video_feed.read(video_sample) == false) { break; }

            cv::resize(video_sample, video_sample_small, cv::Size(), down_ratio, down_ratio);

            // Turn OpenCV's Mat into something dlib can deal with. Note that this just
            // wraps the Mat object, it doesn't copy anything. So cimg is only valid as
            // long as temp is valid. Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers. This basically means you shouldn't modify temp
            // while using cimg.

            cv_image <bgr_pixel> d_video_sample_small(video_sample_small);

            // Detect faces every "image_detection_ratio" frames, this helps
            // speed up the program since face detection is the slowest step.

            static uint8_t iteration_count = 0;
            static std::vector<rectangle> faces;

            if (iteration_count++ % image_detection_ratio == 0) {

                faces = detector(d_video_sample_small);

            }

            // Find the pose of each face. Even tough the faces bounding boxes
            // might not be the same due to the skip frames the speed up, we
            // can still look for landmarks in the previous areas.

            std::vector<full_object_detection> shapes;

            for (int i = 0; i < faces.size(); i++) {

                shapes.push_back(pose_model(d_video_sample_small, faces[i]));

            }

            // Draw faces bounding boxes and their facial landmarks over the
            // downsized image.

            for (int i = 0; i < faces.size(); i++) {

                cv::Rect r = dlibRectangleToOpenCV(faces[i]);
                cv::Scalar c = cv::Scalar(0, 0, 255);

                cv::rectangle(video_sample_small, r, c);

                // Landmarks in order: Jaw line, left eyebrow, right eyebrow,
                // nose bridge, lower nose, left eye, right eye, outer lip and
                // inner lip.

                draw_polyline(video_sample_small, shapes[i], 0, 16);
                draw_polyline(video_sample_small, shapes[i], 17, 21);
                draw_polyline(video_sample_small, shapes[i], 22, 26);
                draw_polyline(video_sample_small, shapes[i], 27, 30);
                draw_polyline(video_sample_small, shapes[i], 30, 35, true);
                draw_polyline(video_sample_small, shapes[i], 36, 41, true);
                draw_polyline(video_sample_small, shapes[i], 42, 47, true);
                draw_polyline(video_sample_small, shapes[i], 48, 59, true);
                draw_polyline(video_sample_small, shapes[i], 60, 67, true);

            }

            // Calculate frames per second.

            static float fps = 0;
            static unsigned int fps_count = 0;

            time_t end;
            time(&end);

            if (difftime(end, start) >= 1) {

                fps = fps_count/difftime(end, start);
                fps_count = 0;
                start = end;

            }else{

                fps_count++;

            }

            // Draw the fps text over the downsized image.

            std:stringstream fps_text_stream;
            fps_text_stream << fps << " fps";
            std::string fps_text = fps_text_stream.str();

            cv::Point fps_text_point = cv::Point(50, 50);
            cv::Scalar fps_text_color = cv::Scalar(255, 255, 255);
            int fps_text_font = cv::FONT_HERSHEY_COMPLEX_SMALL;

            cv::putText(video_sample_small, fps_text, fps_text_point, fps_text_font, 1.0, fps_text_color, 1);

            // Show detected faces and their facial landmarks in a opencv window,
            // using opencv proved to be faster than dlib's implementation.

            cv::namedWindow("Facial Landmarks", cv::WINDOW_AUTOSIZE);
            cv::imshow("Facial Landmarks", video_sample_small);

        }

    } catch(serialization_error& e) {

        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;

    } catch(exception& e) {

        cout << e.what() << endl;

    }

}
