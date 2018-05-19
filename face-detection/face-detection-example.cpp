//
// face-detection-example.cpp
//
// Face detection program based on Dlib's original face detection example.
//
// - Dlib's face detection file comments (shortened):
//
//   This example program shows how to find frontal human faces in an image.  In
//   particular, this program shows how you can take a list of images from the
//   command line and display each on the screen with red boxes overlaid on each
//   human face.
//
//   This face detector is made using the now classic Histogram of Oriented
//   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
//   and sliding window detection scheme.  This type of object detector is fairly
//   general and capable of detecting many types of semi-rigid objects in
//   addition to human faces. Therefore, if you are interested in making your
//   own object detectors then read the fhog_object_detector_ex.cpp example
//   program. It shows how to use the machine learning tools which were used to
//   create dlib's face detector.
//
//   https://github.com/davisking/dlib/blob/master/examples/face_detection_ex.cpp
//


#include <iostream>

#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace std;
using namespace dlib;


int main(int argc, char** argv) {

    if (argc == 1) {

        cout << "Provide image paths as command line arguments to this program." << endl;

        return 0;

    }

    try {

        image_window win;
        frontal_face_detector detector = get_frontal_face_detector();

        // Loop over all the images provided on the command line.

        for (int i = 1; i < argc; ++i) {

            cout << "Processing image: " << argv[i] << endl;

            pyramid_down<2> pyr;

            array2d <rgb_pixel> image;
            array2d <unsigned char> grayscale_image;

            load_image(image, argv[i]);
            assign_image(grayscale_image, image);

            // Make the image bigger by a factor of two.  This is useful since
            // the face detector looks for faces that are about 80 by 80 pixels
            // or larger.  Therefore, if you want to find faces that are smaller
            // than that then you need to upsample the image as we do here by
            // calling pyramid_up().  So this will allow it to detect faces that
            // are at least 40 by 40 pixels in size.  We could call pyramid_up()
            // again to find even smaller faces, but note that every time we
            // upsample the image we make the detector run slower since it must
            // process a larger image.

            pyramid_up(grayscale_image);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces it can find in the image.

            std::vector<rectangle> dets = detector(grayscale_image);

            cout << "Found " << dets.size() << " faces." << endl;

            // Draw rectangles in the original image around the faces found. The
            // coordinates of the bounding boxes will be doubled since we scaled
            // up the grayscale image, we need to adjust them before drawing.

            for (int j = 0; j < dets.size(); j++) {

                dets[j] = pyr.rect_down(dets[j]);

                draw_rectangle(image, dets[j], rgb_pixel(255, 0, 0), 2);

            }

            // Scale down the image to a fixed size, so, all of them are
            // presented in a window with the same size. We need to keep the
            // image proportion in order to keep it looking minimally good (TO-DO).

            array2d<rgb_pixel> sizeImg(480, 640);
            resize_image(image, sizeImg);
            assign_image(image, sizeImg);

            win.clear_overlay();
            win.set_image(image);

            cout << "Hit enter to process the next image...";
            cin.get();

        }

    } catch (exception& e) {

        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;

    }

}
