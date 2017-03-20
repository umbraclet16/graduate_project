/// Create an image of chessboard for camera calibration.
/// Print the generated image on a A4 paper.
/// (Actually displaying the image on the screen may be better)

#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

// A4 paper size: 210mm * 297mm. If the printer resolution is 300 dpi, then pixel should be 2479*3508
const int a4_width = 210 * 3;
const int a4_height = 297 * 3;

const int board_width = 6;      // the number of inner corners per row
const int board_height = 9;     // the number of inner corners per col

const int square_size = a4_width / (board_width + 1); // the number of square per row/col is one more than the number of corners

int main()
{
    // Create a black board
    Mat a4(a4_height, a4_width, CV_8U, Scalar(0));  // must assign with Scalar!

    // Change half squares to white
    for(int i = 0; i < a4_height; i++) {
        uchar* data = a4.ptr<uchar>(i);
        for(int j = 0; j < a4_width; j++) {
            if((i/square_size + j/square_size) % 2) {
                data[j] = 255;
            }
        }
    }

    // Display and write to file
    namedWindow("chessboard", WINDOW_AUTOSIZE);
    imshow("chessboard", a4);

    imwrite("chessboard.jpg", a4);
    cout << "The generated image has been written to chessboard.jpg, "
        << "board_width =" << board_width << ", board_height = " << board_height << endl;

    waitKey(0);

    return 0;
}

