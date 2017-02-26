/// Play multichannel videos using multithread technique(OpenMP).
///
/// Reference:
/// http://tuicool.com/articles/26fei2 (http://blog.csdn.net/dengtaocs/article/details/38065955)
/// http://blog.csdn.net/jiyangsb/article/details/48866593
/// OpenMP: http://www.openmp.org/

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"
#include <iostream>

using namespace cv;
using namespace std;

#define CAM_NUM 2   // binocular

int main(int argc, char* argv[])
{
    // In case that the computer has built-in cameras, we allow the ID of left camera as optional input
    int camera_offset = 0;
    if(argc == 2)
        if(*argv[1] >= '0' && *argv[1] <= '9')
            camera_offset = *argv[1] - '0';

	VideoCapture cap[CAM_NUM];
	int i;

	#pragma omp parallel for //private(i)    // loop index is private by default
	for (i = camera_offset; i < CAM_NUM; i++)
	{
		cap[i].open(i);
        // Check if the file was opened properly
        if(!cap[i].isOpened())
        {
            cout << "Capture could not be opened successfully" << endl;
            return -1;
        }
	}

    // Origin size of camera input
	int origin_width = cap[0].get(CV_CAP_PROP_FRAME_WIDTH);
	int origin_height = cap[0].get(CV_CAP_PROP_FRAME_HEIGHT);
    // If we put two video in a row directly, the window will be too wide for the screen.
    // So scale them by 4/5.
    int width = origin_width * 4 / 5;
    int height = origin_height * 4 / 5;
    // Put two videos in a row in the same window
	int display_width = width * 2;
	int display_height = height;

	Mat imageShow(display_height, display_width, CV_8UC3);  // used for display
	Mat img;                    // store input frames
	Mat img_scaled;             // used for scaling the inputs
    // coordinates of top left corner of each camera input at different place of the display window
	int coord_left, coord_top;  
    bool runflag = true;

    namedWindow("Binocular camera", WINDOW_AUTOSIZE);

	while (runflag)
	{
#define PARALLEL_METHOD 1
        //----------------------------------------------------------------------
#if PARALLEL_METHOD == 1
        // 1.We can use parallel loops
		#pragma omp parallel for private(img, img_scaled, coord_left, coord_top)    // private eliminates data competition
		for (i = camera_offset; i < CAM_NUM; i++)
		{
			cap[i] >> img;
			if (img.empty())
			{
				runflag = false;    // Either input channel finishes will stop both channels
				break;
			}

            // Scale the input
            resize(img, img_scaled, Size(width, height));

            // Solve for the coordinates of top left corner of the child window
			coord_left = i % 2 * width;
			coord_top = 0;
            // Copy the scaled image into the child window
			img_scaled.copyTo(imageShow(Rect(coord_left, coord_top, width, height)));  // This is the KEY!
		}
        //----------------------------------------------------------------------
#elif PARALLEL_METHOD == 2
        // 2.or we can also use sections worksharing construct(usually used for acyclic structure)
        #pragma omp parallel sections private(img img_scaled coord_left coord_top)
        {
            #pragma omp section
            {
                cap[0] >> img;
                if (img.empty())
                {
                    runflag = false;    // Either input channel finishes will stop both channels
                    break;
                }

                // Scale the input
                resize(img, img_scaled, Size(width, height));

                // Solve for the coordinates of top left corner of the child window
                coord_left = 0;
                coord_top = 0;
                // Copy the scaled image into the child window
                img_scaled.copyTo(imageShow(Rect(coord_left, coord_top, width, height)));  // This is the KEY!

            }
            #pragma omp section
            {
                cap[1] >> img;
                if (img.empty())
                {
                    runflag = false;    // Either input channel finishes will stop both channels
                    break;
                }

                // Scale the input
                resize(img, img_scaled, Size(width, height));

                // Solve for the coordinates of top left corner of the child window
                coord_left = width;
                coord_top = 0;
                // Copy the scaled image into the child window
                img_scaled.copyTo(imageShow(Rect(coord_left, coord_top, width, height)));  // This is the KEY!
            }
        }
        //----------------------------------------------------------------------
#endif

        imshow("Binocular camera", imageShow);

		char key = waitKey(33);    // 30 fps
        if(key == 'q' || key == 27)
            break;
	}
	return 0;
}
