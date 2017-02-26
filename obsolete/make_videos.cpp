/// Make video and save to file for camera calibration.
/// Videos are displayed in separated windows.
/// The program is single threaded, so videos are not synchronous.

#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int  camera_ID[2] = {-1, -1};      // support monocular and binocular camera
int  camera_number;
bool save_to_file = false;

int makeSingleVideo(int camera_ID, bool save_to_file);
int makeBinocularVideo(int camera_ID[], bool save_to_file);

int main(int argc, const char* argv[])  // or: char** argv, char argv[][]
{
    cout << "monocular(1) or binocular(2)?" <<endl;
    cin >> camera_number;
    cout << "Input camera ID(if binocular, only input the first)" << endl;
    cin >> camera_ID[0];
    if(camera_number == 2) camera_ID[1] = camera_ID[0] + 1;
    cout << "Save to file(y/n)?" << endl;
    char save;
    cin >> save;
    if(save == 'y' || save == 'Y')
        save_to_file = true;

    if(camera_number == 1)
        makeSingleVideo(camera_ID[0], save_to_file);
    else
        makeBinocularVideo(camera_ID, save_to_file);

    return 0;
}

int makeSingleVideo(int camera_ID, bool save_to_file)
{
    // Open the camera
    VideoCapture cap(camera_ID);
    // Check if the camera was opened properly
    if(!cap.isOpened())
    {
        cout << "Camera " << camera_ID << " could not be opened successfully" << endl;
        return -1;
    }

    // Get size of frames
    Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

    // Make a video writer object and initialize it at 30 FPS
    char file_name[10];  // "x.mpg\0"
    sprintf(file_name, "%d.mpg", camera_ID);
    VideoWriter put(file_name, CV_FOURCC('M', 'P', 'E', 'G'), 30, S);
    if(save_to_file && !put.isOpened())
    {
        cout << "File could not be created for writing. Check permission" << endl;
        return -1;
    }

    char window_name[20];
    sprintf(window_name, "Video: camera %d", camera_ID);
    namedWindow(window_name);

    // Play the video in a loop till keyboard input
    while(char(waitKey(1)) != 'q' && cap.isOpened())
    {
        Mat frame;
        cap >> frame;
        imshow(window_name, frame);
        if(save_to_file)
            put << frame;
    }

    return 0;
}


int makeBinocularVideo(int camera_ID[], bool save_to_file)
{
    const char* filename[2] = {"left.mpg", "right.mpg"};
    VideoCapture cap[2];
    char window_name[2][20];

    for(int i = 0; i < 2; i++)
    {
        // Open the camera
        cap[i].open(camera_ID[i]);
        // Check if the camera was opened properly
        if(!cap[i].isOpened())
        {
            cout << "Camera " << camera_ID[i] << " could not be opened successfully" << endl;
            return -1;
        }
    }

    // Get size of frames
    Size S = Size((int)cap[0].get(CV_CAP_PROP_FRAME_WIDTH), (int)cap[0].get(CV_CAP_PROP_FRAME_HEIGHT));

    VideoWriter put[2];
    // Make a video writer object and initialize it at 30 FPS
    if(save_to_file)
    {
        for(int i = 0; i < 2; i++)
        {
            put[i].open(filename[i], CV_FOURCC('M', 'P', 'E', 'G'), 30, S);
            if(!put[i].isOpened())
            {
                cout << "File " << filename[i] << " could not be created for writing. Check permission" << endl;
                return -1;
            }
        }

        for(int i = 0; i < 2; i++)
        {
            sprintf(window_name[i], "Video: camera %d", camera_ID[i]);
            namedWindow(window_name[i]);
        }
    }

    // Play the video in a loop till keyboard input
    while(char(waitKey(1)) != 'q' && cap[0].isOpened() && cap[1].isOpened())
    {
        Mat frame;
        for(int i = 0; i < 2; i++)
        {
            cap[i] >> frame;
            imshow(window_name[i], frame);
            if(save_to_file)
                put[i] << frame;
        }
    }

    return 0;
}
