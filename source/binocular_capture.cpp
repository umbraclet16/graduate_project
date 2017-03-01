/// binocular_capture.cpp
/// Program for displaying, taking pictures and recording with binocular camera
/// umbraclet, 2017-02-28

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "omp.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>     // access()
#include <sys/stat.h>   // mkdir()
#include <sys/types.h>
#include <dirent.h>     // readdir()

using namespace cv;
using namespace std;

#define CAM_NUM 2   // binocular

int camera_offset = 0;          // In case that the computer has built-in cameras
bool take_pics = false;
bool record = false;
int cnt_pics = 0;
int cnt_videos = 0;
const char* camera_name[2] = {"left", "right"};
char dir_name[100] = "";        // default directory name for pictures and videos
bool dir_created = false;
bool video_file_created = false;

static void argParsing(int argc, const char* argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-i"))         // ID of the left camera
        {
            //------------------------------
            i++;
            // CAUTIOUS! argv[i] is a string, *argv[i] is a character!!!
            if (*argv[i] >= '0' && *argv[i] <= '9')
                camera_offset = *argv[i] - '0';
            else
                cout << "Invalid camera ID!" << endl;
            //----------OR------------------
            /*
             *if (sscanf(argv[++i], "%d", &camera_offset) != 1 || camera_offset < 0 || camera_offset > 9)
             *    cout << "Invalid camera ID!" << endl;
             */
            //------------------------------

        }
        else if (!strcmp(argv[i], "-p"))    // directory name
        {
            i++;
            strcpy(dir_name, argv[i]);
        }
    }
}

// TODO: display usage on the screen?
static void usage(const char* argv[])
{
    cout << "--------------------------------------------------" << endl;
    cout << "Optional arguments:" << endl;
    cout << "       -i: ID of left camera, default = 0;" << endl;
    cout << "       -p: name of the directory to store the pics and videos." << endl;
    cout << " e.g. " << argv[0] << " -i 1 -p folder" << endl;       // argv[0] already includes "./"!
    cout << "--------------------------------------------------" << endl;
    cout << "Usage:" << endl;
    cout << "       hit Enter to take pictures;" << endl;
    cout << "       hit 'r' to start/stop recording videos;" << endl;
    cout << "       hit 'q' or ESC to quit." << endl;
    cout << "--------------------------------------------------" << endl;
}

static void currTimeToStr(char* str)    // "yyyymmdd_hhMM"
{
        // Get date
        time_t tm;
        time(&tm);
        struct tm *t_local = localtime(&tm);
        // Custom time format. %F:2017-xx-xx, %T:hh:mm:ss, %H:hh, %M:MM
        strftime(str, 100, "%Y%m%d_%H%M", t_local);     // 2017mmdd_hhMM
        return;
}

// return value: 0: Success; -1: Fail
static int mkDirRecursive(const char *sPathName)
{
    char dirName[256];
    strcpy(dirName, sPathName);

    int len = strlen(dirName);
    if(dirName[len-1] != '/')
        strcat(dirName, "/");
    len = strlen(dirName);

    int cnt_mkdir;      // count the times that mkdir() has been called

    // Create directory recursively. mkdir() cannot create multi-level path at once!
    for (int i = 1; i < len; i++)
    {
        if (dirName[i] == '/')
        {
            dirName[i] = 0;
            if (access(dirName, F_OK) != 0)     // directory not exists
            {
                cnt_mkdir++;
                // mkdir(dirName, S_IRWXU | S_IRWXG | S_IRWXO)
                int ret = mkdir(dirName, 0755);
                if (ret == -1) // fails to creat directory
                {
                    perror("mkdir error");
                    return -1;
                }
            }
            dirName[i] = '/';
        }
    }

    // If the directory already exists, ask if overwrite
    if (cnt_mkdir == 0)     // the entire path already exists
    {
            cout << "Directory " << dirName << " already exists! Overwrite?(y/n)" << endl;
            char c;
            cin >> c;
            if (c == 'y' || c == '\n')
                return 0;
            else
                return -1;
    }
    else
    {
        cout << "Created path: " << dirName << endl;
    }

    return 0;
}

static bool dirEmpty(const char* dir_name)
{
    // Use readdir() to count files in the directory. If there are only . and .., the directory is empty
    DIR *dirp;
    int n = 0;

    dirp = opendir(dir_name);
    while (dirp)
    {
        if (readdir(dirp) != NULL)
            ++n;
        else
            break;
    }

    if (n == 2)         // . and .. always exist
        return true;
    else
        return false;
}

static void rmEmptyDir(const char* dir_name)
{
    if (dirEmpty(dir_name))
    {
        cout << "Dir " << dir_name << " is empty, removed." << endl;
        rmdir(dir_name);
    }
}

int main(int argc, const char* argv[])
{
    // Parse input arguments
    argParsing(argc, argv);

    // Display usage information
    usage(argv);

    // Use argument as directory name. If argument is empty, use "data/date_and_time" as default
    if (!strlen(dir_name))
    {
            strcpy(dir_name, "data/");

            char str[30];
            currTimeToStr(str);
            strcat(dir_name, str);
    }

	VideoCapture cap[CAM_NUM];
    VideoWriter  put[CAM_NUM];
	int i;

    // Open the cameras
	#pragma omp parallel for
	for (i = 0; i < CAM_NUM; i++)
	{
		cap[i].open(i+camera_offset);
        // Check if the file was opened properly
        if (!cap[i].isOpened())
        {
            cout << "Capture could not be opened successfully, exiting." << endl;
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
        // Make directory for storage if taking pictures or recording
        if (!dir_created && (take_pics || record))
        {
            dir_created = true;
            int ret = mkDirRecursive(dir_name);
            if (ret) return -1;     // Failed to make directory, exit
        }

        //----------------------------------------------------------------------
        // Use parallel loops. (private eliminates data competition)
		#pragma omp parallel for private(img, img_scaled, coord_left, coord_top)
		for (i = 0; i < CAM_NUM; i++)
		{
			cap[i+camera_offset] >> img;
			if (img.empty())
			{
				runflag = false;    // Either input channel finishes will stop both channels
				break;
			}

            //-------------------- Take pictures --------------------
            if (take_pics)
            {
                if (i == 0) cnt_pics++;         // update picture index before left camera takes a picture
                if (i == 1) take_pics = false;  // reset flag when right camera takes a picture
                char file_path[50];
                sprintf(file_path, "%s/%s%d.jpg", dir_name, camera_name[i], cnt_pics);
                imwrite(file_path, img);
                cout << "A picture has been written to " << file_path << "!" << endl;
            }

            //-------------------- Record videos --------------------
            if (record)
            {
                // Check if the video files have been created
                if (!video_file_created)
                {
                    char file_path[50];
                    sprintf(file_path, "%s/v_%s%d.mpg", dir_name, camera_name[i], cnt_videos);
                    put[i].open(file_path, CV_FOURCC('M', 'P', 'E', 'G'), 30, Size(origin_width, origin_height));
                    if (!put[i].isOpened())
                    {
                        cout << "File could not be opened for writing. Check permission. Exiting." << endl;
                        rmEmptyDir(dir_name);
                        return -1;
                    }
                    if (i == 1) video_file_created = true;  // reset flag after video file for right camera is created
                    cout << "Start recording, video file is " << file_path << endl;
                }

                // Write frame to file
                put[i] << img;
            }
            else
            {
                if (video_file_created) // A video has been finished
                {
                    cout << "Stop recording." << endl;
                    video_file_created = false;
                }
            }
            //--------------------------------------------------

            // Scale the input
            resize(img, img_scaled, Size(width, height));

            // Solve for the coordinates of top left corner of the child window
			coord_left = i % 2 * width;
			coord_top = 0;
            // Copy the scaled image into the child window
			img_scaled.copyTo(imageShow(Rect(coord_left, coord_top, width, height)));
		}
        //----------------------------------------------------------------------
        // Output text
        string msg_pics = format("Pictures taken: %d", cnt_pics);
        string msg_videos = "Recording";
        int baseLine1 = 0;
        int baseLine2 = 0;
        Size textSize1 = getTextSize(msg_pics, 1, 1, 1, &baseLine1);
        Size textSize2 = getTextSize(msg_videos, 1, 1, 1, &baseLine2);
        Point textOrigin1(imageShow.cols - textSize1.width - textSize2.width - 30, imageShow.rows - 2*baseLine1 - 10);
        Point textOrigin2(imageShow.cols - textSize2.width - 20, imageShow.rows - 2*baseLine2 - 10);
        putText(imageShow, msg_pics, textOrigin1, 1, 1, Scalar(0, 255, 0)); // green
        if (record)
            putText(imageShow, msg_videos, textOrigin2, 1, 1, Scalar(0, 0, 250));   // red

        //----------------------------------------------------------------------
        // Show image and check for input commands
        imshow("Binocular camera", imageShow);

		char key = waitKey(33);    // 30 fps
        switch (key)
        {
            case '\n':
                take_pics = true;
                break;

            case 'r':
                record = !record;
                if(record)
                    cnt_videos++;
                break;

            case 'q':
            case 27:    // ESC
                runflag = false;
                break;

            default:
                break;
        }
	}

	return 0;
}
