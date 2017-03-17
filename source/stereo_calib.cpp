/// stereo_calib.cpp
/// Stereo calibration and rectification(Boguet's method).
///
/// Ref:
///     opencv/samples/cpp/stereo_calib.cpp;
///     <Learning OpenCV> Ch12;
///     http://blog.csdn.net/zc850463390zc/article/details/48975263

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;

#define ESC_KEY 27
//--------------------------------------------------
// Parameters. Edit according to your condition(camera, chessboard, assumptions)
//--------------------------------------------------
int boardWidth = 6;       // number of corners per row
int boardHeight = 5;      // number of corners per column
float squareSize = 30;    // the size of a square in the chessboard(in mm)
Size boardSize(boardWidth, boardHeight);
bool showRectified = true;
int delay_ms = 300;       // time delay between displaying two images

bool useIndividualCalibResult = true;  // use individual calib result
// individual calib result filenames
string calibResultLFn("calib_result_l.xml");
string calibResultRFn("calib_result_r.xml");
//--------------------------------------------------
// Global Variables
//--------------------------------------------------
string imageListFn;             // image list filename
string outputFn = "stereo_params.xml";
vector<string> imageList;       // list of images
vector<string> goodImageList;   // list of images in which corners are detected
//--------------------------------------------------
// Function Declarations
//--------------------------------------------------
static void usage();
static void argParsing(int argc, char** argv, string& imageListFn);
static bool readStringList(const string& filename, vector<string>& l);
static void calcBoardCornerPositions(const Size& boardSize, const float squareSize,
        vector<vector<Point3f> >& corners);
static void stereoCalib(const vector<string>& imageList, const Size& boardSize,
        bool showRectified=true);
static int findCorners(const vector<string>& imageList,
        vector<vector<Point2f> > imagePoints[],
        Size& imageSize, int& nimages);
static void computeReprojectionError(const vector<vector<Point2f> > imagePoints[],
        const int nimages, const Mat cameraMatrix[], const Mat distCoeffs[], const Mat& F);
static void mergeImages(Mat& canvas, const Size imageSize,
        const Mat& imgL, const Mat& imgR);
static void saveStereoCalibResult(const string& outputFn, const Mat cameraMatrix[],
        const Mat distCoeffs[], const Mat& R, const Mat& T, const Mat& E, const Mat& F,
        const double rms);
static void saveRectificationResult(const string& outputFn, Mat& R1, Mat& R2,
        Mat& P1, Mat& P2, Mat& Q);
static void rectify(Mat cameraMatrix[], Mat distCoeffs[], Size& imageSize,
        const Mat& R, const Mat& T, const string& outputFn);
//--------------------------------------------------

int main(int argc, char** argv)
{
    argParsing(argc, argv, imageListFn);

    // Read image list. Exit if fails
    bool ok = readStringList(imageListFn, imageList);
    if (!ok || imageList.empty())
    {
        cout << "Cannot open " << imageListFn << " or the string is empty. Exiting." << endl;
        return -1;
    }

    stereoCalib(imageList, boardSize, showRectified);

    return 0;
}

void usage()
{
    cout << "Usage:" << endl;
    cout << "\t./stereo_calib -w board_witdh -h board_height <image list XML/YML file>" << endl;
    cout << "\tdefault: ./stereo_calib -w 6 -h 5 stereo_calib.xml" << endl;
}

void argParsing(int argc, char** argv, string& imageListFn)
{
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "-w")
        {
            if (sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
            {
                cout << "Invalid board width!" << endl;
                return usage();
            }
        }
        else if (string(argv[++i]) == "-h")
        {
            if (sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
            {
                cout << "Invalid board height!" << endl;
                return usage();
            }
        }
        else if (string(argv[i]) == "-nr")
            showRectified = false;
        else if (argv[i][0] == '-')
        {
            cout << "Invalid option " << argv[i] << endl;
            return usage();
        }
        else
        {
            imageListFn = argv[i];
        }
    }

    if (imageListFn.empty())
        imageListFn = "stereo_calib.xml";
}

bool readStringList(const string& filename, vector<string>& l)
{
    l.clear();
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open file " << filename << endl;
        return false;
    }
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
    {
        cout << "File content is not a sequence! FAIL" << endl;
    }
    FileNodeIterator it = n.begin(), it_end = n.end();
    for ( ; it != it_end; it++)
        l.push_back((string)*it);
    return true;
}

// calculate the coordinates of board corners in world coord system
void calcBoardCornerPositions(const Size& boardSize, const float squareSize, vector<vector<Point3f> >& corners)
{
    for (int n = 0; n < corners.size(); n++)
    {
        for (int i = 0; i < boardSize.height; i++)
            for (int j = 0; j < boardSize.width; j++)
                corners[n].push_back(Point3f(i*squareSize, j*squareSize, 0));
    }
}

void stereoCalib(const vector<string>& imageList, const Size& boardSize, bool showRectified)
{
    vector<vector<Point2f> > imagePoints[2];   // set of corners on each images in image coordinate
    vector<vector<Point3f> > objectPoints;     // set of corners on each images in world coordinate
    Size imageSize;

    //-------------------- 1.collect corners in image coord --------------------
    int nimages;
    int ret = findCorners(imageList, imagePoints, imageSize, nimages);
    if (ret) return;

    //-------------------- 2.calc corners coords in world coord --------------------
    objectPoints.resize(nimages);
    calcBoardCornerPositions(boardSize, squareSize, objectPoints);

    //-------------------- 3.calibrate --------------------
    cout << "Running stereo calibration..." << endl;

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    int flag = 0;
    flag = CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_ZERO_TANGENT_DIST;

    if (useIndividualCalibResult)
    {
        flag = CV_CALIB_FIX_INTRINSIC;  // only R, T, E, and F are estimated

        FileStorage fs(calibResultLFn, FileStorage::READ);
        fs["cameraMatrix"] >> cameraMatrix[0];
        fs["distCoeffs"]   >> distCoeffs[0];
        fs.release();

        fs.open(calibResultRFn, FileStorage::READ);
        fs["cameraMatrix"] >> cameraMatrix[1];
        fs["distCoeffs"]   >> distCoeffs[1];
        fs.release();
    }


    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
            cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, E, F,
            TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1e-6),
            flag);

    cout << "Finished, with RMS error = " << rms << endl;

    // check calibration quality
    computeReprojectionError(imagePoints, nimages, cameraMatrix, distCoeffs, F);

    // save intrinsic params
    cout << "Saving stereo calibration result to " << outputFn << "...";
    saveStereoCalibResult(outputFn, cameraMatrix, distCoeffs, R, T, E, F, rms);
    cout << " Done." << endl;

    //-------------------- 4.rectify, display, and save --------------------
    destroyAllWindows();
    if (showRectified)
        rectify(cameraMatrix, distCoeffs, imageSize, R, T, outputFn);
}

int findCorners(const vector<string>& imageList, vector<vector<Point2f> > imagePoints[],
                Size& imageSize, int& nimages)
{
    if (imageList.size() % 2 != 0)
    {
        cout << "Error: the image list contains odd number of elements!" << endl;
        return -1;
    }

    bool displayCorners = true;
    nimages = (int)imageList.size()/2;
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);

    int npairs = 0;     // count image pairs that chessboard pattern is found in both images
    for (int i = 0; i < nimages; i++)
    {
        int k;
        Mat imgL, imgR;     // to display image pairs in the same window
        for (k = 0; k < 2; k++)
        {
            const string& filename = imageList[i*2+k];  // 'left01.jpg','right01.jpg','left02.jpg',...
            Mat img = imread(filename, CV_LOAD_IMAGE_COLOR);
            if (img.empty())
                break;

            if (k == 0) imgL = img;
            if (k == 1) imgR = img;

            if (imageSize == Size())    // imageSize not assigned?
                imageSize = img.size();
            else if (img.size() != imageSize)
            {
                cout << "The image " << filename
                     << " has different size from the first image. Skipping the pair." << endl;
                break;
            }

            // This saves the effort to call vector::push_back().
            // (If the 2nd image is not good, npairs will not increase, imagePoints[k][npairs] will be assigned again)
            vector<Point2f>& corners = imagePoints[k][npairs];
            bool found = findChessboardCorners(img, boardSize, corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
            if (found)
            {
                // improve the found corners' coordinate accuracy
                Mat imageGray;
                cvtColor(img, imageGray, CV_BGR2GRAY);
                cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1),
                        TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

                // draw the corners on the image
                if (displayCorners)
                {
                    drawChessboardCorners(img, boardSize, Mat(corners), found);
                    if (k == 0) imgL = img;
                    if (k == 1) imgR = img;
                }

                //cout << "Detected corners in " << filename << endl;
            }
            else
            {
                cout << "Failed to detect corners in " << filename << endl;
                break;
            }

        }

        // successfully detected corners in both images
        if (k == 2)
        {
            goodImageList.push_back(imageList[i*2]);
            goodImageList.push_back(imageList[i*2+1]);
            npairs++;
        }

        // display two images in the same window
        Mat canvas;
        mergeImages(canvas, imageSize, imgL, imgR);
        imshow("searching for corners...", canvas);

        char key = waitKey(delay_ms);
        // start calibration immediately if 'q' or ESC is hitted
        if (key == ESC_KEY || key == 'q' || key == 'Q')
            break;
    }

    cout << npairs << " pairs have been successfully detected." << endl;
    nimages = npairs;

    if (nimages < 2)
    {
        cout << "Error: too little pairs to run the calibration. Exiting." << endl;
        return -1;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);

    return 0;
}

// put two images in a row so they can be displayed in a window
void mergeImages(Mat& canvas, const Size imageSize, const Mat& imgL, const Mat& imgR)
{
    int w, h;
    // scale factor to set the larger of width/height to 600
    double sf = 600./MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h, w*2, CV_8UC3); // put two images in a row
    Mat canvasL = canvas(Rect(0, 0, w, h));
    Mat canvasR = canvas(Rect(w, 0, w, h));
    // put two images in the window
    resize(imgL, canvasL, canvasL.size(), 0, 0, CV_INTER_LINEAR);
    resize(imgR, canvasR, canvasR.size(), 0, 0, CV_INTER_LINEAR);
    // or:
    // imgL.copyTo(canvasL);
    // imgR.copyTo(canvasR);
}

void computeReprojectionError(const vector<vector<Point2f> > imagePoints[],
                              const int nimages, const Mat cameraMatrix[],
                              const Mat distCoeffs[], const Mat& F)
{
    // because the output fundamental matrix implicitly includes all the output information,
    // we can check the quality of calibration using the epipolar geometry constraint:
    // m2^T*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for (int i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for(int k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " << err/npoints << endl;
}

void saveStereoCalibResult(const string& outputFn, const Mat cameraMatrix[],
                           const Mat distCoeffs[], const Mat& R, const Mat& T,
                           const Mat& E, const Mat& F, const double rms)
{
    FileStorage fs(outputFn, CV_STORAGE_WRITE);
    if (fs.isOpened())
    {
        char buf[1024];
        time_t tm;
        time(&tm);
        struct tm *t2 = localtime(&tm);
        strftime(buf, sizeof(buf)-1, "%c", t2);

        fs << "calibration_Time" << buf;

        cvWriteComment(*fs, "Intrinsic params:\n", 0);
        fs << "cameraMatrix1" << cameraMatrix[0] << "distCoeffs1" << distCoeffs[0]
           << "cameraMatrix2" << cameraMatrix[1] << "distCoeffs2" << distCoeffs[1];
        cvWriteComment(*fs, "Extrinsic params:\n", 0);
        fs << "R" << R << "T" << T << "E" << E << "F" << F;
        fs << "RMS" << rms;
        fs.release();
    }
    else
        cout << "Failed to save stereo calibration result to file." << endl;
}

void saveRectificationResult(const string& outputFn, Mat& R1, Mat& R2, Mat& P1, Mat& P2 , Mat& Q)
{
    FileStorage fs(outputFn, CV_STORAGE_APPEND);
    if (fs.isOpened())
    {
        cvWriteComment(*fs, "\nRectification params:\n", 0);
        fs << "R1" << R1 << "R2" << R2
           << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Failed to save rectification result to file." << endl;
}

// rectify, display, and save extrinsic params
void rectify(Mat cameraMatrix[], Mat distCoeffs[], Size& imageSize,
             const Mat& R, const Mat& T, const string& outputFn)
{
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    // If alpha=0, the ROIs cover the whole images.
    // Otherwise, they are likely to be smaller.
    double alpha = 1;
    stereoRectify(cameraMatrix[0], distCoeffs[0],
            cameraMatrix[1], distCoeffs[1],
            imageSize, R, T, R1, R2, P1, P2, Q,
            CALIB_ZERO_DISPARITY, alpha, imageSize, &validRoi[0], &validRoi[1]);

    cout << "Saving rectification result to " << outputFn << "...";
    saveRectificationResult(outputFn, R1, R2, P1, P2, Q);
    cout << " Done." << endl;

    // compute and display rectification
    Mat map[2][2];
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1,
                            imageSize, CV_16SC2, map[0][0], map[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2,
                            imageSize, CV_16SC2, map[1][0], map[1][1]);

    for (int i = 0; i < goodImageList.size(); i++)
    {
        int k;
        Mat imgL, imgR;
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i*2+k], CV_LOAD_IMAGE_COLOR);
            Mat imgRectified;
            remap(img, imgRectified, map[k][0], map[k][1], CV_INTER_LINEAR);

            // draw rectangle if alpha != 0(there are black areas after rectification)
            if (alpha != 0)
                rectangle(imgRectified, validRoi[k], Scalar(0, 0, 255), 3, 8);

            if (k == 0) imgL = imgRectified;
            if (k == 1) imgR = imgRectified;
        }

        Mat canvas;
        mergeImages(canvas, imageSize, imgL, imgR);
        // draw horizontal lines
        for (int j = 0; j < canvas.rows; j += 16)
            line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

        imshow("rectified", canvas);

        char c = (char)waitKey();
        if (c == ESC_KEY || c == 'q' || c == 'Q')
            break;
    }
}
