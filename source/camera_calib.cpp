/// camera_calib.cpp
/// A simplified calibration program.
/// Calibrate single camera with a series of chessboard photos.
///
/// Input: xml/yaml file containing image list, or input with keyboard;
/// Output: save calibration result to xml file.
///
/// Ref:
///     opencv/sample/cpp/calib3d/camera_calibration/camera_calibration.cpp;
///     http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html#cameracalibrationopencv
///     http://blog.csdn.net/zc850463390zc/article/details/48946855

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;

#define ESC_KEY 27

//--------------------------------------------------
// Parameters. Edit according to your condition(camera, chessboard, assumptions)
//--------------------------------------------------
const int boardWidth = 6;       // number of corners per row
const int boardHeight = 5;      // number of corners per column
const int frameNumber = 15;     // number of input images for calibration
const float squareSize = 30;    // the size of a square in the chessboard(in mm)
const int imageWidth = 640;
const int imageHeight = 480;
const Size boardSize(boardWidth, boardHeight);
const Size imageSize(imageWidth, imageHeight);
string outputFileName;
int delay_ms = 800;         // time delay between displaying two images
int flag = 0;

//--------------------------------------------------
// Global Variables
//--------------------------------------------------
Mat cameraMatrix;           // camera intrinsic matrix
Mat distCoeffs;             // distortion coefficients
vector<Point2f> cornerBuf;  // corners found by findChessboardCorners()
vector<vector<Point2f> > imagePoints;   // set of corners on each images in image coordinate
vector<vector<Point3f> > objectPoints;  // set of corners on each images in world coordinate
vector<string> imageList;               // list of image names
//--------------------------------------------------
// Function Declarations
//--------------------------------------------------
static void usage(void);
static bool readImageListFile(const string& filename, vector<string>& imageList);
static void createImageList(vector<string>& imageList);
static Mat getImage(const vector<string>& imageList, const int currentIndex);
static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners);
static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
                                        const vector<vector<Point2f> >& imagePoints,
                                        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                        const Mat& cameraMatrix, const Mat& distCoeffs,
                                        vector<float>& perViewErrors);
static bool runCalibration(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                           vector<vector<Point2f> > imagePoints, vector<vector<Point3f> > objectPoints,
                           vector<Mat>& rvecs, vector<Mat>& tvecs,
                           vector<float>& reprojErrs, double& totalAvgErr);
static void saveCameraParams(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                             const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                             const vector<float>& reprojErrs, double totalAvgErr);
static void displayUndistortedImage(const vector<string>& imageList, const Mat& cameraMatrix, const Mat& distCoeffs);
//--------------------------------------------------

int main(int argc, const char* argv[])
{

    usage();

    //-------------------- 0.parse arguments --------------------
    for (int i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "-i")    // !must convert to string! or use !strcmp()
            readImageListFile(argv[++i], imageList);
        if (string(argv[i]) == "-o")
            outputFileName = argv[++i];
    }
    // if have not read image list from file, create one from keyboard input
    if (imageList.size() == 0)
        createImageList(imageList);
    // if no output file name assigned, name by 'result_DATE.xml'
    if (outputFileName.empty())
    {
        char buf[1024];
        time_t tm;
        time(&tm);
        struct tm *t2 = localtime(&tm);
        strftime(buf, sizeof(buf)-1, "%m%d", t2);
        outputFileName = "calib_result_" + string(buf) + ".xml";
    }

    // comment out undesired flag
    // TODO: maybe make flag input arguments? -fp, -z, -fa
    flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
    flag |= CV_CALIB_ZERO_TANGENT_DIST;
    flag |= CV_CALIB_FIX_ASPECT_RATIO;
    //flag |= CV_CALIB_USE_INTRINSIC_GUESS;

    //-------------------- 1.collect corners in image coord --------------------
    int goodFrameCnt = 0, currentIndex = 0;
    namedWindow("Camera Calibration");
    while (goodFrameCnt < frameNumber)
    {
        Mat image = getImage(imageList, currentIndex);
        if (image.empty())
            break;

        // look for corners in the current image
        bool found = findChessboardCorners(image, boardSize, cornerBuf, 
                CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        if (found)
        {
            // improve the found corners' coordinate accuracy
            Mat imageGray;
            cvtColor(image, imageGray, CV_BGR2GRAY);
            cornerSubPix(imageGray, cornerBuf, Size(11, 11), Size(-1, -1),
                    TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));  // just follow reference manual
            imagePoints.push_back(cornerBuf);

            // draw the corners on the image
            drawChessboardCorners(image, boardSize, Mat(cornerBuf), found);

            goodFrameCnt++;
            cout << "Detected corners in " << imageList[currentIndex] << endl;
        }
        else
            cout << "Failed to detect corners in" << imageList[currentIndex] << endl;

        // output text
        string msg = format("%d/%d", (int)imagePoints.size(), frameNumber);
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(image.cols - 2*textSize.width - 10, image.rows - 2*baseLine - 10);
        putText(image, msg, textOrigin, 1, 1, Scalar(0, 255, 0));
        imshow("Camera Calibration", image);
        
        currentIndex++;
        char key = waitKey(delay_ms);
        // start calibration immediately if 'q' or ESC is hitted
        if (key == 'q' || key == ESC_KEY)
            break;
        // if any other key is hitted, speed up the procedure.
        // scan the next image right away.
        else if (key != -1)
            continue;
    }

    //-------------------- 2.calc corners coords in world coord--------------------
    // NOTICE: function output corresponds to corners in one image(vector<Point3f>)
    // but objectPoints is vector<vector<Point3f> >, so we need to replicate!
#if 1
    vector<Point3f> temp;
    calcBoardCornerPositions(boardSize, squareSize, temp);
    for ( int i = 0; i < goodFrameCnt; i++)
        objectPoints.push_back(temp);
#elif 0
    //--------------------------------------------------
    // the opencv sample use vector::resize() to duplicate the coords for all images
    // this has a side effect: objectPoints must contain at least one element,
    // otherwise it's illegal to access objectPoints[0].
    // then need to call vector::clear() at the beginning of calcBoardCornerPositions().
    //--------------------------------------------------
    objectPoints.push_back(vector<Point3f>(1)); // make objectPoints[0] legal
    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
    // function prototype: void Mat::resize(size_t sz, const Scalar& s)
    objectPoints.resize(imagePoints.size(), objectPoints[0]);
#endif

    //-------------------- 3.calibrate --------------------
    vector<Mat> rvecs, tvecs;   // rotation vectors and translation vectors
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imageSize, cameraMatrix, distCoeffs,
            imagePoints, objectPoints, rvecs, tvecs, reprojErrs, totalAvgErr);

    cout << (ok ? "Calibration succeeded" : "Calibration failed")
         << ". avg re-projection error = " << totalAvgErr << endl;

    //-------------------- 4.save calibration result --------------------
    if(ok)
        saveCameraParams(imageSize, cameraMatrix, distCoeffs,
                rvecs, tvecs, reprojErrs, totalAvgErr);

    //-------------------- 5.display undistorted images --------------------
    destroyWindow("Camera Calibration");
    displayUndistortedImage(imageList, cameraMatrix, distCoeffs);

    return 0;
}

void usage()
{
    cout << "Usage:" << endl
         << "\t-i: xml/yaml file containing image list;" << endl
         << "\t    (if omitted, program will prompt to input from keyboard)" << endl
         << "\t-o: output filename to save calibration result, default is 'result_TIME.xml'." << endl;
}

bool readImageListFile(const string& filename, vector<string>& imageList)
{
    imageList.clear();
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Failed to open the file " << filename << endl;
        return false;
    }
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
    {
        cout << "File content is not a sequence! FAIL" << endl;
        return false;
    }
    FileNodeIterator it = n.begin(), it_end = n.end();
    for ( ; it != it_end; it++)
        imageList.push_back((string)*it);
    return true;
}

void createImageList(vector<string>& imageList)
{
    string prefix;
    int imageNumber;

    cout << "Input the relative path and prefix of images, e.g. 'images/left':" << endl;
    cin >> prefix;
    cout << "Input the number of images:" << endl;
    cin >> imageNumber;

    for (int i = 1; i <= imageNumber; i++)
    {
        char idx[5];
        sprintf(idx, "%02d", i);
        string filename = prefix + (string)idx + ".jpg";
        imageList.push_back(filename);
    }
}

Mat getImage(const vector<string>& imageList, const int currentIndex)
{
    Mat ret;
    if (currentIndex < (int)imageList.size())
        ret = imread(imageList[currentIndex], CV_LOAD_IMAGE_COLOR);
    else
       cout << "There are no more images in the list!" << endl;

   return ret;
} 

// calculate the coordinates of board corners in world coord system
void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.clear();

    // findChessboardCorners() searches by rows, so we need to comply with it?
    // No it does not matter at all...
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
            corners.push_back(Point3f(i*squareSize, j*squareSize, 0));
}

bool runCalibration(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                    vector<vector<Point2f> > imagePoints, vector<vector<Point3f> > objectPoints,
                    vector<Mat>& rvecs, vector<Mat>& tvecs,
                    vector<float>& reprojErrs, double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if (flag & CV_CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = 1.0;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    // find intrinsic and extrinsic camera parameters
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize,
            cameraMatrix, distCoeffs, rvecs, tvecs, flag|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

    cout << "Re-projection error reported by calibrateCamera(): " << rms << endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}

double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
                                 const vector<vector<Point2f> >& imagePoints,
                                 const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                 const Mat& cameraMatrix, const Mat& distCoeffs,
                                 vector<float>& perViewErrors)
{
    vector<Point2f> imagePointsProjected;
    int totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (int i = 0; i < (int)objectPoints.size(); i++)
    {
        // project 3D points to an image plane
        projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePointsProjected);
        // calculate an absolute difference norm
        err = norm(Mat(imagePoints[i]), Mat(imagePointsProjected), NORM_L2);

        int n = (int)objectPoints.size();
        perViewErrors[i] = (float)sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return sqrt(totalErr/totalPoints);
}

// write calibration result to the output file
void saveCameraParams(Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
                      const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                      const vector<float>& reprojErrs, double totalAvgErr)
{
    FileStorage fs(outputFileName, FileStorage::WRITE);

    char buf[1024];
    time_t tm;
    time(&tm);
    struct tm *t2 = localtime(&tm);
    strftime(buf, sizeof(buf)-1, "%c", t2);

    fs << "calibration_Time" << buf;
    fs << "numberOfGoodFrames" << (int)rvecs.size();
    fs << "image_Width" << imageSize.width;
    fs << "image_Height" << imageSize.height;
    fs << "board_Width" << boardWidth;
    fs << "board_Height" << boardHeight;
    fs << "square_Size" << squareSize;

    if(flag)
    {
        sprintf(buf, "flags: %s%s%s%s",
                flag & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flag & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flag & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flag & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tagent_dist" : "");
        cvWriteComment(*fs, buf, 0);
    }
    fs << "flagValue" << flag;

    fs << "camera_Matrix" << cameraMatrix;
    fs << "Distortion_Coefficients" << distCoeffs;
    fs << "Avg_Reprojection_Errors" << totalAvgErr;
}

void displayUndistortedImage(const vector<string>& imageList, const Mat& cameraMatrix, const Mat& distCoeffs)
{
    Mat view, viewUndistorted;
    Mat map1, map2;
#if 0
    for (int i = 0; i < (int)imageList.size(); i++)
    {
        view = imread(imageList[i], CV_LOAD_IMAGE_COLOR);
        undistort(view, viewUndistorted, cameraMatrix, distCoeffs);
        imshow("Camera Calibration", viewUndistorted);

        char key = waitKey();
        if (key == 'q' || key == ESC_KEY)
            break;
    }
#elif 1
    //--------------------------------------------------
    // undistort() is simply a combination of initUndistortRectifyMap() and remap(),
    // and actually we only need to call initUndistortRectifyMap() once.
    // So this is faster:
    //--------------------------------------------------
    // compute the undistortion and rectification transformation map
    // (getOptimalNewCameraMatrix() 4th arg is alpha. see refman)
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
            imageSize, CV_16SC2, map1, map2);
    // apply a generic geometrical transformation to an image
    for (int i = 0; i < (int)imageList.size(); i++)
    {
        view = imread(imageList[i], CV_LOAD_IMAGE_COLOR);
        if (view.empty())
            continue;
        remap(view, viewUndistorted, map1, map2, INTER_LINEAR);
        imshow("Original Image", view);
        imshow("Undistorted Image", viewUndistorted);

        char key = waitKey();
        if (key == 'q' || key == ESC_KEY)
            break;
    }
#endif
}
