// ASiftDetector.cpp
// Author: Matt Sheckells, from http://www.mattsheckells.com/opencv-asift-c-implementation/
// Modified by: Derek Slater, Shakeel Khan

// Misc. imports.
#include "ASiftDetector.h"

// STD imports.
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <thread>

using namespace std;

// OpenCV imports.
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


ASiftDetector::ASiftDetector() { }

void ASiftDetector::detectAndCompute(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors)
{
    keypoints.clear();
    descriptors = Mat(0, 128, CV_32F);
    thread threads[5];
    for (int tl = 1; tl < 6; tl++)
    {
        threads[tl - 1] = thread(&ASiftDetector::compute, this, tl, cref(img), ref(keypoints), ref(descriptors));
    }

    for (int i = 0; i < 5; i++)
    {
        threads[i].join();
    }
}

void ASiftDetector::compute(int tl, const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors)
{
    double t = pow(sqrt(2), tl - 1);
    for (int phi = 0; phi < 180; phi += 72.0 / t)
    {
        vector<KeyPoint> kps;
        Mat desc;

        Mat timg, mask, Ai;
        img.copyTo(timg);

        affineSkew(t, phi, timg, mask, Ai);
#if 0
        Mat img_disp;
        bitwise_and(mask, timg, img_disp);
        namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
        imshow("Skew", img_disp);
        waitKey(0);
#endif

        Ptr<BRISK> ptrBrisk = BRISK::create();
        ptrBrisk->detect(timg, kps, mask);

        ptrBrisk->compute(timg, kps, desc);

        for (unsigned int i = 0; i < kps.size(); i++)
        {
            Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
            Mat kpt_t = Ai * Mat(kpt);
            kps[i].pt.x = kpt_t.at<float>(0, 0);
            kps[i].pt.y = kpt_t.at<float>(1, 0);
        }
        km.lock();
        keypoints.insert(keypoints.end(), kps.begin(), kps.end());
        km.unlock();

        dm.lock();
        descriptors.push_back(desc);
        dm.unlock();
    }
}

void ASiftDetector::affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
    int h = img.rows;
    int w = img.cols;

    mask = Mat(h, w, CV_8UC1, Scalar(255));

    Mat A = Mat::eye(2, 3, CV_32F);

    if (phi != 0.0)
    {
        phi *= M_PI / 180.;
        double s = sin(phi);
        double c = cos(phi);

        A = (Mat_<float>(2, 2) << c, -s, s, c);

        Mat corners = (Mat_<float>(4, 2) << 0, 0, w, 0, w, h, 0, h);
        Mat tcorners = corners * A.t();
        Mat tcorners_x, tcorners_y;
        tcorners.col(0).copyTo(tcorners_x);
        tcorners.col(1).copyTo(tcorners_y);
        vector<Mat> channels;
        channels.push_back(tcorners_x);
        channels.push_back(tcorners_y);
        merge(channels, tcorners);

        Rect rect = boundingRect(tcorners);
        A = (Mat_<float>(2, 3) << c, -s, -rect.x, s, c, -rect.y);

        warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
    }

    if (tilt != 1.0)
    {
        double s = 0.8 * sqrt(tilt * tilt - 1);
        GaussianBlur(img, img, Size(0, 0), s, 0.01);
        resize(img, img, Size(0, 0), 1.0 / tilt, 1.0, INTER_NEAREST);
        A.row(0) = A.row(0) / tilt;
    }

    if (tilt != 1.0 || phi != 0.0)
    {
        h = img.rows;
        w = img.cols;
        warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
    }

    invertAffineTransform(A, Ai);
}