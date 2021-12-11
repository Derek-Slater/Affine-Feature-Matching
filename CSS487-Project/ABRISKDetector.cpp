// File: ABRISKDetector.cpp
// 
// This file provides an implementation for ABRISK, or Affine-BRISK, a keypoint detection
// algorithm that is invariant to affine transformations and uses BRISK.
// 
// Author: Matt Sheckells: http://www.mattsheckells.com/opencv-asift-c-implementation/
// Modified by: Derek Slater, Shakeel Khan

// Misc. imports.
#include "ABRISKDetector.h"

// STD imports.
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

// detectAndCompute: Find keypoints within the given image and computes their
//                   descriptors.
// Preconditions: None.
// Postconditions: All keypoints found are stored in keypoints and their descriptors are
//                 put in descriptors.
void const ABRISKDetector::detectAndCompute(const Mat& img,
                                            vector<KeyPoint>& keypoints,
                                            Mat& descriptors)
{
    // Clear both data structures.
    keypoints.clear();
    descriptors = Mat(0, 128, CV_32F);

    // If we're building the parallel version, we need an array for the threads.
#ifdef PARALLELIZE
    thread threads[NUM_THREADS];
#endif // PARALLELIZE

    
    for (int tl = 1; tl < 6; tl++)
    {
        // If we're building the parallel version, start a thread, otherwise, call
        // computeTask directly.
#ifdef PARALLELIZE
        threads[tl - 1] = thread(&ABRISKDetector::computeTask, this,
                                    tl, cref(img), ref(keypoints), ref(descriptors));
#else
        computeTask(tl, img, keypoints, descriptors);
#endif // PARALLELIZE
    }

    // If we're building the parallel version, we need to join all our threads.
#ifdef PARALLELIZE
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads[i].join();
    }
#endif // PARALLELIZE
}

// computeTask: This is where we perform several affine transformations of the input
//              image and for each of them find keypoints and compute their descriptors.
// Preconditions: tl must range from 1-5.
// Postconditions: The keypoints found will be put in keypoints and their descriptors are
//                 put in descriptors.
void const ABRISKDetector::computeTask(int tl, const Mat &img,
                                        vector<KeyPoint> &keypoints, Mat &descriptors)
{
    double t = pow(sqrt(2), tl - 1);
    for (int phi = 0; phi < 180; phi += 72.0 / t)
    {
        vector<KeyPoint> kps;
        Mat desc;

        Mat timg, mask, Ai;
        img.copyTo(timg);

        affineSkew(t, phi, timg, mask, Ai);

        // Detect the kepoints and compute their descriptors.
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

        // Store our keypoints.
        keypointsMutex.lock();
        keypoints.insert(keypoints.end(), kps.begin(), kps.end());
        keypointsMutex.unlock();

        // Along with their descriptors.
        descriptorsMutex.lock();
        descriptors.push_back(desc);
        descriptorsMutex.unlock();
    }
}

// affineSkew: Performs an affine transformation according to the specified parameters.
// Preconditions: img must be a valid grayscale (CV_8UC1) image.
// Postconditions: Performs an affine transformation according to the specified
//                 parameters to img. Also stores the inverse of the affine
//                 transformation in Ai and the mask for the transformation into mask.
void const ABRISKDetector::affineSkew(double tilt, double phi,
                                        Mat& img, Mat& mask, Mat& Ai)
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

        warpAffine(img, img, A, Size(rect.width, rect.height),
                    INTER_LINEAR, BORDER_REPLICATE);
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