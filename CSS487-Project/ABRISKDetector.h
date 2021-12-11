// File: ABRISKDetector.h
// 
// This header file describes an interface for a class which implements ABRISK, a
// keypoint detection algorithm that is invariant to affine transformations and uses
// BRISK.
// 
// Author: Matt Sheckells: http://www.mattsheckells.com/opencv-asift-c-implementation/
// Modified by: Derek Slater, Shakeel Khan

#define NUM_THREADS 5
#define PARALLELIZE

// STD imports.
#include <vector>
#include <thread>

using namespace std;

// OpenCV imports.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

class ABRISKDetector
{
	public:
		// detectAndCompute: Find keypoints within the given image and computes their
		//                   descriptors.
		// Preconditions: None.
		// Postconditions: All keypoints found are stored in keypoints and their
		//				   descriptors are put in descriptors.
		void const detectAndCompute(const Mat& img,
									vector<KeyPoint>& keypoints,
									Mat& descriptors);

	private:
		mutex keypointsMutex, descriptorsMutex;

		// computeTask: This is where we perform several affine transformations of the
		//              input image and for each of them find keypoints and compute their
		//				descriptors.
		// Preconditions: tl must range from 1-5.
		// Postconditions: The keypoints found will be put in keypoints and their
		//				   descriptors are put in descriptors.
		void const computeTask(int tl, const Mat &img,
								vector<KeyPoint> &keypoints, Mat &descriptors);

		// affineSkew: Performs an affine transformation according to the specified
		//			   parameters.
		// Preconditions: img must be a valid grayscale (CV_8UC1) image.
		// Postconditions: Performs an affine transformation according to the specified
		//                 parameters to img. Also stores the inverse of the affine
		//                 transformation in Ai and the mask for the transformation into
		//				   mask.
		void const affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai);
};