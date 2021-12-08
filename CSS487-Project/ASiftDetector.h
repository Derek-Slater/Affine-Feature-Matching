// ASiftDetector.h
// Author: Matt Sheckells, from http://www.mattsheckells.com/opencv-asift-c-implementation/

#define NUM_THREADS 5

// STD imports.
#include <vector>
#include <thread>

using namespace std;

// OpenCV imports.
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;

class ASiftDetector
{
	public:
		ASiftDetector();

		void const detectAndCompute(const Mat& img, vector< KeyPoint >& keypoints, Mat& descriptors);

	private:
		mutex keypointsMutex, descriptorsMutex;

		void const computeTask(int tl, const Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors);
		void const affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai);
};