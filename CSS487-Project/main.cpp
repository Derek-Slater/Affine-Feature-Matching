// main.cpp
// 
// Authors: Derek Slater, Shakeel Khan

// STD imports.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>
using namespace std;

// OpenCV imports.
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

// Misc. imports.
#include "ASiftDetector.h"		// ASIFT implementation.

// Constants.
const string IMAGE_1_FILENAME = "image1.png";
const string IMAGE_2_FILENAME = "image2.png";

const int BEST_MATCHES_TO_DISPLAY = 75;

const float DISTANCE_RATIO_THRESHOLD = 0.65f;

// standardSIFT: Performs the standard function of SIFT.
// Preconditions: image passed in is a Mat that represents a grayscale image.
// Postconditions: keypoints and descriptors will be filled in.
void SIFTDetectAndCompute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
	Ptr<BRISK> ptrBrisk = BRISK::create();
	ptrBrisk->detect(image, keypoints);
	ptrBrisk->compute(image, keypoints, descriptors);
}

// main:
// preconditions: 
// postconditions: 
int main(int argc, char* argv[])
{
	// Load up the images.
	Mat image1 = imread(IMAGE_1_FILENAME, IMREAD_GRAYSCALE);
	Mat image2 = imread(IMAGE_2_FILENAME, IMREAD_GRAYSCALE);
	
	// Find keypoints and descriptors using ASIFT.
	ASiftDetector asd;
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;

	// First image.
	auto startTime = chrono::high_resolution_clock::now();
	SIFTDetectAndCompute(image1, keypoints1, descriptors1);
	//asd.detectAndCompute(image1, keypoints1, descriptors1);		// ASIFT.
	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	cout << "Keypoints for first image found ("
			<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Second image.
	startTime = chrono::high_resolution_clock::now();
	SIFTDetectAndCompute(image2, keypoints2, descriptors2);
	//asd.detectAndCompute(image2, keypoints2, descriptors2);		// ASIFT.
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << "Keypoints for second image found ("
		<< time / chrono::milliseconds(1) << " ms)" << endl;
	cout << "Performing matching..." << endl;
	
	// Match descriptors between images.
	BFMatcher bfm = BFMatcher(NORM_HAMMING);
	vector<vector<DMatch>> matches;
	vector<DMatch> bestMatches;

	startTime = chrono::high_resolution_clock::now();
	bfm.knnMatch(descriptors1, descriptors2, matches, 2);
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << "Found " << matches.size() << " matches("
			<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Extract the best matches.
	startTime = chrono::high_resolution_clock::now();
	for (int i = 0; i < matches.size(); i++) {
		float distanceRatio = matches[i][0].distance / matches[i][1].distance;
		if (distanceRatio <= DISTANCE_RATIO_THRESHOLD) {
			bestMatches.push_back(matches[i][0]);
		}
	}
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << "# of Good Matches Found: "  << bestMatches.size() << " ("
			<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Choose the 75 best matches.
	sort(bestMatches.begin(), bestMatches.end());
	vector<DMatch>::iterator end = bestMatches.end();
	if (bestMatches.size() > BEST_MATCHES_TO_DISPLAY)
	{
		end = bestMatches.begin() + BEST_MATCHES_TO_DISPLAY;
	}
	bestMatches = vector<DMatch>(bestMatches.begin(), end);

	// Draw the matches between the images and display them.
	Mat matchesImage;
	drawMatches(image1, keypoints1,
				image2, keypoints2,
				bestMatches, matchesImage, 
				Scalar::all(-1), 1, vector< char >(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("myoutput", matchesImage);
	waitKey(0);

	// Save the result.
	imwrite("matches.png", matchesImage);

	return EXIT_SUCCESS;
}