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
const string IMAGE_1_FILENAME = "image9.png";
const string IMAGE_2_FILENAME = "image10.png";

const int BEST_MATCHES_TO_DISPLAY = 75;

const float DISTANCE_RATIO_THRESHOLD = 0.7f;

// BRISKDetectAndCompute: Performs keypoint detection and descriptor calculation via BRISK.
// Preconditions: Image passed in is a Mat that represents a grayscale image.
// Postconditions: Keypoints and descriptors will be filled in.
void BRISKDetectAndCompute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
	Ptr<BRISK> ptrBrisk = BRISK::create();
	ptrBrisk->detect(image, keypoints);
	ptrBrisk->compute(image, keypoints, descriptors);
}

// extractBestMatches: Uses Lowe's ratio test to get the best matches.
// Preconditions: vector<vector<DMatch>> matches contains matches between images.
// Postconditions: the best matches from matches will be pushed into bestMatches,
// and the ratioSum will contain the sum of all the distance ratios between matches.
void extractBestMatches(const vector<vector<DMatch>>& matches, vector<DMatch> &bestMatches, float ratioSum) {
	for (int i = 0; i < matches.size(); i++) {
		float distanceRatio = matches[i][0].distance / matches[i][1].distance;
		if (distanceRatio <= DISTANCE_RATIO_THRESHOLD) {
			bestMatches.push_back(matches[i][0]);
			ratioSum += distanceRatio;
		}
	}
}

// trimBestMatches: Trims bestMatches so as to not clutter the display for too many matches.
// Preconditions: bestMatches is not empty.
// Postconditions: bestMatches will only contain at most BEST_MATCHES_TO_DISPLAY entries.
void trimBestMatches(vector<DMatch>& bestMatches) {
	sort(bestMatches.begin(), bestMatches.end());
	vector<DMatch>::iterator end = bestMatches.end();
	if (bestMatches.size() > BEST_MATCHES_TO_DISPLAY) {
		end = bestMatches.begin() + BEST_MATCHES_TO_DISPLAY;
	}
	bestMatches = vector<DMatch>(bestMatches.begin(), end);
}

// main:
// Preconditions: 
// Postconditions: 
int main(int argc, char* argv[])
{
	// Load up the images.
	Mat image1 = imread(IMAGE_1_FILENAME, IMREAD_GRAYSCALE);
	Mat image2 = imread(IMAGE_2_FILENAME, IMREAD_GRAYSCALE);

	// Simple error checking.
	if (image1.data == NULL || image2.data == NULL) {
		cout << "One of the two image strings are either invalid or don't exist" << endl;
		return 0;
	}
	
	// Find keypoints and descriptors using ASIFT.
	ASiftDetector asd;
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;

	// First image.
	auto startTime = chrono::high_resolution_clock::now();
	//BRISKDetectAndCompute(image1, keypoints1, descriptors1);
	asd.detectAndCompute(image1, keypoints1, descriptors1);		// ASIFT.
	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	cout << keypoints1.size() << " Keypoints for first image found ("
			<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Second image.
	startTime = chrono::high_resolution_clock::now();
	//BRISKDetectAndCompute(image2, keypoints2, descriptors2);
	asd.detectAndCompute(image2, keypoints2, descriptors2);		// ASIFT.
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << keypoints2.size() << " Keypoints for second image found ("
		<< time / chrono::milliseconds(1) << " ms)" << endl;
	cout << "Performing matching..." << endl;
	
	// Match descriptors between images.
	FlannBasedMatcher matcher;
	//BFMatcher matcher = BFMatcher(NORM_HAMMING);
	vector<vector<DMatch>> matches;

	descriptors1.convertTo(descriptors1, CV_32F);				// Convert to CV_32F to work with FLANN.
	descriptors2.convertTo(descriptors2, CV_32F);				// Convert to CV_32F to work with FLANN.
	startTime = chrono::high_resolution_clock::now();
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << "Found " << matches.size() << " matches("
			<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Extract the best matches using Lowe's ratio test.
	vector<DMatch> bestMatches;
	float ratioSum = 0;
	startTime = chrono::high_resolution_clock::now();
	extractBestMatches(matches, bestMatches, ratioSum);
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << "# of Good Matches Found: "  << bestMatches.size() << " ("
			<< time / chrono::milliseconds(1) << " ms)" << endl;
	float averageRatio = ratioSum / bestMatches.size();
	cout << "Average distance ratio among good matches: " << averageRatio << endl;

	// Choose the 75 best matches.
	trimBestMatches(bestMatches);

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