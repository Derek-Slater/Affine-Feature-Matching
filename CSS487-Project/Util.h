// File: Util.h
// 
// This header file describes the interface fir a collection of utility methods for
// testing the ABRISK detector.
// 
// Authors: Derek Slater, Shakeel Khan

#pragma once

// Misc. imports.
#include "ABRISKDetector.h"		// ABRISK implementation.

// STD imports.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>

using namespace std;

// OpenCV imports.
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Constants.
const int BEST_MATCHES_TO_DISPLAY = 75;
const float DISTANCE_RATIO_THRESHOLD = 0.7f;

// loadImages: Loads the two images for the specified set (this is for internal use).
// Preconditions: setNum must be between 0-4.
// Postconditions: The respective images are read into img1 and img2, and a boolean is
//				   returned indicating whether the images were successfully loaded.
bool loadImages(Mat& img1, Mat& img2, int setNum);

// BRISKDetectAndCompute: Performs keypoint detection and descriptor calculation via
//						  BRISK.
// Preconditions: Image passed in is a Mat that represents a grayscale image.
// Postconditions: Keypoints and descriptors will be filled in.
void BRISKDetectAndCompute(const Mat& image,
	vector<KeyPoint>& keypoints,
	Mat& descriptors);

// extractBestMatches: Uses Lowe's ratio test to get the best matches.
// Preconditions: vector<vector<DMatch>> matches contains matches between images.
// Postconditions: the best matches from matches will be pushed into bestMatches, and the
//				   ratioSum will contain the sum of all the distance ratios between
//				   matches.
void extractBestMatches(const vector<vector<DMatch>>& matches,
	vector<DMatch>& bestMatches,
	float& ratioSum);

// trimBestMatches: Trims bestMatches so as to not clutter the display for too many
//				    matches.
// Preconditions: bestMatches is not empty.
// Postconditions: bestMatches will only contain at most BEST_MATCHES_TO_DISPLAY entries.
void trimBestMatches(vector<DMatch>& bestMatches);

// findKeypoints: Finds the keypoints within a given image and computes their descriptors.
// Preconditions: Specify whether you would like to use the ABRISK method by setting
//				  abrisk to true, or regular BRISK by setting abrisk to false.
// Postconditions: Any keypoints found will be put in keypoints and their corresponding
//				   descriptor will be places in descriptors.
void findKeypoints(const Mat& img,
	vector<KeyPoint>& keypoints,
	Mat& descriptors,
	bool abrisk);

// matchDescriptors: Matches the descriptors from one set to another.
// Preconditions: Set bruteForce to true if you want to use the brute force method for
//				  matching, otherwise set it to false.
// Postconditions: The matches found will be placed in matches.
void matchDescriptors(Mat& descriptors1, Mat& descriptors2,
	vector<vector<DMatch>>& matches,
	bool bruteForce);

// showAndSave: Draws and displays the found matches of keypoints from the two given
// images, then saves it with the given filename.
// Preconditions: keypoints1[i] must have a corresponding point in
//				  keypoints2[matches[i]]. 
// Postconditions: An image showing the matches between the two given images will be
//				   shown and saved to disk with the given filename.
void showAndSave(const Mat& image1, const vector<KeyPoint> keypoints1,
	const Mat& image2, const vector<KeyPoint> keypoints2,
	const vector<DMatch> matches,
	string filename);

// findAndMatchKeypoints: Given two images, this will find the keypoints within each
//						  image, match them, then draw the matches between the two
//						  images. The resulting image with the matches drawn is then
//					      saved to disk.
// Preconditions: You must specify whether you would like to use ABRISK (abrisk - true if
//				  so), the brute force matching method (bruteForce - true if so), and the
//				  filename for the resulting image (fileName).
// Postconditions: The resulting image with the drawn matches will be saved to disk with
//				   the specified filename.
void findAndMatchKeypoints(const Mat& image1, const Mat& image2,
	bool abrisk, bool bruteForce, string fileName);