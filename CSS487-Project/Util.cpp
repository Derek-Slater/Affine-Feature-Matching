// File: Util.cpp
// 
// This file provides an implementation for the utility methods specified in Util.h.
// 
// Authors: Derek Slater, Shakeel Khan

#include "Util.h"		// ABRISK implementation.

ABRISKDetector abd;

// loadImages: Loads the two images for the specified set (this is for internal use).
// Preconditions: setNum must be between 0-4.
// Postconditions: The respective images are read into img1 and img2, and a boolean is
//				   returned indicating whether the images were successfully loaded.
bool loadImages(Mat& img1, Mat& img2, int setNum)
{
	string img1Filename;
	string img2Filename;

	if (setNum == 0)
	{
		img1Filename = "Input/image1.png";
		img2Filename = "Input/image2.png";
	}
	else if (setNum == 1)
	{
		img1Filename = "Input/image3.png";
		img2Filename = "Input/image4.png";
	}
	else if (setNum == 2)
	{
		img1Filename = "Input/image5.png";
		img2Filename = "Input/image6.png";
	}
	else if (setNum == 3)
	{
		img1Filename = "Input/image7.png";
		img2Filename = "Input/image8.png";
	}
	else
	{
		img1Filename = "Input/image9.png";
		img2Filename = "Input/image10.png";
	}

	// Load the images
	img1 = imread(img1Filename, IMREAD_GRAYSCALE);
	img2 = imread(img2Filename, IMREAD_GRAYSCALE);

	// Simple error checking.
	return img1.data != NULL && img2.data != NULL;
}

// BRISKDetectAndCompute: Performs keypoint detection and descriptor calculation via
//						  BRISK.
// Preconditions: Image passed in is a Mat that represents a grayscale image.
// Postconditions: Keypoints and descriptors will be filled in.
void BRISKDetectAndCompute(const Mat& image,
	vector<KeyPoint>& keypoints,
	Mat& descriptors)
{
	Ptr<BRISK> ptrBrisk = BRISK::create();
	ptrBrisk->detect(image, keypoints);
	ptrBrisk->compute(image, keypoints, descriptors);
}

// extractBestMatches: Uses Lowe's ratio test to get the best matches.
// Preconditions: vector<vector<DMatch>> matches contains matches between images.
// Postconditions: the best matches from matches will be pushed into bestMatches, and the
//				   ratioSum will contain the sum of all the distance ratios between
//				   matches.
void extractBestMatches(const vector<vector<DMatch>>& matches,
	vector<DMatch>& bestMatches,
	float& ratioSum)
{
	for (int i = 0; i < matches.size(); i++)
	{
		float distanceRatio = matches[i][0].distance / matches[i][1].distance;
		if (distanceRatio <= DISTANCE_RATIO_THRESHOLD)
		{
			bestMatches.push_back(matches[i][0]);
			ratioSum += distanceRatio;
		}
	}
}

// trimBestMatches: Trims bestMatches so as to not clutter the display for too many
//				    matches.
// Preconditions: bestMatches is not empty.
// Postconditions: bestMatches will only contain at most BEST_MATCHES_TO_DISPLAY entries.
void trimBestMatches(vector<DMatch>& bestMatches)
{
	sort(bestMatches.begin(), bestMatches.end());
	vector<DMatch>::iterator end = bestMatches.end();
	if (bestMatches.size() > BEST_MATCHES_TO_DISPLAY)
	{
		end = bestMatches.begin() + BEST_MATCHES_TO_DISPLAY;
	}
	bestMatches = vector<DMatch>(bestMatches.begin(), end);
}

// findKeypoints: Finds the keypoints within a given image and computes their descriptors.
// Preconditions: Specify whether you would like to use the ABRISK method by setting
//				  abrisk to true, or regular BRISK by setting abrisk to false.
// Postconditions: Any keypoints found will be put in keypoints and their corresponding
//				   descriptor will be places in descriptors.
void findKeypoints(const Mat& img,
	vector<KeyPoint>& keypoints,
	Mat& descriptors,
	bool abrisk)
{
	if (abrisk)
	{
		abd.detectAndCompute(img, keypoints, descriptors);
	}
	else
	{
		BRISKDetectAndCompute(img, keypoints, descriptors);
	}
}

// matchDescriptors: Matches the descriptors from one set to another.
// Preconditions: Set bruteForce to true if you want to use the brute force method for
//				  matching, otherwise set it to false.
// Postconditions: The matches found will be placed in matches.
void matchDescriptors(Mat& descriptors1, Mat& descriptors2,
	vector<vector<DMatch>>& matches,
	bool bruteForce)
{
	if (bruteForce)
	{
		BFMatcher matcher = BFMatcher(NORM_HAMMING);
		matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	}
	else
	{
		FlannBasedMatcher matcher;
		descriptors1.convertTo(descriptors1, CV_32F);	// Convert to CV_32F to work
														// with FLANN.
		descriptors2.convertTo(descriptors2, CV_32F);	// Convert to CV_32F to work
														// with FLANN.
		matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	}
}

// showAndSave: Draws and displays the found matches of keypoints from the two given
// images, then saves it with the given filename.
// Preconditions: keypoints1[i] must have a corresponding point in
//				  keypoints2[matches[i]]. 
// Postconditions: An image showing the matches between the two given images will be
//				   shown and saved to disk with the given filename.
void showAndSave(const Mat& image1, const vector<KeyPoint> keypoints1,
	const Mat& image2, const vector<KeyPoint> keypoints2,
	const vector<DMatch> matches,
	string filename)
{
	// Show the matches.
	Mat matchesImage;
	drawMatches(image1, keypoints1,
		image2, keypoints2,
		matches, matchesImage,
		Scalar::all(-1), 1, vector< char >(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow(filename, matchesImage);
	waitKey(0);

	// Save it.
	imwrite("Output/" + filename, matchesImage);
}

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
	bool abrisk, bool bruteForce, string fileName)
{
	// Find keypoints and descriptors.
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;

	// First image.
	auto startTime = chrono::high_resolution_clock::now();
	findKeypoints(image1, keypoints1, descriptors1, abrisk);
	auto endTime = chrono::high_resolution_clock::now();
	auto time = endTime - startTime;
	cout << keypoints1.size() << " Keypoints for first image found ("
		<< time / chrono::milliseconds(1) << " ms)" << endl;

	// Second image.
	startTime = chrono::high_resolution_clock::now();
	findKeypoints(image2, keypoints2, descriptors2, abrisk);
	endTime = chrono::high_resolution_clock::now();
	time = endTime - startTime;
	cout << keypoints2.size() << " Keypoints for second image found ("
		<< time / chrono::milliseconds(1) << " ms)" << endl;
	cout << "Performing matching..." << endl;

	// Match descriptors between images.
	vector<vector<DMatch>> matches;
	startTime = chrono::high_resolution_clock::now();
	matchDescriptors(descriptors1, descriptors2, matches, bruteForce);
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
	cout << "# of Good Matches Found: " << bestMatches.size() << " ("
		<< time / chrono::milliseconds(1) << " ms)" << endl;
	float averageRatio = ratioSum / bestMatches.size();
	cout << "Average distance ratio among good matches: " << averageRatio << endl;

	// Choose the 75 best matches.
	trimBestMatches(bestMatches);

	// Draw the matches between the images and display them.
	showAndSave(image1, keypoints1, image2, keypoints2, bestMatches, fileName);
}