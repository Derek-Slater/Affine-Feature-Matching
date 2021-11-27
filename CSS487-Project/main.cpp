// main.cpp
// 
// Authors: Derek Slater, Shakeel Khan

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "ASiftDetector.h"

#include <iostream>
using namespace std;

//standardSIFT: Performs the standard function of SIFT
//Preconditions: image passed in is a Mat that represents a grayscale image
//Postconditions: keypoints and descriptors will be filled in
void standardSIFT(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) {
	Ptr<BRISK> ptrBrisk = BRISK::create();
	ptrBrisk->detect(image, keypoints);
	ptrBrisk->compute(image, keypoints, descriptors);
}

// main:
// preconditions: 
// postconditions: 
int main(int argc, char* argv[])
{
	//if (argc != 3)
	//{
	//	cout << " Need to give two image file paths" << endl; 
	//	return -1;
	//}
	string image1 = "image1.png";
	Mat my_image1; 
	my_image1 = imread(image1, IMREAD_GRAYSCALE);
	string image2 = "image2.png";
	Mat my_image2;
	my_image2 = imread(image2, IMREAD_GRAYSCALE);

	
	//find keypoints and descriptors using asift
	ASiftDetector asd;

	//first image
	Mat descriptors1;
	vector<KeyPoint> keypoints1;
	//standardSIFT(my_image1, keypoints1, descriptors1);
	asd.detectAndCompute(my_image1, keypoints1, descriptors1); //asift
	cout << "Keypoints for first image found" << endl;

	//second image
	Mat descriptors2;
	vector<KeyPoint> keypoints2;
	//standardSIFT(my_image2, keypoints2, descriptors2);
	asd.detectAndCompute(my_image2, keypoints2, descriptors2); //asift
	cout << "Keypoints for second image found\nPerforming matching..." << endl;

	//match descriptors between images
	BFMatcher bfm = BFMatcher(NORM_L1);
	vector<vector<DMatch>> matches;
	bfm.knnMatch(descriptors1, descriptors2, matches, 2);
	vector<DMatch> bestMatches;
	//get best matches
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance > matches[i][1].distance * 0.5f) {
			bestMatches.push_back(matches[i][0]);
		}
	}
	sort(bestMatches.begin(), bestMatches.end());
	int bestMatchesToDisplay = 75;
	cout << "Total Good Matches: " << bestMatches.size() << endl;
	bestMatches = vector<DMatch>(bestMatches.begin(), bestMatches.begin() + bestMatchesToDisplay);

	//display keypoints found
	//Mat keypointImage1;
	//drawKeypoints(my_image1, keypoints1, keypointImage1);
	//namedWindow("myoutput", WINDOW_NORMAL);
	//imshow("myoutput", keypointImage1);

	//draw the matches between the images and display them
	Mat matchesImage;
	drawMatches(my_image1, keypoints1, my_image2, keypoints2, bestMatches, matchesImage, 
				Scalar::all(-1), 1, vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("myoutput");
	imshow("myoutput", matchesImage);
	imwrite("matches.png", matchesImage);
	waitKey(0);
	destroyAllWindows();

	return 0;
}