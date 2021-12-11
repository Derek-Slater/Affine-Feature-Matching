// File: main.cpp
// 
// This is the driver for the program where all the tests are run.
// 
// Authors: Derek Slater, Shakeel Khan

// Misc. imports.
#include "ABRISKDetector.h"		// ABRISK implementation.
#include "Util.h"				// Utility methods used for testing the ABRISK detector.

// STD imports.
#include <iostream>
#include <cstdlib>
#include <string>

using namespace std;

// OpenCV imports.
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

// Constants
const string IMAGE_1_FILENAME = "image1.png";
const string IMAGE_2_FILENAME = "image2.png";

// runPerformanceTests: This is used to compare the performance of all the combinations
//						of the 3 different detection methods (BRISK, ABRISK,
//						parallelized ABRISK) and the 2 matching methods (KNN bruteforce
//						and KD-tree). Various metrics are printed to the terminal and the
//						an images with the matches drawn between the two inputs is shown
//						and saved to disk.
// Preconditions: You must define the constants IMAGE_1_FILENAME and IMAGE_2_FILENAME
//				  with the names of the images to be used. To run the parallel tests, you
//				  must ensure the preprocessor macro PARALLELIZE is defined at the top of
//				  the header file for ABRISKDetector. To run the regular sequential BRISK
//				  you can uncomment that line.
// Postconditions: Various metrics are printed to the terminal for comparison and the
//				   resulting image with the matches drawn is shown and saved to disk.
void runPerformanceTests()
{
	// Load the images
	Mat image1 = imread("Input/" + IMAGE_1_FILENAME, IMREAD_GRAYSCALE);
	Mat image2 = imread("Input/" + IMAGE_2_FILENAME, IMREAD_GRAYSCALE);

	// Simple error checking.
	if (image1.data == NULL || image2.data == NULL) {
		cout << "One of the two image strings are either invalid or don't exist" << endl;
		return;
	}

	/* ===== ABRISK ===== */

	// ABRISK + Brute-force KNN matching.
#ifdef PARALLELIZE
	string filename = "ABRISK-KNN-Parallel";
#else
	string filename = "ABRISK-KNN-Sequential";
#endif // PARALLELIZE

	cout << "=====" << filename << "=====" << endl;
	findAndMatchKeypoints(image1, image2, true, true, filename + ".png");
	cout << endl;

	// ABRISK + FLANN-Based Matcher.
#ifdef PARALLELIZE
	filename = "ABRISK-KD-Parallel";
#else
	filename = "ABRISK-KD-Sequential";
#endif // PARALLELIZE

	cout << "=====" << filename << "=====" << endl;
	findAndMatchKeypoints(image1, image2, true, false, filename + ".png");
	cout << endl;

	/* ===== BRISK ===== */

	// BRISK + Brute-force KNN matching.
	filename = "BRISK-KNN";

	cout << "=====" << filename << "=====" << endl;
	findAndMatchKeypoints(image1, image2, false, true, filename + ".png");
	cout << endl;

	// BRISK + FLANN-Based Matcher.
	filename = "BRISK-KD";

	cout << "=====" << filename << "=====" << endl;
	findAndMatchKeypoints(image1, image2, false, false, filename + ".png");
	cout << endl;
}

// runDemo: This goes over the 5 sets of input images comparing the results of BRISK to
//			ABRISK.
// Preconditions: Ensure the 5 sets of input images are in the Input/ directory.
// Postconditions: The number of matches found is printed to the terminal and the result
//				   images with the matches drawn are saved to disk.
void runDemo()
{
	// 5 sets of images to go over.
	for (int i = 0; i < 5; i++)
	{
		// Load the images.
		Mat img1, img2;
		loadImages(img1, img2, i);

		cout << "===== " << "Set " << i + 1 << " =====" << endl;

		// ABRISK + FLANN-Based Matcher.
		cout << "===== " << "ABRISK + FLANN-Based Matcher" << " =====" << endl;
		string filename = format("Set %i ABRISK", i + 1);
		findAndMatchKeypoints(img1, img2, true, false, filename + ".png");

		// BRISK + FLANN-Based Matcher.
		cout << endl << "===== " << "BRISK + FLANN-Based Matcher" << " =====" << endl;
		filename = format("Set %i BRISK", i + 1);
		findAndMatchKeypoints(img1, img2, false, false, filename + ".png");

		cout << endl << endl;
	}
}

// main: The driver where we run tests.
// Preconditions: None.
// Postconditions: Depends on the test - see the documentation for each test.
int main(int argc, char* argv[])
{
	// This removes the logging messages from the terminal.
	utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
	
	// If you'd like to run the performance tests, uncomment this line.
	// To test with the sequential version of BRISK, make sure the PARALLELIZE
	// preprocessor macro isn't defined at the top of ABRISKDetector.h. To test with the
	// parallel version, make sure the PARALLELIZE macro is defined at the top of
	// ABRISKDetector.h
	//runPerformanceTests();

	// If you'd like to reproduce the images found in the presentation slides, uncomment
	// this line.
	runDemo();

	// Exit.
	return EXIT_SUCCESS;
}