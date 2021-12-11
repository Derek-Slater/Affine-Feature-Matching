# ABRISK and Feature Matching, Documentation
### Derek Slater & Shakeel Khan

# The Goal
&emsp;Our goal was to implement ASIFT (affine-scale invariant feature transform). As will be described below, this goal quickly changed to be something slightly different.
# What Was Accomplished
&emsp;We started the project by first using an external resource of what appeared to be a full implementation of ASIFT for OpenCV. The implementation worked after slight modification but ran considerably slower than standard SIFT. We ended up replacing SIFT with BRISK (**B**inary **R**obust **I**nvariant **S**calable **K**eypoints) in the ASIFT implementation we used, and so what we accomplished may be more correctly called *ABRISK* (or Affine BRISK). We initially did the swap because we believed the implementation for SIFT in OpenCV was only found in its contrib package (an optional package you need to compile yourself), but we found that BRISK showed a significant speedup compared to SIFT anyways (after finding out we were able to just substitute BRISK for SIFT) and decided to stick with it. Indeed, the paper for BRISK claims that it can be faster than even SURF by an order of magnitude in some cases.

&emsp;We then moved to a higher scope of feature matching, in which we attempted to match descriptors from two images. To do this, we started with a brute force KNN (k nearest neighbors) matching, with k = 2, and we tried to eliminate bad matches by using Lowe’s ratio test. 

&emsp;For our first working implementation of feature matching, this is what we got, with 50 of the “best” matches shown:![image](https://user-images.githubusercontent.com/60079702/145663668-6170f880-f2ef-4165-a6cc-734019158316.png)	
&emsp;We found this to be of poorer quality than expected, but after finding that we had initially improperly implemented the Lowe’s test, and then fixing it (it was the difference between > and <), we eventually ended up with match counts similar to those found here for the same pair of images (close to 900 matches): <https://web.archive.org/web/20210805133647/http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html>.

&emsp;Our decision to both eliminate bad matches and to limit how many matches were displayed was due to a couple of reasons. First, eliminating bad matches simply improved the quality of the resulting keypoints, with a negligible cost to speed with how simple Lowe’s ratio test is. And second, it was to improve the visual output to aid with determining the quality of the matches. The first image shown below is when bad matches are eliminated, but we allow all of them to be displayed. And the second is when we limit how many are displayed, but without eliminating any bad matches first.

![image](https://user-images.githubusercontent.com/60079702/145663708-943ef73d-9ae9-472a-b77d-7472b08079a8.png)

![image](https://user-images.githubusercontent.com/60079702/145663710-448a2048-0b6e-4b37-91e1-780cbd1ca282.png)

&emsp;The first image makes it extremely difficult to tell if there’s even any bad matches in the mix, making it hard to tell if a change in what keypoint descriptor we used made a difference in quality. And the second image just simply contains bad matches.

&emsp;We then decided to try and speed up the matching because of how, for the magazine images, it took around 38 seconds to match keypoints with the brute force matching we were using. We first tried a method called locality-sensitive hashing because of how it’s meant to work best with binary descriptors such as BRISK. Though it showed a slight speedup, it wasn’t particularly noticeable and so we decided to try out KD-tree matching with FLANN through OpenCV. This proved to provide the kind of speed up we were looking for, and so we set it as the default matching mode for our final implementation.

&emsp;Finally, for some additional speedup, we utilized threading for some parallelization for our ABRISK implementation. It uses 5 threads to split up the work of performing affine transformations on the original image, detecting keypoints and computing descriptors. This made our ABRISK detector 1.5-2 times faster than it originally was. We considered adding more threads, but that would make the program more complicated, and we figured most people running our code would probably have a quad-core machine (it’s most common in laptops), so eventually adding more threads could have a negative impact on performance. So we left the number of threads at 5.
# Results
## Tables comparing Brute Force vs. KD-Tree matching
Data was generated from the pair of magazine pictures.

![image](https://user-images.githubusercontent.com/60079702/145663721-810f00cd-8662-49c2-9d88-a937befa5795.png)

![image](https://user-images.githubusercontent.com/60079702/145663722-29760198-40f4-494b-b775-63bf6b380f04.png)

## Image results (max of 75 matches shown for each image)
### ABRISK + FLANN-Based Matcher + Parallelization
**869 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663726-07c876e0-f1b8-4eff-8101-8a7113ca3d4d.png)
Left keypoints: 1381ms, Right keypoints: 1329ms, 5808ms to match keypoints
### BRISK + FLANN-Based Matcher
**1 Match**

![image](https://user-images.githubusercontent.com/60079702/145663731-feec0f36-6762-4225-842a-9747aedb2857.png)
Left keypoints: 125ms, Right keypoints: 138ms, 417ms to match keypoints

### ABRISK + FLANN-Based Matcher + Parallelization
**856 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663737-d55eed77-b296-4612-8b32-4c3964a2e97a.png)
Left keypoints: 1011ms, Right keypoints: 1020ms, 3492ms to match keypoints
### BRISK + FLANN-Based Matcher
**164 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663738-70e05333-11b8-4ffa-9e03-7d1c70633ac0.png)
Left keypoints: 99ms, Right keypoints: 102ms, 279ms to match keypoints

### ABRISK + FLANN-Based Matcher + Parallelization
**522 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663742-18977e14-fcd0-4ded-99f3-19f25e256a37.png)
Left keypoints: 1490ms, Right keypoints: 958ms, 10689ms to match keypoints
### BRISK + FLANN-Based Matcher
**2 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663746-8187e03a-498b-4e03-b371-584004855cf6.png)
Left keypoints: 153ms, Right keypoints: 111ms, 448ms to match keypoints

### ABRISK + FLANN-Based Matcher + Parallelization
**57 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663748-69e4abbb-a170-45fe-8ac5-1da664c729d0.png)
Left keypoints: 590ms, Right keypoints: 646ms, 1118ms to match keypoints
### BRISK + FLANN-Based Matcher
**13 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663750-ab43d3f6-9c01-4d49-8635-4fae8988b332.png)
Left keypoints: 51ms, Right keypoints: 60ms, 93ms to match keypoints

### ABRISK + FLANN-Based Matcher + Parallelization
**191 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663751-d86b028f-e20b-4ee0-9987-f741870cfaca.png)
Left keypoints: 2732ms, Right keypoints: 2157ms, 13387ms to match keypoints
### BRISK + FLANN-Based Matcher
**19 Matches**

![image](https://user-images.githubusercontent.com/60079702/145663755-43cdbf6b-a0df-4a0e-9c0a-2e67ebb29d01.png)
Left keypoints: 231ms, Right keypoints: 229ms, 838ms to match keypoints
## Results Analysis
&emsp;As expected, in comparison to BRISK, ABRISK was generally superior in quality of matching in exchange for reduced speed. Although it generally took around 10 times as long as BRISK, it was able to capture magnitudes more keypoints. We expect that if we were to have used something such as CUDA to improve parallelization for ABRISK and feature matching, that it’s quite possible the speed would be comparable or greater than (non-parallelized) BRISK.

&emsp;Even though ABRISK found more matches in every case that we tested, it’s reasonable to say that it definitely doesn’t need to be used for every situation. In images in which the transformations to get from one to another were more linear, BRISK had relatively similar accuracy in finding good keypoints compared to ABRISK. It’s pictures such as the one with the graffiti and the magazine that it’s best suited for, as BRISK fails under these conditions of heavy perspective change. With BRISK resulting in only 1 match between the magazine images, and 0 correct matches in the graffiti images, compared to the 869 and 522 matches resulting from ABRISK, respectively.

# Lessons Learned and Possible Improvements
&emsp;We learned the difficulties of determining the quality and accuracy of matched keypoints. Despite implementing a metric for the average distance ratio between keypoints as a rough estimate of quality, we remained unsure of how well this demonstrated matching quality and mostly stuck with judging the quality from how the matching visually appeared to be.

&emsp;We learned, more clearly, the differences between SIFT and ASIFT, as well as some of the specifics of BRISK. BRISK, or Binary Robust Invariant Scalable Keypoints is a descriptor, that, as the name implies, uses binary descriptors. Under the hood it’s like SIFT and SURF, but it’s much faster than even SURF in some cases by an order of magnitude (at least, according to the authors of the paper describing it). We didn’t compare SIFT with BRISK in a formal manner, which means we didn’t record any data, but we did indeed find it to be much faster than SIFT.

&emsp;There a few of possible improvements we could have made if we had extra time. One of which would be incorporating GPU/CUDA parallelization in both keypoint detection and matching. Though dated (considering the CPU/GPU used), this graph from the OpenCV documentation shows the possible speedup that might be expected:

Tesla C2050 versus Core i5-760 2.8Ghz, SSE, TBB ([opencv.org/platforms/cuda/](https://opencv.org/platforms/cuda/))

![image](https://user-images.githubusercontent.com/60079702/145663827-e5c88911-714c-40bb-8d53-18ef4abfdc30.png)

&emsp;To also further filter out bad matches, we could have determined the fundamental matrix for a pair of images through RANSAC, and eliminate matches that did not fit on the calculated epipolar line. Although Lowe’s ratio test already trims the matches relatively well for how simple it is, this would only help add further quality to the matches.

&emsp;And finally, as briefly talked about earlier, we had troubles assessing the quality of the matches without just using the visuals, and so something else we could have done would be to search for more ways to evaluate the quality of matches.
# References
Yu, Guoshen, and Jean-Michel Morel. “ASIFT: An Algorithm for Fully Affine Invariant Comparison.” *Image Processing On Line*, 24 Feb. 2011, <https://www.ipol.im/pub/art/2011/my-asift/>.

Leutenegger, Stefan, et al. “Brisk: Binary Robust Invariant Scalable Keypoints.” <http://margaritachli.com/papers/ICCV2011paper.pdf>.
