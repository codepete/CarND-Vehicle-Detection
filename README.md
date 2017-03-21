**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_visualization.png
[image3]: ./examples/pipeline_output_without_heatmap.png
[image4]: ./examples/heatmap_visualization.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

If you look at the first 3 section IPython file, I started by reading in all the `vehicle` and `non-vehicle` images provided for this project. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I experimented with different color spaces such RGB, HSV, and YUV, but ultimately found that YCrCb generated the best results.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

This was purely through experimentation. I started with `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and already got decent results. With this I modified each parameter at a time to see the resulting HOG images - halving, increasing/decreasing orientations, etc. However, I found that the values I started with did very well and decided to stick with the original values.

Please take a look at my Jupiter Notebook and look at the `Visualizing HOG` section to see how I played around with HOG values.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In my Jupiter Notebook, in the section `Training Classifier`, you'll see that I take the images I loaded in the very first section and:
- Utilized HOG (all color channels), Spatial Binning of Colors, and Color Histogram for full feature extraction using my `extract_features` method for both car and non-car images.
-  I appended the results together and created a label to match the resulting feature array.
- I scaled the features using `StandardScaler` from sklearn
- Utilized `train_test_split` to generate our training and test set (20% of our entire feature set)
- I then utilized `LinearSVC` to train the classifier.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the `Testing Classifier`, I went with the sub-sampling window search approach for a more efficient sliding window approach. I essentially extract the hog features, from `ystart=400` to `ystop=646` (as that section of the image contains the majority of the features we care about), once and sub-sample the results.

In my first attempt I overdid the amount of scales to use. I started out incrementing by .3 starting at 1.0 to 3.4. I then started playing with different intervals and settled with increments of .5 from 1.0 to 3.0 and found that it did OK identifying the cars.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images (this is before heat map technique to remove false positives):

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_track1.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the the `Multiple Detections & False Positives` section of my Jupiter Notebook, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded (zeroing out pixels below threshold of 3) that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In addition to this, to smooth out the results I utilize the last 10 heatmaps in order to create a better transition from frame to frame.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are visualizations of the test images and their corresponding heatmap. The left image is after application (looks at section 2 above to see what it looked like before) of heatmap technique. The right image is what the heat map looks like.

![alt text][image4]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that the processing of the video was very slow. I attempted to rectify this issue by limiting scales <=2.0 to the upper half of hog features I extracted, and anything greater then 2.0 to the bottom half. This significantly sped up my processing. However, its still not very fast.

I still find that finding the "perfect" parameters for hog and scales really had to do with a lot of trial and error. Although, the results are OK I find that there is still a lot left to be tinkered with to achieve even more optimal results.

There are issues with cars in the very left on cars going in the opposite direction. Although it is detecting cars correctly I struggled with completely eliminating them. I found also that there would be some false positives, but overall I think the pipeline did OK.

I definitely feel like there should be a better way to do scaling. I don't think it would perform as well if there were large cargo trucks in the left or right lanes in the road (with the current approach). I also think that this would fail to detect large cargo trucks right in front.

I want to try using CNN approach with a much more robust set of training data to see if I can get better results. I am currently experimenting with Tiny YOLO to see if I can generate better and cleaner results.  
