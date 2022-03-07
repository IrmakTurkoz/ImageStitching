

# IMAGE STITCHING

# A. Feature Descriptors
Since we were allowed to use existing code packages, We
have used OpenCV’s SIFT for feature extraction from the
images. One challenge we have faced was the newly released
version of OpenCV has removed the SIFT due to license
issues, so instead, we had to use the older version’s SIFT
[1]. We call out the method and it returns the key-points and
features of the image. Each of the descriptors has the same
size of features, which is fixed to 128 by default. We have not
to change the default value of the SIFT descriptor package.
#  B. Matching Feature Points
To match the features of two descriptors, we calculate
two images feature points separately and do one to one
match for each of the set in two images. In each iteration,
we calculate the distance between two descriptors using
”numpy.linalg.norm” which applies n2 normalization to the
distances we gather from the feature points [2]. To map every
pair of key points together, we call match feature points to
function in every step. However, since we have many matches,
we calculate the best k best matches and then save those
values. Afterward, we use these values to decide if a match
is a good match or not. Currently, our system uses k = 2,
which means it calculates the first and second-best matches
of keypoints. Then, a ratio is given as a hyperparameter. This
ratio decides if our first match is relatively much better than
the second one when multiplied with ratio. Our experiments
concluded that the best value for ratio 0.75. Therefore, we
eliminate most of the false matches if they do not have a big
difference between their first order match and second-order
match. This is called Lowe’s ratio test [3] and it has proven
to work best with SIFT matches with removing 90% of the
false matches. Once we eliminate the matches, we check if the
size of raw matches is larger than 4, because we can calculate
homography with at least 4 points.
#  C. Homography
To find homography, we are given a set of points from
image1 that match to set of points from image2. One important
aspect of homography is to not lose the order of the points
because their indices are ordered with respect to the matched
points. In the first developed version, a huge mistake was to
shuffle the points with different seeds. The system did not
work at all, as we were not trying to fit homography between
the matched pairs but any of the points in the images. Once
we have fixed the seed for both of the points, we have defined
a RANSAC size of 4, which means that We always get a
random subset of size 4 from the matches. We also define a
max iteration size, which decides how many times we will try
out different subset to find the best homography matrix. We
define two steps of the process to find a homography matrix
at each step of the iteration. The first one is the creation of
the homography matrix with given points of the first image
and second image. We are using the Eq. (1) to calculate
homography matrix where the (x1, y 2), (x2, y 2), (x3, y 3),
(x4, y 4) are the points from the first image and (x1
’, y 2
’),
(x2
’, y 2
’), (x3
’, y 3
’), (x4
’, y 4
’) are points from the second
image.
![image](https://user-images.githubusercontent.com/12885387/157049811-425d88c3-caa4-4f12-940d-35f849c9cfab.png)


Because we should be able to do this for any arbitrary
number points where the RANSAC size is a parameter,
numpy.linalg.svd is used to compose singular value transformation.
The output is a 9x1 matrix, and then we convert it
to a 3x3 matrix which we call homography transformation.
However, a proper normalization is needed to ensure h9 is
always 1, so we also divide the each of the element in the
homography matrix to the h9.
Once we find the homography matrix from the 4 points
subset, we test out for the rest of the points. To do this, we
use Eq. (2) where we put the However because our points
set still have false matches; it is not logical to expect that
the homography matrix will work for all of the points. The
false matches may increase the error and cause us to find
punish a good homography matrix. In order to prevent this,
we test the points only with the subset of it. This was another
hyperparameter, to decide which percentage of the points we
are aiming to get the homography matrix accurate for. We
have experimented for some values and found out that only
testing a random subset of one of the third of the point set is
enough to decide the homography matrix is good or not The
error is calculated with the least-squares error, as it was stated
in course slides. Once we find a less error than the previous
minimum error, we update the error and homography matrix
to the current iteration. Once we reach the maximum number
of iterations, we quit the findHomography function.
Max iteration size has a significant effect to get accurate
results. Because we always pick random subsets, the accuracy
of the homography matrix increase as we have more trials.
Convergence could be checked however, it would take much
more time because we need to check every each of the
combinations. In our implementation, we are able to find not
the global minima in terms of error but we can get very
close, or find a local minima. Essentially, to stitch two images;
max iteration can be between 30000 - 40000 because it does
not have to be a perfect homography matrix; however to
stitch four images; max iteration should be at least 12000-
13000 to gather more accurate results which will be fed to
the next stitching network. Another important point with max
iteration is; when we have more repeating patterns in the
image, we need more iterations to be more precise about
the best homography matrix. This is why, we have only max
iteration as 3000 in 3 whereas we have max iteration as 15000
in 2. Moreover, we can also increase the RANSAC size which
is equal to 4 by default, when we give the two images that
have repeating patterns so that, we are checking correlations
between keypoints.

![image](https://user-images.githubusercontent.com/12885387/157049936-c83df0b3-876e-4b7f-94e3-52b230096fda.png)
![image](https://user-images.githubusercontent.com/12885387/157049947-c6d72654-e223-4347-9dc8-633775a3f012.png)(2)

# D. Warping Image
To warp the second image with respect to the first image,
we use WarpPerspective. The basic idea is to make the first
image clipped and apply the homography matrix to the second
image to scale, localize, shear and apply any perspective
transformation to the second image. One constraint with our
program is that we always allow side stitching. This means
that we fix the height of the image as a constraint. Since the
example images are only side stitched images, we have not
described another warping function to warp images from top
to bottom.
The warping perspective was most challenging when we
have shared objects. Because we stretch the image with a float
value; we need to cast it to an integer as pixels do not have float
indices. This is either done with ceiling or flooring the results.
However, this would cause black marks in the direction of
stretching because the in-between pixels are remained empty.
To solve this, we have not an efficient but sufficient method.
We first construct the new image with only taking the floor of
the resulting matrix when we multiply homography with the
subject, and then we repaint the ceil - floor combination of x
and y coordinates.
We also create an alpha mask for the newly inserted image,
to be used in the alpha blending and find the corner points of
the new image, however as we will discuss in II-E we could
not complete it.
# E. Results
![image](https://user-images.githubusercontent.com/12885387/157050164-9148aa4e-7521-429a-8cee-0e7906cccfe0.png)
Fig. 1: Image 1 and Image 2 Stitching result
![image](https://user-images.githubusercontent.com/12885387/157050188-a7fd3414-dd5f-4454-90a3-94646c4f5938.png)
Fig. 2: Image 3 through Image 6 Stitching result
![image](https://user-images.githubusercontent.com/12885387/157050216-075f69f3-1a5e-4fb6-ba59-30d551d240c6.png)
Fig. 3: Image 7 and Image 8 Stitching result
# References
[1] G. Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools,
2000.
[2] Travis Oliphant. NumPy: A guide to NumPy. USA: Trelgol Publishing,
2006–. [Online; accessed ¡today¿].
[3] D.G. Lowe. Distinctive image features from scale-invariant keypoints.
International Journal of Computer Vision 60, 91–110.
