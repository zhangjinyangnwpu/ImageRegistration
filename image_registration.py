# Created by Qixun Qu
# quqixun@gmail.com
# 2017/04/10
#


import numpy as np
from regex import P
from affine_ransac import Ransac
from align_transform import Align
from affine_transform import Affine
import matplotlib.pyplot as plt



# Affine Transform
# |x'|  = |a, b| * |x|  +  |tx|
# |y'|    |c, d|   |y|     |ty|
# pts_t =    A   * pts_s  + t

# -------------------------------------------------------------
# Test Class Affine
# -------------------------------------------------------------
point_num = 1000
# Create instance
af = Affine()

# Generate a test case as validation with
# a rate of outliers
outlier_rate = 0.9
A_true, t_true, pts_s, pts_t = af.create_test_case(outlier_rate,point_num=1000)

# At least 3 corresponding points to
# estimate affine transformation
K = 3
# Randomly select 3 pairs of points to do estimation
idx = np.random.randint(0, pts_s.shape[1], (K, 1))
A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

# Display known parameters with estimations
# They should be same when outlier_rate equals to 0,
# otherwise, they are totally different in some cases
print("gt_info:")
print(A_true, '\n', t_true, '\n')
print("estimate_affine:")
print(A_test, '\n', t_test, '\n')

# -------------------------------------------------------------
# Test Class Ransac
# -------------------------------------------------------------

# Create instance
rs = Ransac(K=K, threshold=1)

residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)

# Run RANSAC to estimate affine tansformation when
# too many outliers in points set
A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
print("ransac:")
print(A_rsc, '\n', t_rsc, '\n')

colors = ['green' if i in inliers[0] else 'red' for i in range(point_num)]
colors = np.array(colors)

plt.subplot(1, 2, 1)
plt.scatter(pts_s[0,:], pts_s[1,:],c=colors)

plt.subplot(1, 2, 2)
plt.scatter(pts_t[0,:], pts_t[1,:],c=colors)

plt.show()

# -------------------------------------------------------------
# Test Class Align
# -------------------------------------------------------------

# Load source image and target image
source_path = 'Images/mona_source.png'
target_path = 'Images/mona_target.jpg'

# Create instance
al = Align(source_path, target_path, threshold=1)

# Image transformation
al.align_image()
