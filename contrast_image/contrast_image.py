import math
import numpy as np
import numpy.ma as ma
import cv2 as cv
import utils
import histogram

# Global Histogram Equalization
def GHE(image):
	# histogram = pdf(image, 'GRAY')
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	LUT = histogram.histogram_equalization(image_1d)
	image_gray_equalization = LUT[image_gray]

	return image_gray_equalization


########################################
#
# Brightness Preservation
#
########################################

# Brightness-preserving Bi-Histogram Equalization
def BBHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	mean = np.mean(image_1d)
	mean = math.floor(mean)
	LUT = histogram.histogram_equalization_threshold(image_1d, mean)

	return LUT[image_gray]

# Dualistic Sub-Image Histogram Equalization
def DSIHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	median = np.median(image_1d)
	median = math.floor(median)
	LUT = histogram.histogram_equalization_threshold(image_1d, median)

	return LUT[image_gray]

# Minimum Mean Brightness Error Histogram Equalization
def MMBEBHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	mbe = utils.minimum_mean_brightness_error(image_1d)
	LUT = histogram.histogram_equalization_threshold(image_1d, mbe)

	return LUT[image_gray]

# Recursive Mean-Separate Histogram Equalization
def RMSHE(image, recursive = 2):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	LUT = histogram.recursive_mean_histogram(image_1d, recursive)

	return LUT[image_gray]

# Recursive Sub-Image Histogram Equalization
def RSIHE(image, recursive = 2):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	LUT = histogram.recursive_median_histogram(image_1d, recursive)

	return LUT[image_gray]

# Brightness Preserving Histogram Equalization with Maximum Entropy
def BPHEME(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	mean = np.mean(image_1d)
	if mean == 127.5:
		cdf_output = np.array([mean] * 255)
	else:
		scale_mean = mean / 255
		lamda = 1
		output_newton_method = utils.maximum_histogram_entropy(lamda, scale_mean)

		while abs(output_newton_method) > 0.01:
			lamda = lamda - (output_newton_method / utils.derivative_maximum_histogram_entropy(lamda))
			output_newton_method = utils.maximum_histogram_entropy(lamda, scale_mean)
		
		cdf_output = utils.maximum_cumulative_entropy(lamda)
	
	pdf_input, _ = np.histogram(image_1d, 256, [0, 255])
	cdf_input = pdf_input.cumsum()
	LUT = histogram.cdf_matching(cdf_input, cdf_output)

	return LUT[image_gray]

# Range Limited Bi-Histogram Equalization
def RLBHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	otsu_threshold, _ = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	otsu_threshold = math.floor(otsu_threshold)

	pdf, _ = np.histogram(image_1d, 256, [0, 255])
	image_mean = 0
	lower_cumulative = 0
	for i in range(0, 256):
		image_mean += i * pdf[i]

		if i <= otsu_threshold:
			lower_cumulative += pdf[i]

	length = len(image_1d)
	image_mean /= length
	lower_cumulative /= length

	a = lower_cumulative
	b = 2 * image_mean - otsu_threshold - (1 - a)
	x_0 = 0
	x_l = 255
	for i in range(0, 500):
		temp = 2 * (a * x_0 + (1 - a) * x_l - b)
		x_0 -= temp * a
		x_l -= temp * (1 - a)

	if x_0 < 0:
		x_0 = 0
	elif x_0 > otsu_threshold:
		x_0 = otsu_threshold
	else:
		x_0 = math.floor(x_0)

	if x_l > 255:
		x_l = 255
	elif x_l <= otsu_threshold:
		x_l = otsu_threshold + 1
	else:
		x_l = math.floor(x_l)

	LUT = histogram.histogram_equalization_threshold(image_1d, otsu_threshold, x_0, x_l)

	return LUT[image_gray]


########################################
#
# Contrast Enhancement
#
########################################

# Adaptive Histogram Equalization
def AHE(image):
	return image

# Non-Overlapped Sub-block Histogram Equalization
def NOSHE(image):
	return image

# Partially Overlapped Sub-block Histogram Equalization
def POSHE(image):
	return image

# Cascadede Multistep Binominal Filtering Histogram Equalization
def CMBFHE(image):
	return image

# Contrast Limited Adaptive Histogram Equalization
def CLAHE(image):
	return image


########################################
#
# Brightness Preservation & Contrast Enhancement
#
########################################

# Recursive Separated and Weighted Histogram Equalization
def RSWHE(image, type = 'mean', beta = 0, recursive = 2):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()
	length = len(image_1d)
	
	histogram_segmentation = histogram.histogram_segmentation_by_mean(image_1d, recursive)

	pdf, _ = np.histogram(image_1d, 256, [0, 255])
	highest_probabilitiy = pdf.max() / length
	lowest_probability = pdf.min() / length

	histogram_weight = []
	for sub_histogram in histogram_segmentation:
		sub_histogram_scale = sub_histogram / length
		alpha = np.sum(sub_histogram_scale)
		for i in range(0, len(sub_histogram_scale)):
			sub_histogram_scale[i] = highest_probabilitiy * ((sub_histogram_scale[i] - lowest_probability) / (highest_probabilitiy - lowest_probability)) ** alpha + 255 * beta
		
		histogram_weight += [sub_histogram_scale]

	histogram_weight_sum = 0
	for sub_histogram in histogram_weight:
		histogram_weight_sum += np.sum(sub_histogram)
	
	histogram_weight_scale = []
	for sub_histogram in histogram_weight:
		sub_histogram = sub_histogram * length / histogram_weight_sum
		histogram_weight_scale += [sub_histogram.astype('uint8')]

	start = 0
	end = -1
	LUT = np.array([])
	for sub_histogram in histogram_weight_scale:
		start = end + 1
		end = start + len(sub_histogram) - 1
		LUT = np.concatenate((LUT, histogram.sub_histogram_equalization(sub_histogram, start, end)))
	
	return LUT[image_gray]