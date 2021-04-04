import math
import numpy as np
import numpy.ma as ma
import cv2 as cv
from .utils import Utils
from .histogram import Histogram

class CI:
	def __init__(self, image, color_space = 'HSV'):
		self.image = image
		self.color_space = color_space

	########################################
	#
	# Global histogram
	#
	########################################

	# Global Histogram Equalization
	def GHE(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		histogram = Histogram()
		LUT = histogram.histogram_equalization(image_1d)
		return utils.LUT_image(LUT)

	# Brightness-preserving Bi-Histogram Equalization
	def BBHE(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		mean = np.mean(image_1d)
		mean = math.floor(mean)
		histogram = Histogram()
		LUT = histogram.histogram_equalization_threshold(image_1d, mean)
		return utils.LUT_image(LUT)

	# Dualistic Sub-Image Histogram Equalization
	def DSIHE(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		median = np.median(image_1d)
		median = math.floor(median)
		histogram = Histogram()
		LUT = histogram.histogram_equalization_threshold(image_1d, median)
		return utils.LUT_image(LUT)

	# Minimum Mean Brightness Error Histogram Equalization
	def MMBEBHE(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		mbe = utils.minimum_mean_brightness_error(image_1d)
		histogram = Histogram()
		LUT = histogram.histogram_equalization_threshold(image_1d, mbe)
		return utils.LUT_image(LUT)

	# Recursive Mean-Separate Histogram Equalization
	def RMSHE(self, recursive = 2):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		histogram = Histogram()
		LUT = histogram.histogram_equalization_recursively(image_1d, np.mean, recursive)
		return utils.LUT_image(LUT)

	# Recursive Sub-Image Histogram Equalization
	def RSIHE(self, recursive = 2):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		histogram = Histogram()
		LUT = histogram.histogram_equalization_recursively(image_1d, np.median, recursive)
		return utils.LUT_image(LUT)

	# Brightness Preserving Histogram Equalization with Maximum Entropy
	def BPHEME(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		mean = np.mean(image_1d)
		if mean == 127.5:
			cdf_output = np.array([mean] * 255)
		else:
			def maximum_histogram_entropy(x, mean):
				return (x * math.exp(x) - math.exp(x) + 1) / (x * (math.exp(x) - 1)) - mean

			def derivative_maximum_histogram_entropy(x):
				return (-math.exp(x) * (x * x + 2) + math.exp(2 * x) + 1) / ((math.exp(x) - 1) * (math.exp(x) - 1) * x * x)

			def maximum_cumulative_entropy(x):
				return np.array([(math.exp(x * (i / 255)) - 1) / (math.exp(x) - 1) for i in range(0, 256)])

			scale_mean = mean / 255
			lamda = 1
			output_newton_method = maximum_histogram_entropy(lamda, scale_mean)

			while abs(output_newton_method) > 0.01:
				lamda = lamda - (output_newton_method / derivative_maximum_histogram_entropy(lamda))
				output_newton_method = maximum_histogram_entropy(lamda, scale_mean)
			
			cdf_output = maximum_cumulative_entropy(lamda)
		
		pdf_input, _ = np.histogram(image_1d, 256, [0, 255])
		cdf_input = pdf_input.cumsum()

		histogram = Histogram()
		LUT = histogram.cdf_matching(cdf_input, cdf_output)
		return utils.LUT_image(LUT)

	# Range Limited Bi-Histogram Equalization
	def RLBHE(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
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

		histogram = Histogram()
		LUT = histogram.histogram_equalization_threshold(image_1d, otsu_threshold, x_0, x_l)
		return utils.LUT_image(LUT)

	# Flattest Histogram Specification with Accurate Brightness Preservation
	def FHSABP(self):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		pdf_input, _ = np.histogram(image_1d, 256, [0, 255])
		pdf_input = pdf_input / 1.0
		pdf_output = np.array([0.0] * 256)

		mean = np.mean(image_1d)
		if mean < 84.67:
			x_0 = 3 * math.floor(mean) + 1
			a = (-6 * x_0 + 12 * mean) / (x_0 * (x_0 + 1) * (x_0 + 2))
			b = (4 * x_0 - 6 * mean + 2) / ((x_0 + 1) * (x_0 + 2))

			for i in range(0, 256):
				pdf_output[i] = max(0, a * i + b)
		elif mean <= 170.33:
			a = (mean - 127.5) / 1398080
			b = (511 - 3 * mean) / 32896

			for i in range(0, 256):
				pdf_output[i] = a * i + b
		else:
			invert_mean = 255 - mean
			x_0 = 3 * math.floor(invert_mean) + 1
			a = (-6 * x_0 + 12 * invert_mean) / (x_0 * (x_0 + 1) * (x_0 + 2))
			b = (4 * x_0 - 6 * invert_mean + 2) / ((x_0 + 1) * (x_0 + 2))

			for i in range(0, 256):
				pdf_output[i] = max(0, a * i + b)
		
		histogram = Histogram()
		LUT = histogram.histogram_specification(pdf_input, pdf_output)
		return utils.LUT_image(LUT)

	# Weighted Thresholded Histogram Equalization
	def WTHE(self, root = 0.5, value = 0.5, lower = 0):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()
		length = len(image_1d)

		pdf, _ = np.histogram(image_1d, 256, [0, 255])
		pdf = pdf / length

		if value < 1:
			upper = pdf.max() * value
		else:
			upper = pdf.max()

		weight_pdf = np.array([0.0] * 256)
		for i in range(0, 256):
			if pdf[i]  < lower:
				weight_pdf[i] = 0
			elif pdf[i] < upper:
				weight_pdf[i] = upper * ((pdf[i] - lower) / (upper - lower)) ** root
			else:
				weight_pdf[i] = upper
			
		weight_pdf_sum = np.sum(weight_pdf)
		weight_pdf_scale = weight_pdf / weight_pdf_sum

		histogram = Histogram()
		LUT = histogram.sub_histogram_equalization(weight_pdf_scale)
		return utils.LUT_image(LUT)

	# Recursive Separated and Weighted Histogram Equalization
	def RSWHE(self, type = 'mean', recursive = 2):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()
		length = len(image_1d)
		
		histogram = Histogram()
		if (type == 'mean'):
			histogram_segmentation = histogram.histogram_segmentation(image_1d, np.mean, recursive)
		elif (type == 'median'):
			histogram_segmentation = histogram.histogram_segmentation(image_1d, np.median, recursive)

		pdf, _ = np.histogram(image_1d, 256, [0, 255])
		highest_probabilitiy = pdf.max() / length
		lowest_probability = pdf.min() / length

		image_mean = np.mean(image_1d)
		image_min = image_1d.min()
		image_max = image_1d.max()
		image_middle = (int(image_min) + int(image_max)) / 2
		beta = highest_probabilitiy * abs(image_mean - image_middle) / (image_max - image_min)

		histogram_weight = []
		for sub_histogram in histogram_segmentation:
			sub_histogram_scale = sub_histogram / length
			alpha = np.sum(sub_histogram_scale)
			for i in range(0, len(sub_histogram_scale)):
				sub_histogram_scale[i] = highest_probabilitiy * ((sub_histogram_scale[i] - lowest_probability) / (highest_probabilitiy - lowest_probability)) ** alpha + beta
			
			histogram_weight += [sub_histogram_scale]

		histogram_weight_sum = 0
		for sub_histogram in histogram_weight:
			histogram_weight_sum += np.sum(sub_histogram)
		
		histogram_weight_scale = []
		for sub_histogram in histogram_weight:
			sub_histogram = sub_histogram / histogram_weight_sum
			histogram_weight_scale += [sub_histogram]

		start = 0
		end = -1
		LUT = np.array([])
		for sub_histogram in histogram_weight_scale:
			start = end + 1
			end = start + len(sub_histogram) - 1
			LUT = np.concatenate((LUT, histogram.sub_histogram_equalization(sub_histogram, start, end)))
		
		return utils.LUT_image(LUT)

	# Automatic Weighting Mean-separated Histogram Equalization
	# def AWMHE(image, color_space = 'HSV'):
	# 	image_gray, image_color = utils.image_gray(image, color_space)
	# 	image_1d = image_gray.flatten()

	# 	pdf, _ = np.histogram(image_1d, 256, [0, 255])
	# 	cdf = pdf.cumsum()

	# 	print(histogram.histogram_weighting_mean(cdf))

	########################################
	#
	# Adaptive histogram
	#
	########################################

	# Adaptive Histogram Equalization
	def AHE(self):
		return self.image

	# Non-Overlapped Sub-block Histogram Equalization
	def NOSHE(self):
		return self.image

	# Partially Overlapped Sub-block Histogram Equalization
	def POSHE(self):
		return self.image

	# Cascadede Multistep Binominal Filtering Histogram Equalization
	def CMBFHE(self):
		return self.image

	# Contrast Limited Adaptive Histogram Equalization
	def CLAHE(self):
		return self.image


	########################################
	#
	# Gamma Correction
	#
	########################################


	# Adaptive Gamma Correction with Weighting Distribution
	def AGCWD(self, alpha = 0.5):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()
		length = len(image_1d)

		pdf, _ = np.histogram(image_1d, 256, [0, 255])
		pdf = pdf / length
		highest_probabilitiy = pdf.max()
		lowest_probability = pdf.min()

		weight_distribution = highest_probabilitiy * ((pdf - lowest_probability) / (highest_probabilitiy - lowest_probability)) ** alpha
		weight_distribution_sum = np.sum(weight_distribution)

		weight_distribution_scale = weight_distribution / weight_distribution_sum

		LUT = np.array([0.0] * 256)
		for i in range(0, 256):
			LUT[i] = 255 * (i / 255) ** (1 - weight_distribution_scale[i])
		LUT = LUT.astype('uint8')
		
		return utils.LUT_image(LUT)

	# Adaptive Gamma Correction Color Preserving Framework
	def AGCCPF(self, alpha = 0.5):
		utils = Utils(self.image, self.color_space)
		image_gray = utils.image_gray()
		image_1d = image_gray.flatten()

		pdf, _ = np.histogram(image_1d, 256, [0, 255])
		
		image_equalization = self.GHE()
		image_equalization_1d = image_equalization.flatten()

		pdf_equalization, _ = np.histogram(image_equalization_1d, 256, [0, 255])

		smooth_pdf = 0.5 * pdf + 0.5 * pdf_equalization
		smooth_pdf_scale = smooth_pdf / np.sum(smooth_pdf)

		cdf = smooth_pdf_scale.cumsum()

		LUT = np.array([0.0] * 256)
		for i in range(0, 256):
			LUT[i] = 255 * (i / 255) ** (1 - cdf[i])
		LUT = LUT.astype('uint8')
		
		return utils.LUT_image(LUT)