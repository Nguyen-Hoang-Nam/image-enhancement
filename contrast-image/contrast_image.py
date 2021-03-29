import numpy as np
import numpy.ma as ma
import cv2 as cv
import math

def RGB_TO_HSI(img):

	with np.errstate(divide='ignore', invalid='ignore'):

		#Load image with 32 bit floats as variable type
		bgr = np.float32(img) / 255

		#Separate color channels
		blue = bgr[:,:,0]
		green = bgr[:,:,1]
		red = bgr[:,:,2]

		minimum = np.minimum(np.minimum(red, green), blue)
		maximum = np.maximum(np.maximum(red, green), blue)
		delta = maximum - minimum

		intensity = np.divide(blue + green + red, 3)

		if intensity == 0:
			saturation = 0
		else:
			saturation = 1 - 3 * np.divide(minimum, red + green + blue)

		sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))
		if (green >= blue).any():
			hue = np.arccos((1 / 2 * ((red-green) + (red - blue)) / sqrt_calc))
		else:
			hue = 2 * math.pi - np.arccos((1 / 2 * ((red-green) + (red - blue)) / sqrt_calc))

		hue = hue * 180 / math.pi

		#Merge channels into picture and return image
		hsi = cv2.merge((intensity, saturation, hue))
		return hsi

def pdf(image, parameter = 'HSI'):
	if parameter == 'GRAY':
		image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		image_1d = image_gray.flatten()
		
		histogram, _ = np.histogram(image_1d, 256, [0, 256])
	elif parameter == 'INTENSITY':
		image_hsi = RGB_TO_HSI(image)

		intensity = image_hsi[:, :, 0]
		intensity_1d = intensity.flatten()

	return histogram


def equalization(image_1d, range_min = 0, range_max = 255):
	histogram, _ = np.histogram(image_1d, range_max - range_min + 1, [range_min, range_max])
	histogram_mask = np.ma.masked_equal(histogram, 0)

	length = len(image_1d)
	cdf_mask = histogram_mask.cumsum()

	cdf_mask_equalization = ((cdf_mask - cdf_mask.min()) * (range_max - range_min) / (cdf_mask.max() - cdf_mask.min())) + range_min
	cdf_equalization = np.ma.filled(cdf_mask_equalization, 0).astype('uint8')

	return cdf_equalization

# Traditional Histogram Equalization
def THE(image):
	# histogram = pdf(image, 'GRAY')
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	cdf_equalization = equalization(image_1d)
	image_gray_equalization = cdf_equalization[image_gray]

	return image_gray_equalization

# Bi-Histogram Equalization
def BHE(image_1d, threshold):
	lower_filter = image_1d <= threshold
	lower_1d = image_1d[lower_filter]

	upper_filter = image_1d > threshold
	upper_1d = image_1d[upper_filter]

	lower_equalization = equalization(lower_1d, 0, threshold)
	upper_equalization = equalization(upper_1d, threshold + 1, 255)

	histogram_equalization = np.concatenate((lower_equalization, upper_equalization))

	
	return histogram_equalization

# Brightness-preserving Bi-Histogram Equalization
def BBHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	mean = np.mean(image_gray)
	mean = math.floor(mean)

	image_1d = image_gray.flatten()

	histogram_equalization = BHE(image_1d, mean)
	return histogram_equalization[image_gray]

# Dualistic Sub-Image Histogram Equalization
def DSIHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	medium = np.median(image_gray)
	medium = math.floor(medium)

	image_1d = image_gray.flatten()

	histogram_equalization = BHE(image_1d, medium)
	return histogram_equalization[image_gray]

# Minimum Mean Brightness Error
def MMBE(image_1d):
	length = len(image_1d)

	unique_1d = np.unique(image_1d)
	max_1d = len(unique_1d)

	histogram, _ = np.histogram(image_1d, 256, [0, 255])

	mean = 0
	for i in range(0, len(unique_1d)):
		mean += i * histogram[unique_1d[i]]

	smbe = max_1d * (length - histogram[unique_1d[0]]) - 2 * mean
	asmbe = abs(smbe)
	position = 0
	for i in range(1, len(unique_1d)):
		smbe += (length - max_1d * histogram[unique_1d[i]])
		if asmbe > abs(smbe):
			asmbe = abs(smbe)
			position = i

	return unique_1d[position]

# Dualistic Sub-Image Histogram Equalization
def MMBEBHE(image):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	mbe = MMBE(image_1d)

	histogram_equalization = BHE(image_1d, mbe)
	return histogram_equalization[image_gray]

def RMSH(image_1d, start, end, recursive):
	if recursive > 0:
		mean = np.mean(image_1d)
		mean = math.floor(mean)

		lower_filter = image_1d <= mean
		lower_1d = image_1d[lower_filter]
		
		lower_equalization = RMSH(lower_1d, start, mean, recursive - 1)

		upper_filter = image_1d > mean
		upper_1d = image_1d[upper_filter]

		upper_equalization = RMSH(upper_1d, mean + 1, end, recursive - 1)

		return np.concatenate((lower_equalization, upper_equalization))
	else:
		return equalization(image_1d, start, end)

# Recursive Mean-Separate Histogram Equalization
def RMSHE(image, recursive = 1):
	image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	image_1d = image_gray.flatten()

	histogram_equalization = RMSH(image_1d, 0, 255, recursive)
	# print(histogram_equalization)
	return histogram_equalization[image_gray]