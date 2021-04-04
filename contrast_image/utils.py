import math
import numpy as np
import cv2 as cv

class Utils:
	def __init__(self, image, color_space = 'HSV'):
		self.image = image
		self.color_space = color_space

	def image_gray(self):
		if (self.color_space == 'HSV'):
			image_hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
			self.image_color = image_hsv

			return image_hsv[:, :, 2]
		elif (self.color_space == 'Gray'):
			image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
			self.image_color = image_gray

			return image_gray
		else:
			self.image_color = self.image
			return self.image

	def LUT_image(self, LUT):
		if (self.color_space == 'HSV'):
			for i in range(0, len(self.image_color)):
				for j in range(0, len(self.image_color[0])):
					self.image_color[i][j][2] = LUT[self.image_color[i][j][2]]

			return cv.cvtColor(self.image_color, cv.COLOR_HSV2BGR)
		elif (self.color_space == 'Gray'):
			return LUT[self.image_color]
		else:
			return self.image_color

	def is_gray_image(self):
		blue, gree, red = cv2.split(self.image)

		difference_red_green = np.count_nonzero(abs(red - green))
		difference_green_blue = np.count_nonzero(abs(green - blue))
		difference_blue_red = np.count_nonzero(abs(blue - red))

		difference_sum = float(difference_red_green + difference_green_blue + difference_blue_red)

		ratio = diff_sum / self.image.size

		if ratio>0.005:
				return False
		else:
				return True

	def minimum_mean_brightness_error(self, image_1d):
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

	
	

def RGB_TO_HSI(image):
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
