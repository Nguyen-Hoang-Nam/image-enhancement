import math
import numpy as np

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

def minimum_mean_brightness_error(image_1d):
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

def maximum_histogram_entropy(x, mean):
	return (x * math.exp(x) - math.exp(x) + 1) / (x * (math.exp(x) - 1)) - mean

def derivative_maximum_histogram_entropy(x):
	return (-math.exp(x) * (x * x + 2) + math.exp(2 * x) + 1) / ((math.exp(x) - 1) * (math.exp(x) - 1) * x * x)

def maximum_cumulative_entropy(x):
	return np.array([(math.exp(x * (i / 255)) - 1) / (math.exp(x) - 1) for i in range(0, 256)])