import math
import numpy as np

def cdf_matching(input_cdf, output_cdf):
  ratio = input_cdf[255] / output_cdf[255]
  output_cdf = output_cdf * ratio

  LUT = np.array([])
  position_new = 0
  for i in range(0, 256):
    while output_cdf[position_new] < input_cdf[i]:
      position_new += 1
    
    LUT = np.append(LUT, position_new)

  return LUT

def histogram_specification(input_histogram, output_histogram):
	input_cdf = input_histogram.cumsum()
	output_cdf = output_histogram.cumsum()
	
	return cdf_matching(input_cdf, output_cdf)

def sub_histogram_equalization(histogram, range_min = 0, range_max = 255):
	cdf = histogram.cumsum()
	cdf_mask = np.ma.masked_equal(cdf, 0)

	# Scale cdf to [range_min, range_max]
	scale_cdf_mask = ((cdf_mask - cdf_mask.min()) * (range_max - range_min) / (cdf_mask.max() - cdf_mask.min())) + range_min
	LUT = np.ma.filled(scale_cdf_mask, 0).astype('uint8')

	return LUT

def histogram_equalization(image_1d, range_min = 0, range_max = 255):
	histogram, _ = np.histogram(image_1d, range_max - range_min + 1, [range_min, range_max])

	cdf = histogram.cumsum()
	cdf_mask = np.ma.masked_equal(cdf, 0)

	# Scale cdf to [range_min, range_max]
	scale_cdf_mask = ((cdf_mask - cdf_mask.min()) * (range_max - range_min) / (cdf_mask.max() - cdf_mask.min())) + range_min
	LUT = np.ma.filled(scale_cdf_mask, 0).astype('uint8')

	return LUT

def histogram_equalization_threshold(image_1d, threshold, start = 0, end = 255):
	lower_filter = image_1d <= threshold
	lower_1d = image_1d[lower_filter]

	upper_filter = image_1d > threshold
	upper_1d = image_1d[upper_filter]

	lower_input_lut = np.array([])
	if start > 0:
		for i in range(0, start):
			lower_input_lut = np.append(lower_input_lut, i)

	upper_input_lut = np.array([])
	if end < 255:
		for i in range(end + 1, 256):
			upper_input_lut = np.append(upper_input_lut, i)

	lower_LUT = histogram_equalization(lower_1d, start, threshold)
	upper_LUT = histogram_equalization(upper_1d, threshold + 1, end)

	lower_LUT = np.concatenate((lower_input_lut, lower_LUT))
	upper_LUT = np.concatenate((upper_LUT, upper_input_lut))

	LUT = np.concatenate((lower_LUT, upper_LUT))

	return LUT

def histogram_equalization_recursively(image_1d, separate_func, recursive, start = 0, end = 255):
	if recursive > 0:
		separate = separate_func(image_1d)
		separate = math.floor(separate)

		lower_filter = image_1d <= separate
		lower_1d = image_1d[lower_filter]
		
		lower_equalization = histogram_equalization_recursively(lower_1d, separate_func, recursive - 1, start, separate)

		upper_filter = image_1d > separate
		upper_1d = image_1d[upper_filter]

		upper_equalization = histogram_equalization_recursively(upper_1d, separate_func, recursive - 1, separate + 1, end)

		return np.concatenate((lower_equalization, upper_equalization))
	else:
		return histogram_equalization(image_1d, start, end)

def histogram_segmentation(image_1d, separate_func, recursive, start = 0, end = 255):
	if recursive > 0:
		separate = separate_func(image_1d)
		separate = math.floor(separate)

		lower_filter = image_1d <= separate
		lower_1d = image_1d[lower_filter]
		
		lower_equalization = histogram_segmentation(lower_1d, separate_func, recursive - 1, start, separate)

		upper_filter = image_1d > separate
		upper_1d = image_1d[upper_filter]

		upper_equalization = histogram_segmentation(upper_1d, separate_func, recursive - 1, separate + 1, end)

		return lower_equalization + upper_equalization
	else:
		sub_histogram, _ = np.histogram(image_1d, end - start + 1, [start, end])
		return [sub_histogram]