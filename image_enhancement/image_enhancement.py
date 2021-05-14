import math
import numpy as np
import numpy.ma as ma
import cv2 as cv
from .utils import Utils
from .histogram import Histogram

class IE:
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


    # Kim, Yeong-Taeg.
    # Contrast enhancement using brightness preserving bi-histogram equalization. 
    # IEEE transactions on Consumer Electronics 43, no. 1 (1997): 1-8.
    # Brightness-preserving Bi-Histogram Equalization (BBHE)
    def BBHE(self):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        mean = np.mean(image_1d)
        mean = math.floor(mean)
        histogram = Histogram()
        LUT = histogram.histogram_equalization_threshold(image_1d, mean)
        return utils.LUT_image(LUT)


    # Kim, Yeong-Taeg. 
    # Quantized bi-histogram equalization." 
    # In 1997 IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 4, pp. 2797-2800. IEEE, 1997.
    # Quantized Bi-Histogram Equalization (QBHE)
    def QBHE(self, number_gray):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        mean = np.mean(image_1d)
        mean = math.floor(mean)

        number_divide = 256 / number_gray
        image_1d = image_1d / number_divide
        image_1d = image_1d.astype('uint8')
        image_1d = image_1d * number_divide
        image_1d = image_1d.astype('uint8')

        mean = round(round(mean / number_divide) * number_divide)

        histogram = Histogram()
        LUT = histogram.histogram_equalization_threshold(image_1d, mean)
        return utils.LUT_image(LUT)


    # Wang, Yu, Qian Chen, and Baeomin Zhang.
    # Image enhancement based on equal area dualistic sub-image histogram equalization method.
    # IEEE Transactions on Consumer Electronics 45, no. 1 (1999): 68-75.
    # Dualistic Sub-Image Histogram Equalization (DSIHE)
    def DSIHE(self):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        median = np.median(image_1d)
        median = math.floor(median)
        histogram = Histogram()
        LUT = histogram.histogram_equalization_threshold(image_1d, median)
        return utils.LUT_image(LUT)


    # Chen, Soong-Der, and Abd Rahman Ramli. 
    # Minimum mean brightness error bi-histogram equalization in contrast enhancement.
    # IEEE transactions on Consumer Electronics 49, no. 4 (2003): 1310-1319.
    # Minimum Mean Brightness Error Histogram Equalization (MMBEBHE)
    def MMBEBHE(self):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        mbe = utils.minimum_mean_brightness_error(image_1d)
        histogram = Histogram()
        LUT = histogram.histogram_equalization_threshold(image_1d, mbe)
        return utils.LUT_image(LUT)


    # Chen, Soong-Der, and Abd Rahman Ramli. 
    # Contrast enhancement using recursive mean-separate histogram equalization for scalable brightness preservation.
    # IEEE Transactions on consumer Electronics 49, no. 4 (2003): 1301-1309.
    # Recursive Mean-Separate Histogram Equalization (RMSHE)
    def RMSHE(self, recursive = 2):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        histogram = Histogram()
        LUT = histogram.histogram_equalization_recursively(image_1d, np.mean, recursive)
        return utils.LUT_image(LUT)


    # Yang, Seungjoon, Jae Hwan Oh, and Yungfun Park. 
    # Contrast enhancement using histogram equalization with bin underflow and bin overflow.
    # In Proceedings 2003 International Conference on Image Processing (Cat. No. 03CH37429), vol. 1, pp. I-881. IEEE, 2003.
    # Bin Underflow and Bin Overflow Histogram Equalization (BUBOHE)
    def BUBOHE(self, underflow, overflow):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        histogram, _ = np.histogram(image_1d, 256, [0, 255])
        histogram = histogram / len(image_1d)

        histogram[histogram > overflow] = overflow
        histogram[histogram < underflow] = underflow

        cdf = histogram.cumsum()
        LUT = np.array([0.0] * 256)

        for i in range(0, 256):
            LUT[i] = 255 * (cdf[i] - (cdf[i] / 255) * i) + i

        LUT = LUT.astype('uint8')
        return utils.LUT_image(LUT)

    # Wang, Chao, and Zhongfu Ye.
    # Brightness preserving histogram equalization with maximum entropy: a variational perspective.
    # IEEE Transactions on Consumer Electronics 51, no. 4 (2005): 1326-1334.
    # Brightness Preserving Histogram Equalization with Maximum Entropy (BPHEME)
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


    # Sim, K. S., C. P. Tso, and Y. Y. Tan. 
    # Recursive sub-image histogram equalization applied to gray scale images.
    # Pattern Recognition Letters 28, no. 10 (2007): 1209-1221.
    # Recursive Sub-Image Histogram Equalization (RSIHE)
    def RSIHE(self, recursive = 2):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        histogram = Histogram()
        LUT = histogram.histogram_equalization_recursively(image_1d, np.median, recursive)
        return utils.LUT_image(LUT)


    # Wang, Qing, and Rabab K. Ward. 
    # Fast image/video contrast enhancement based on weighted thresholded histogram equalization.
    # IEEE transactions on Consumer Electronics 53, no. 2 (2007): 757-764.
    # Weighted Thresholded Histogram Equalization (WTHE)
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


    # Kim, Mary, and Min Gyo Chung. 
    # Recursively separated and weighted histogram equalization for brightness preservation and contrast enhancement.
    # IEEE Transactions on Consumer Electronics 54, no. 3 (2008): 1389-1397.
    # Recursive Separated and Weighted Histogram Equalization (RSWHE)
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


    # Wang, C., J. Peng, and Z. Ye. 
    # Flattest histogram specification with accurate brightness preservation.
    # IET Image Processing 2, no. 5 (2008): 249-262.
    # Flattest Histogram Specification with Accurate Brightness Preservation (FHSABP)
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


    # Ooi, Chen Hee, Nicholas Sia Pik Kong, and Haidi Ibrahim.
    # Bi-histogram equalization with a plateau limit for digital image enhancement.
    # IEEE transactions on consumer electronics 55, no. 4 (2009): 2072-2080.
    # Bi-Histogram Equalization with a Plateau Limit (BHEPL)
    def BHEPL(self):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        mean = np.mean(image_1d)
        mean = math.floor(mean)

        lower_filter = image_1d <= mean
        lower_1d = image_1d[lower_filter]

        upper_filter = image_1d > mean
        upper_1d = image_1d[upper_filter]

        lower_histogram, _ = np.histogram(lower_1d, mean + 1, [0, mean])
        upper_histogram, _ = np.histogram(upper_1d, 255 - mean, [mean + 1, 255])

        lower_plateau_limit = np.sum(lower_histogram) / (mean + 1)
        upper_plateau_limit = np.sum(upper_histogram) / (255 - mean)

        lower_histogram[lower_histogram > lower_plateau_limit] = lower_plateau_limit
        upper_histogram[upper_histogram > upper_plateau_limit] = upper_plateau_limit

        lower_histogram_sum = np.sum(lower_histogram)
        upper_Histogram_sum = np.sum(upper_histogram)

        lower_cdf = lower_histogram.cumsum()
        lower_cdf_mask = np.ma.masked_equal(lower_cdf, 0)

        # Scale cdf to [range_min, range_max]
        lower_scale_cdf_mask = ((lower_cdf_mask - 0.5 * lower_histogram - lower_cdf_mask.min()) * (mean - 0) / (lower_cdf_mask.max() - lower_cdf_mask.min())) + 0
        lower_LUT = np.ma.filled(lower_scale_cdf_mask, 0).astype('uint8')

        upper_cdf = upper_histogram.cumsum()
        upper_cdf_mask = np.ma.masked_equal(upper_cdf, 0)

        # Scale cdf to [range_min, range_max]
        upper_scale_cdf_mask = ((upper_cdf_mask - 0.5 * upper_histogram - upper_cdf_mask.min()) * (255 - mean - 1) / (upper_cdf_mask.max() - upper_cdf_mask.min())) + mean + 1
        upper_LUT = np.ma.filled(upper_scale_cdf_mask, 0).astype('uint8')

        LUT = np.concatenate((lower_LUT, upper_LUT))

        return utils.LUT_image(LUT)


    # Zuo, Chao, Qian Chen, and Xiubao Sui. 
    # Range limited bi-histogram equalization for image contrast enhancement.
    # Optik 124, no. 5 (2013): 425-431.
    # Range Limited Bi-Histogram Equalization (RLBHE)
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


    # Wang, Zhi-Guo, Zhi-Hu Liang, and Chun-Liang Liu. 
    # A real-time image processor with combining dynamic contrast ratio enhancement and inverse gamma correction for PDP.
    # Displays 30, no. 3 (2009): 133-139.
    # Dynamic Contrast Ratio Gamma Correction (DCRGC)
    def DCRGC(self, contrast_intensity, gamma):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        gamma_reverse = [0.0] * 256
        for i in range(1, 256):
            gamma_reverse[i] = (i / 255) ** gamma

        normalized_foundation_histogram = np.array([0.0] * 256)
        for i in range(1, 256):
            normalized_foundation_histogram[i] = gamma_reverse[i] - gamma_reverse[i - 1]

        histogram, _ = np.histogram(image_1d, 256, [0, 255])
        histogram = histogram / len(image_1d)

        normalized_combination_histogram = histogram * contrast_intensity + normalized_foundation_histogram * (1 - contrast_intensity)
        normalized_combination_cdf = normalized_combination_histogram.cumsum()

        LUT = normalized_combination_cdf * 255
        LUT = LUT.astype('uint8')

        return utils.LUT_image(LUT)


    # Huang, Shih-Chia, Fan-Chieh Cheng, and Yi-Sheng Chiu. 
    # Efficient contrast enhancement using adaptive gamma correction with weighting distribution.
    # IEEE transactions on image processing 22, no. 3 (2012): 1032-1041.
    # Adaptive Gamma Correction with Weighting Distribution (AGCWD)
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

    # Rahman, Shanto, Md Mostafijur Rahman, Mohammad Abdullah-Al-Wadud, Golam Dastegir Al-Quaderi, and Mohammad Shoyaib.
    # An adaptive gamma correction for image enhancement.
    # EURASIP Journal on Image and Video Processing 2016, no. 1 (2016): 1-13.
    # Adaptive Gamma Correction Image Enhancement (AGCIE)
    def AGCIE(self, contrast_threshold = 3):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten() / 255

        mean = np.mean(image_1d)
        std = np.std(image_1d)
        LUT = np.arange(0, 256) / 255

        if std <= 1 / (4 * contrast_threshold):
            gamma = -math.log(std, 2)
        else:
            gamma = math.exp((1 - (mean + std))/2)

        if mean >= 0.5:
            LUT = 255 * (LUT ** gamma)
        else:
            for i in range(0, 256):
                LUT[i] = 255 * (LUT[i] ** gamma / (LUT[i] ** gamma + (1 - LUT[i] ** gamma) * mean ** gamma)) 

        LUT = LUT.astype('uint8')

        return utils.LUT_image(LUT)

    # Gupta, Bhupendra, and Mayank Tiwari. 
    # Minimum mean brightness error contrast enhancement of color images using adaptive gamma correction with color preserving framework.
    # Optik 127, no. 4 (2016): 1671-1676.
    # Adaptive Gamma Correction Color Preserving Framework (AGCCPF)
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
    ########################################
    #
    # Genetic Algorithm
    #
    ########################################

    # Saitoh, Fumihiko.
    # Image contrast enhancement using genetic algorithm.
    # In IEEE SMC'99 Conference Proceedings. 1999 IEEE International Conference on Systems, Man, and Cybernetics (Cat. No. 99CH37028), vol. 4, pp. 899-904. IEEE, 1999.
    # Saitoh Genetic Algorithm (SGA)

    # Hashemi, Sara, Soheila Kiani, Navid Noroozi, and Mohsen Ebrahimi Moghaddam.
    # An image contrast enhancement method based on genetic algorithm.
    # Pattern Recognition Letters 31, no. 13 (2010): 1816-1824.
    # Constrast Enhancement Based Genetic Algorithm (CEBGA)

    ########################################
    #
    # Other
    #
    ########################################


    # Raju, G., and Madhu S. Nair. 
    # A fast and efficient color image enhancement method based on fuzzy-logic and histogram.
    # AEU-International Journal of electronics and communications 68, no. 3 (2014): 237-243.
    # Fuzzy-Logic and Histogram (FLH)
    def FLH(self, enhancement):
        utils = Utils(self.image, self.color_space)
        image_gray = utils.image_gray()
        image_1d = image_gray.flatten()

        histogram, _ = np.histogram(image_1d, 256, [0, 255])
        sum_weight_histogram = 0
        for i in range(0, 256):
            sum_weight_histogram += i * histogram[i]

        control = sum_weight_histogram / np.sum(histogram)
        control = math.floor(control)

        lower_fuzzy = np.array([0.0] * (control))
        upper_fuzzy = np.array([0.0] * (256 - control))

        for i in range(0, control):
            lower_fuzzy[i] = i + (1 - control + i) * enhancement / control

        for i in range(control, 256):
            upper_fuzzy[i - control] = (255 - i) / (255 - control) + 255 - (255 - i) * enhancement / (255 - control)

        LUT = np.concatenate((lower_fuzzy, upper_fuzzy))
        LUT[LUT < 0] = 0
        LUT = LUT.astype('uint8')

        return utils.LUT_image(LUT)

    # Celik, Turgay, and Tardi Tjahjadi.
    # Contextual and variational contrast enhancement.
    # IEEE Transactions on Image Processing 20, no. 12 (2011): 3431-3441.
    # Contextual and Variational Contrast (CVC)
