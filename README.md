# Contrast-image

Base on multiple papers about contrast, I create this library to contrast images in opencv

## Usage

```python
import contrast-image as ci
```

## API

### GHE (Global Histogram Equalization)

This function is similar to ```equalizeHist(image)``` in opencv.

```python
ci.GHE(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### BBHE (Brightness Preserving Histogram Equalization)

This function separate the histogram by the mean of the image, then equalize histogram of each part.

This method tries to preserve brightness of output image by assume PDF is symmetrical distribution.

```python
ci.BBHE(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### DSIHE (Dualistic Sub-Image Histogram Equalization)

This function is similar to BBHE except using median instead of mean.

Unlike BBHE, DSIHE tries to preserve brightness of output image by maximum entropy after separate.

```python
ci.DSIHE(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### MMBEBHE (Minimum Mean Brightness Error Histogram Equalization)

This function is similar to BBHE except using minimum mean brightness error instead of mean.

Theortically, mean of output image (by GHE) is middle gray level. Therefore, MMBEBHE believe by separate histogram such that mean of output image near mean of input image must preserve brightness.

```python
ci.MMBEBHE(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### BPHEME (Brightness Preserving Histogram Equalization with Maximum Entropy)

This function finds matching function such that make output image maximum entropy, then using histogram specification to match input's histogram and matching function.

Based on idea of DSIHE, BPHEME tries to generalize by using histogram specification and solve optimize problem by Lagrange interpolation.

```python
ci.BPHEME(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### RLBHE (Range Limited Bi-Histogram Equalization)

This function is similar to BBHE except using otsu's method instead of mean. Moreover, this limit range of gray level such that output image has minimum mean brightness error.

This method tries to equalize histogram for foreground and background separately by Otsu's method.

```python
ci.RLBHE(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### RMSHE (Recursively Mean-Separate Histogram Equalization)

This function recursively separate histogram by mean. Therefore, ```recursive = 2``` will create 4 sub-histograms, then equalize each sub-histograms.

Same idea as BBHE but recursively separate to preserve more brightness.

```python
ci.RMSHE(image, recursive = 2)
```

- Parameter image: image that read by opencv
- Parameter recurive: number of recursive time
- Return: image after equalization

### RSIHE (Recursive Sub-Image Histogram Equalization)

This function is similar to RMSHE except using median instead of mean.

Same idea as DSIHE but recursively separate to preserve more brightness.

```python
ci.RSIHE(image, recursive = 2)
```

- Parameter image: image that read by opencv
- Parameter recurive: number of recursive time
- Return: image after equalization

### RSWHE (Recursive Separated and Weighted Histogram Equalization)

This function recursively separate histogram by mean or median, then weighting each sub-histogram before equalize them.

This method similar to RMSHE and RSIHE except weighting sub-histogram to avoid local extreme value in histogram.

```python
ci.RSWHE(image, type = 'mean', beta = 0, recursive = 2)
```

- Parameter image: image that read by opencv
- Parameter type: 'mean' or 'median'
- Parameter beta: increasing more brightness in output image
- Parameter recurive: number of recursive time
- Return: image after equalization

### FHSABP (Flattest Histogram Specification with Accurate Brightness Preservation)

This function finds matching function such that make the flattest output's histogram, then using histogram specification to match input's histogram and matching function.

Because of discrete, histogram equalization does not often the flattest histogram. FHSABP tries to solve optimization function to find the flattest output's histogram.

```python
ci.FHSABP(image)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### WTHE (Weighted Thresholded Histogram Equalization)

This function weight histogram before equalize it.

```python
ci.WTHE(image, root, value, lower = 0)
```

- Parameter image: image that read by opencv
- Return: image after equalization

### AGCWD (Adaptive Gamma Correction with Weighting Distribution)

This function automatic correct gamma using weighting distribution

```python
ci.AGCWD(image, alpha)
```

- Parameter image: image that read by opencv
- Parameter alpha: adjustment
- Return: image after equalization

## Quantitation

### AMBE (Absolute Mean Brightness Error)

```python
ci.AMBE(input_image, output_image)
```

### PSNR (Peak Signal to Noise Ratio)

```python
ci.PSNR(input_image, output_image)
```

### Entropy

```python
ci.Entropy(image)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)