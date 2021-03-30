# Contrast-image

Base on multiple papers about contrast, I create this library to contrast images in opencv

## Usage

```python
import contrast-image as ci
```

### GHE

```python
ci.GHE(image)
```

This function similar with ```cv.equalizeHist(image)```

### BBHE (Brightness Preserving Histogram Equalization)

```python
ci.BBHE(image)
```

This function separate histogram by mean and histogram equalization each part

### DSIHE (Dualistic Sub-Image Histogram Equalization)

```python
ci.DSIHE(image)
```

This function separate histogram by median and histogram equalization each part

### MMBEBHE (Minimum Mean Brightness Error Histogram Equalization)

```python
ci.MMBEBHE(image)
```

This function calculate minimum mean brightness error and separate histogram by min then equalizate each part

### RMSHE (Recursively Mean-Separate Histogram Equalization)

```python
ci.RMSHE(image, recursive = 2)
```

This function recursively separate histogram by mean and histogram equalization each part

### RSIHE (Recursive Sub-Image Histogram Equalization)

```python
ci.RSIHE(image, recursive = 2)
```

This function recursively separate histogram by median and histogram equalization each part

### BPHEME (Brightness Preserving Histogram Equalization with Maximum Entropy)

```python
ci.BPHEME(image)
```

This function calculate maximum entropy and separate histogram by max then equalize each part

### RLBHE (Range Limited Bi-Histogram Equalization)

```python
ci.RLBHE(image)
```

This function separate histogram by otsu's method and limit minimum and maximum gray level befor equalize each part

### RSWHE (Recursive Separated and Weighted Histogram Equalization)

```python
ci.RSWHE(image, type = 'mean', beta = 0, recursive = 2)
```

This function recursively separate histogram by type (mean or median) then calculate weight befor equalize each part