from setuptools import setup, find_packages
 
setup(
  name='contrast_image',
  version='0.1.6',
  url='https://github.com/Nguyen-Hoang-Nam/contrast-image',
  license='MIT',
  author='Nguyen Hoang Nam',
  author_email='nguyenhoangnam.dev@gmail.com',
  description='🌈 Library to work with contrast',
  packages=find_packages(exclude=['tests']),
  long_description=open('README.md').read(),
  long_description_content_type="text/markdown",
  install_requires=["numpy", "opencv-python"],
  zip_safe=False
)