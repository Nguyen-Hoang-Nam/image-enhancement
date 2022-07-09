from setuptools import setup, find_packages
 
setup(
  name='image_enhancement',
  version='0.2.1',
  url='https://github.com/Nguyen-Hoang-Nam/image-enhancement',
  license='MIT',
  author='Nguyen Hoang Nam',
  author_email='nguyenhoangnam.dev@gmail.com',
  description='🌈 Library to enhance image',
  packages=find_packages(exclude=['tests', 'images']),
  long_description=open('README.md').read(),
  long_description_content_type="text/markdown",
  install_requires=["numpy", "opencv-python"],
  zip_safe=False
)
