from setuptools import setup, find_packages
 
setup(
  name='contrast-image',
  version='0.1.0',
  url='https://github.com/Nguyen-Hoang-Nam/contrast-image',
  license='MIT',
  author='Nguyen Hoang Nam',
  author_email='nguyenhoangnam.dev@gmail.com',
  description='🌈 Library to work with contrast',
  packages=find_packages(exclude=['test']),
  long_description=open('README.md').read(),
  zip_safe=False
)