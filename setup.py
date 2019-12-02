# python setup.py bdist_wheel
#!/usr/bin/env python

from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['opencv-python',
                     'scikit-image',
                     "keras==2.0.8",
                     "tensorflow-gpu==1.14",
                     "tqdm"]

setup(name='semantic-segmentation',
      version='0.0.1',
      description='This Package handles toolsfor the image segmentation',
      author='Friedrich Muenke',
      author_email='f.muenke@vialytics.de',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=REQUIRED_PACKAGES,)
