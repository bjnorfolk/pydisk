#!/usr/bin/env python
# coding=utf-8
from setuptools import setup, find_packages

with open("pydisk/__init__.py", "r") as f:
      for line in f:
            if line.startswith('__version__'):
                  version = line.split('=')[1].strip().strip('"\'')

setup(name='pydisk',
      description='Python astronomy tools',
      version=version,
      url='http://github.com/bjnorfolk/pydisk',
      python_requires='>=3',
      include_package_data=True,
      packages=find_packages(),
      install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
      author='Brodie Norfolk',
      license='MIT',
      zip_safe=False)