from setuptools import setup, find_packages

setup(name='pydisk',
      description='Python astronomy tools',
      version='0.0.0',
      url='http://github.com/bjnorfolk/pydisk',
      python_requires='>=3',
      packages=find_packages(),
      install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
      author='Brodie Norfolk',
      license='MIT',
      zip_safe=False)