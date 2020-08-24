from setuptools import setup, find_packages

setup(name='pydisk',
      description='Python astronomy tools',
      url='http://github.com/bjnorfolk/pymcfost',
      packages=find_packages(),
      install_requires=[line.rstrip() for line in open("requirements.txt", "r").readlines()],
      author='Brodie Norfolk',
      license='MIT',
      packages=['pydisk'],
      zip_safe=False)