import os
import distutils.extension
from setuptools import setup
import setuptools.extension
from Cython.Build import cythonize
import numpy

# ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
# with open("README.md", "r", encoding="utf-8") as fh:
#    long_description = fh.read()

global_alignment = setuptools.extension.Extension(
    "tsvdd.ga",
    sources=["tsvdd/ga.pyx", "src/ga/logGAK.c", "src/ga/matrixLogGAK.c"],
    include_dirs=[numpy.get_include(), 'src/ga/'],
    extra_compile_args=['-fopenmp', '-O3'],
    extra_link_args=['-fopenmp'])

libsvdd = setuptools.extension.Extension(
    'tsvdd.libsvdd',
    sources=["tsvdd/libsvdd.pyx", "src/libsvm/svm.cpp"],
    include_dirs=[numpy.get_include(), 'src/libsvm/'],
    extra_compile_args=['-fopenmp', '-O3', '-std=c++11'],
    extra_link_args=['-fopenmp',  '-lstdc++'])

setup(
    name='tsvdd',
    packages=['tsvdd'],
    version='0.0.1',
    author="Haiko Thiessen",
    author_email="haikothiessen@gmail.com",
    url="https://github.com/thiessenh/tsvdd",
    license='Apache 2.0',
    long_description='TODO',
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",],
    description= 'SVDD for time-series data based on libsvm.',
    extras_require={
        'dev': [
            'pytest',
            'pytest-pep8',
            'pytest-cov']},
    include_package_data=True,
    python_requires=">=3.6",
    ext_modules=cythonize([global_alignment, libsvdd]))
