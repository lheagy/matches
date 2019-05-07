#!/usr/bin/env python
"""matches

ML + Physics with pytorch
"""

from distutils.core import setup
from setuptools import find_packages


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.rst") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="matches",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
    ],
    author="Lindsey Heagy",
    author_email="lindseyheagy@gmail.com",
    description="matches",
    long_description=LONG_DESCRIPTION,
    keywords="machine learning, physics",
    url="https://github.com/lheagy/matches",
    download_url="https://github.com/lheagy/matches",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="BSD",
    use_2to3=False,
)

