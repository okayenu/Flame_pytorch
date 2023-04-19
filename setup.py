import codecs
import os

from setuptools import find_packages, setup

# Basic information
NAME = "FLAME_PyTorch"
DESCRIPTION = "PyTorch implementation of the 3D FLAME model."
VERSION = "0.0.1"
AUTHOR = "Soubhik Sanyal"
EMAIL = "soubhik.sanyal@tuebingen.mpg.de"
LICENSE = "See LICENSE"
REPOSITORY = "https://github.com/soubhiksanyal/FLAME_PyTorch"
PACKAGE = "flame_pytorch"

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the keywords
KEYWORDS = ["FLAME", "3D Face Modelling"]

CLASSIFIERS = [
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
]

