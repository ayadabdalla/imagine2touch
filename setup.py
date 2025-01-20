import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="imagine2touch",
    version="1.0.0",
    author="Abdalla Ayad",
    author_email="abdallahayman772@gmail.com",
    description="A touch sense with unknown object recognition demo based on reskin by Raunaq Bhirangi",
    long_description=read("README.md"),
    packages=find_packages(),
    include_package_data=True,
    url="https://rlgit.informatik.uni-freiburg.de/touch/touch2image",
)
