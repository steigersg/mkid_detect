from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mkid_detect",
    version="0.0.1",
    author="Sarah Steiger",
    author_email="ssteiger@stsci.edu",
    description="A package for simulating MKID outputs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)