from setuptools import setup, find_packages

setup(
    name="TabMixer",  # package name on PyPI
    version="0.1.0",
    description="A TabMixer module for mixing tabular data using MLP blocks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/TabMixer",  # update with your repo URL
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",  # specify the minimum version required
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
