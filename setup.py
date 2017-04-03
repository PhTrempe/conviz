from distutils.core import setup

from setuptools import find_packages

setup(
    name="conviz",
    version="0.1",
    description="A convolutional neural layer visualization library",
    author="Philippe Trempe",
    author_email="ph.trempe@gmail.com",
    url="https://github.com/PhTrempe/conviz",
    download_url="https://github.com/phtrempe/conviz/archive/0.1.tar.gz",
    keywords=["convolutional", "neural", "layer", "visualization"],
    classifiers=[],
    packages=find_packages(),
    requires=["numpy", "scipy", "pillow", "keras"]
)
