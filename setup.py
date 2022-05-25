import os

import setuptools


setuptools.setup(
    name="better_einsum",
    version="1.0.0",
    description="np.einsum but better",
    url="https://github.com/evhub/better_einsum",
    author="Evan Hubinger",
    author_email="evanjhub@gmail.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "pyparsing",
        "numpy",
    ],
)
