import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="sparseca",
    version="0.0.6",
    author="Nick Judd",
    author_email="nick@nickjudd.com",
    description="A package for correspondence analysis of large sparse networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nclarkjudd/sparseca",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    install_requires=['numpy >= 1.18.4',
                      'scipy >= 1.3.0',
                      'scikit-learn >= 0.21.2',
                     ],
    python_requires='>=3.6',
    )
