import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="sparseca",
    version="0.0.6",
    author="Nick Judd",
    author_email="ncj@uchicago.edu",
    description="A package for correspondence analysis of large sparse networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://package-url-goes-here,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        ],
    install_requires=['numpy >= 1.18.4',
                      'scipy >= 1.3.0',
                      'scikit-learn >= 0.21.2',
                     ],
    python_requires='>=3.6',
    )
