from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='icare',
    version='0.0.3',
    description='ICARE models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Lrebaud/ICARE',
    author='Louis Rebaud',
    author_email='louis.rebaud@gmail.com',
    license="MIT",
    packages=['icare'],
    install_requires=[
        'seaborn',
        'scikit-survival',
    ],
    tests_require=['pytest'],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
