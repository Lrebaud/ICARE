from setuptools import setup

setup(
    name='icare',
    version='0.0.1',
    description='icare model',
    url='https://github.com/Lrebaud/ICARE',
    author='Louis Rebaud',
    author_email='louis.rebaud@gmail.com',
    license='MIT',
    packages=['icare'],
    install_requires=[
        'numpy==1.19.5',
        'scikit-learn==0.23.2',
        'scikit-survival==0.14.0',
        'pandas==1.1.5',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
