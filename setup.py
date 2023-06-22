from setuptools import setup, find_packages
import os

if __name__ == '__main__':

    setup(
        classifiers=
        ["Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: OS Independent"],
        name='udl_vis',
        description="unified pytorch framework for vision task",
        long_description=open("README.md", encoding='utf-8').read(),
        long_description_content_type="text/markdown",
        author="XiaoXiao-Woo",
        author_email="wxwsx1997@gmail.com",
        url='https://github.com/XiaoXiao-Woo/PanCollection',
        version='0.3.5',
        packages=find_packages(),
        license='GPLv3',
        python_requires='>=3.7',
        install_requires=[
            "psutil",
            "opencv-python",
            "numpy",
            "matplotlib",
            "tensorboard",
            "addict",
            "yapf",
            "imageio",
            "colorlog",
            "scipy",
            "h5py",
            "regex",
            "packaging",
            "colorlog",
            "pyyaml"
        ],
    )