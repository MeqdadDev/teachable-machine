from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="teachable_machine",
    version="1.2.1",
    description="A Python package designed to simplify the integration of exported models from Google's Teachable Machine platform into various environments. \
    This tool was specifically crafted to work seamlessly with Teachable Machine, making it easier to implement and use your trained models.",
    py_modules=["teachable_machine"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "Pillow",
        "tensorflow",
    ],
    python_requires=">=3.7",
    url="https://github.com/MeqdadDev/teachable-machine",
    author="Meqdad Dev",
    author_email="meqdad.darweesh@gmail.com",
)
