import re
from pathlib import Path

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


def _read_version():
    """
    Read `__version__` from src/__init__.py so the package version has a
    single source of truth instead of being duplicated (and drifting out
    of sync, as it previously did) between here and there.
    """
    init_contents = (Path(__file__).parent / "src" / "__init__.py").read_text()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_contents, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in src/__init__.py")
    return match.group(1)


setup(
    name="teachable_machine",
    version=_read_version(),
    description="A Python package designed to simplify the integration of exported models from Google's Teachable Machine platform into various environments. \
    This tool was specifically crafted to work seamlessly with Teachable Machine, making it easier to implement and use your trained models.",
    py_modules=["teachable_machine"],
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
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
        "tensorflow>=2.16",
    ],
    python_requires=">=3.9",
    url="https://github.com/MeqdadDev/teachable-machine",
    author="Meqdad Dev",
    author_email="meqdad.darweesh@gmail.com",
)
