import os
import re

from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req:
    all_requirements = req.readlines()


def get_version():
    path = os.path.join(os.path.dirname(__file__), "ur_env/__init__.py")
    with open(path) as f:
        version_match = re.search(r"__version__.*(\d+.\d+.\d+)", f.read())
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to parse version string.')


setup(
    name="ur-env",
    version=get_version(),
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["dm-env"],
    extras_require={
        "all": all_requirements
    }
)
