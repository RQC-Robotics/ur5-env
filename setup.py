from setuptools import setup, find_packages


setup(
    name="ur_env",
    version="0.0.1",
    packages=find_packages(
        "ur_env",
        include=["third_party"]
    ),
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True
)
