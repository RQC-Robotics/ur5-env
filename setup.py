from setuptools import setup, find_packages


setup(
    name="ur-env",
    version="0.0.1",
    packages=find_packages(),
    package_data={"ur_env.robot": ["observations_schema.yaml"]},
    install_requires=open("requirements.txt", encoding="utf-8").readlines(),
)
