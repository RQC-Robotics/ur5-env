from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req:
    all_requirements = req.readlines()

setup(
    name="ur-env",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.6,<3.10",
    package_data={"ur_env.robot": ["observations_schema.yaml"]},
    install_requires=["numpy", "gym"],
    extras_require={
        "all": all_requirements
    }
)
