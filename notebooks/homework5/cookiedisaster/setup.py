from setuptools import setup

setup(
    name="cookiedisaster",
    version="0.0.1",
    packages=["cookiedisaster", "cookiedisaster.envs"],
    package_dir={"cookiedisaster": ".", "cookiedisaster.envs": "envs"},
    package_data={"cookiedisaster.envs": ["cookie.png"]},
    install_requires=["gym==0.26.0"],
)