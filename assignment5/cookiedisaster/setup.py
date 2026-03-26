from setuptools import setup

setup(
    name="cookiedisaster",
    version="0.0.1",
    packages=["cookiedisaster", "cookiedisaster.envs"],
    package_dir={
        "cookiedisaster": ".",
        "cookiedisaster.envs": "envs",
    },
    package_data={"cookiedisaster.envs": ["cookie.jpg", "cookie.png"]},
    include_package_data=True,
    install_requires=["gymnasium", "pygame"],
)
