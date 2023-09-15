from setuptools import setup, find_packages

setup(
    name="inquire",
    version="0.1",
    description="A Visual Gui Testing Modul for low resolution displays for the robot frame work",
    author="Widukind Stöter",
    author_email="w.stoeter@gmx.de",
    maintainer="",
    maintainer_email="",
    url="",
    download_url="",
    packages=find_packages(where="src", include=["inquire.*"]),
    package_dir={"": "src"}
)
