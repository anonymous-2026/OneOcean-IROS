from glob import glob

from setuptools import find_packages, setup


package_name = "oneocean_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anonymous Authors",
    maintainer_email="anonymous@example.com",
    description="Optional ROS 2 bridge for the OneOcean benchmark core.",
    license="MIT",
    entry_points={"console_scripts": ["oneocean_bridge = oneocean_ros.bridge_node:main"]},
)
