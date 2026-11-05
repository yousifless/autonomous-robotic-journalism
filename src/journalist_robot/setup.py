from glob import glob

from setuptools import find_packages, setup

package_name = "journalist_robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/missions", glob("missions/*.txt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ranulfo Bezerra",
    maintainer_email="ranulfo@tohoku.ac.jp",
    description="Newsworthiness-driven autonomous robotic journalism (ROS 2).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "perception_controller = journalist_robot.nodes.perception_controller_node:main",
            "capture_manager = journalist_robot.nodes.capture_manager_node:main",
            "journalist = journalist_robot.nodes.journalist_node:main",
            "decision_metrics = journalist_robot.analysis.decision_metrics:main",
        ],
    },
)
