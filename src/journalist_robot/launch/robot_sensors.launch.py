"""Example robot-side launch: front and rear USB cameras with v4l2_camera.

Run this on the robot's on-board computer (e.g. Jetson AGX Orin). It is an
example; any camera driver that publishes sensor_msgs/Image (and, with
image_transport plugins, .../compressed) works. Your mobile base driver must
subscribe to /cmd_vel; start it with the base's own launch file.

    ros2 launch journalist_robot robot_sensors.launch.py front_device:=/dev/video0 rear_device:=/dev/video2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera(name: str, device_arg: str) -> Node:
    return Node(
        package="v4l2_camera", executable="v4l2_camera_node", name=name, namespace=name,
        output="screen",
        parameters=[{
            "video_device": LaunchConfiguration(device_arg),
            "image_size": [640, 480],
        }],
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("front_device", default_value="/dev/video0"),
        DeclareLaunchArgument("rear_device", default_value="/dev/video2"),
        _camera("front_camera", "front_device"),
        _camera("rear_camera", "rear_device"),
    ])
