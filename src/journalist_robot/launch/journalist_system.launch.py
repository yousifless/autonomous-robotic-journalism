"""Launch the GPU-side system: perception controller, capture manager and journalist.

Examples:
    ros2 launch journalist_robot journalist_system.launch.py mission:=research_lab_people_working.txt
    ros2 launch journalist_robot journalist_system.launch.py mission:=/abs/path/brief.txt rotate_180:=true
    ros2 launch journalist_robot journalist_system.launch.py start_journalist:=false   # controller only

Launch arguments left empty fall back to the values in the parameter file, so
camera topics, rotation, cmd_vel and output folder can be set either here or
in a copy of config/params.yaml passed with params_file:=. All other settings
are changed in that copy (see docs/configuration.md).
"""

import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEFAULT_SHUTDOWN_GRACE_S = 180.0


def _resolve_mission(path: str, share: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    candidate = os.path.join(share, "missions", path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Mission brief '{path}' not found (also looked in {os.path.dirname(candidate)})")


def _journalist_grace(params_file: str) -> float:
    """shutdown_grace_s of the journalist node, read from the parameter file."""
    try:
        import yaml
        with open(params_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key in ("journalist", "/journalist"):
            value = (data.get(key) or {}).get("ros__parameters", {}).get("shutdown_grace_s")
            if value is not None:
                return float(value)
    except (OSError, ValueError, AttributeError, ImportError):
        pass
    return DEFAULT_SHUTDOWN_GRACE_S


def _non_empty(**values) -> dict:
    """Only pass launch arguments the user actually set (others come from the parameter file)."""
    return {k: v for k, v in values.items() if v != ""}


def _nodes(context, *args, **kwargs):
    share = get_package_share_directory("journalist_robot")
    cfg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    params_file = cfg("params_file") or os.path.join(share, "config", "params.yaml")
    shared = {
        "mission_brief_file": _resolve_mission(cfg("mission"), share),
        "run_id": cfg("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S"),
        **_non_empty(output_root=cfg("output_root")),
    }
    rotate = cfg("rotate_180").lower()
    cameras = _non_empty(front_image_topic=cfg("front_image_topic"), image_transport=cfg("image_transport"))
    if rotate:
        cameras["rotate_180"] = rotate == "true"
    # ros2 launch escalates Ctrl-C to SIGTERM/SIGKILL after a few seconds; give the
    # journalist enough time to finish the article of an open segment.
    grace = _journalist_grace(params_file)

    return [
        Node(package="journalist_robot", executable="perception_controller", name="perception_controller",
             output="screen", condition=IfCondition(LaunchConfiguration("start_controller")),
             parameters=[params_file, shared, cameras, _non_empty(cmd_vel_topic=cfg("cmd_vel_topic"))]),
        Node(package="journalist_robot", executable="capture_manager", name="capture_manager",
             output="screen", condition=IfCondition(LaunchConfiguration("start_capture")),
             parameters=[params_file, {"mission_brief_file": shared["mission_brief_file"]}, cameras,
                         _non_empty(rear_image_topic=cfg("rear_image_topic"))]),
        Node(package="journalist_robot", executable="journalist", name="journalist",
             output="screen", condition=IfCondition(LaunchConfiguration("start_journalist")),
             sigterm_timeout=str(grace + 15.0), sigkill_timeout="10.0",
             parameters=[params_file, {k: v for k, v in shared.items() if k != "mission_brief_file"}]),
    ]


def generate_launch_description() -> LaunchDescription:
    args = [
        ("mission", "research_lab_people_working.txt",
         "Mission brief: a file name in share/journalist_robot/missions or a path"),
        ("params_file", "", "Parameter YAML (default: share/journalist_robot/config/params.yaml)"),
        ("output_root", "", "Folder that receives one sub-folder per run (default: from params_file)"),
        ("run_id", "", "Run folder name (default: current date and time)"),
        ("front_image_topic", "", "Front camera topic (default: from params_file)"),
        ("rear_image_topic", "", "Rear camera topic (default: from params_file)"),
        ("image_transport", "", "compressed | raw (default: from params_file)"),
        ("rotate_180", "", "true | false: rotate camera images by 180 degrees (default: from params_file)"),
        ("cmd_vel_topic", "", "Velocity command topic of the mobile base (default: from params_file)"),
        ("start_controller", "true", "Start the perception controller"),
        ("start_capture", "true", "Start the capture manager"),
        ("start_journalist", "true", "Start the journalist pipeline"),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
        + [OpaqueFunction(function=_nodes)]
    )
