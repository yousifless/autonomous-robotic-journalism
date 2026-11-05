#!/bin/bash
# Builds the repository in ros:humble and runs the checks that do not need a GPU.
# Usage (from the repository root):
#   docker run --rm -v "$PWD:/repo:ro" ros:humble bash /repo/tools/ros_smoke_test/run_in_container.sh
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null
apt-get install -y -qq python3-pytest python3-opencv python3-colcon-common-extensions python3-yaml \
    ros-humble-v4l2-camera >/dev/null
source /opt/ros/humble/setup.bash

echo "=== rosdep check of package.xml keys ==="
rosdep init >/dev/null 2>&1 || true
rosdep update -q >/dev/null 2>&1 || true
mkdir -p /ws/src && cp -r /repo/src/. /ws/src/
cd /ws
rosdep check --from-paths src --ignore-src -y 2>&1 | tail -3 || true
rosdep keys --from-paths src --ignore-src | sort | xargs -I{} sh -c 'rosdep resolve {} >/dev/null 2>&1 && echo "  ok  {}" || echo "  UNRESOLVED {}"'

echo "=== colcon build ==="
colcon build --event-handlers console_cohesion+ 2>&1 | tail -5
source install/setup.bash

echo "=== installed executables and data ==="
ros2 pkg executables journalist_robot
ls install/journalist_robot/share/journalist_robot/{launch,config,missions}

echo "=== interfaces ==="
ros2 interface show journalist_robot_interfaces/msg/SegmentEvent

echo "=== launch file: arguments ==="
ros2 launch journalist_robot journalist_system.launch.py -s
echo "=== launch file: robot sensors ==="
ros2 launch journalist_robot robot_sensors.launch.py -s

echo "=== import every module ==="
python3 - <<'EOF'
import importlib, pkgutil, journalist_robot
bad = []
for m in pkgutil.walk_packages(journalist_robot.__path__, "journalist_robot."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        bad.append((m.name, repr(e)))
print("modules with import errors:", bad or "none")
EOF

echo "=== unit tests ==="
cd /ws/src/journalist_robot && python3 -m pytest -q test 2>&1 | tail -4
cd /ws

echo "=== ROS smoke test ==="
python3 /repo/tools/ros_smoke_test/smoke.py
