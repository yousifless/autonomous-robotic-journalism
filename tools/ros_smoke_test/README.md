# ROS 2 smoke test (no GPU needed)

Builds both packages in the official `ros:humble` image and checks what can be checked without the models:
rosdep keys, `colcon build`, installed executables and data files, the message definitions, both launch files,
that every module imports, the unit tests, and a live test of the `perception_controller` node with a stand-in for
VideoLLaMA3 (motion, stale-camera stop, pause/resume, zero velocity on SIGINT and SIGTERM).

```bash
# from the repository root
docker run --rm -v "$PWD:/repo:ro" ros:humble bash /repo/tools/ros_smoke_test/run_in_container.sh
```

The run ends with `RESULT: ALL PASSED`. It does not load the foundation models; test those on the GPU server.
