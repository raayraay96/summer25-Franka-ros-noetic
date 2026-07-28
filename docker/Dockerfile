# Compatibility Dockerfile. Build with repository root as context.
FROM ros:noetic-ros-base-focal
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 ROS_MASTER_URI=http://localhost:11311
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential python3-pip python3-venv python3-catkin-pkg python3-rosdep \
      ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-tf ros-noetic-rosbag \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/franka-teleop
COPY . /workspaces/franka-teleop
RUN python3 -m venv --system-site-packages /opt/franka-venv
ENV PATH="/opt/franka-venv/bin:${PATH}"
RUN python -m pip install --no-cache-dir --upgrade "pip==23.3.2" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pytest -q tests/ \
    && mkdir -p /ws/src \
    && ln -s /workspaces/franka-teleop/src/vision_arm_control /ws/src/vision_arm_control \
    && source /opt/ros/noetic/setup.bash \
    && cd /ws \
    && catkin_make -DCMAKE_BUILD_TYPE=Release
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
