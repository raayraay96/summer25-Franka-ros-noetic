# ROS Noetic + Python perception dependencies for development / CI.
# Does NOT provide real-time Franka hardware access.
FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    python3-yaml \
    python3-numpy \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf \
    ros-noetic-rosbag \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY requirements.txt /ws/requirements.txt
RUN pip3 install --no-cache-dir -r /ws/requirements.txt || true

COPY . /ws/src/summer25-Franka-ros-noetic
WORKDIR /ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    mkdir -p src && \
    ln -sfn /ws/src/summer25-Franka-ros-noetic/src/vision_arm_control /ws/src/vision_arm_control && \
    catkin_make -DCMAKE_BUILD_TYPE=Release || true"

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
