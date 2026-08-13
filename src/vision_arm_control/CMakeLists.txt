cmake_minimum_required(VERSION 3.0.2)
project(vision_arm_control)

## Do not hardcode a machine-specific Python path.
## Use the interpreter from the active environment / ROS.
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  tf
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    cv_bridge
    image_transport
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/camera_node.py
  scripts/pose_estimator_node.py
  scripts/depth_estimator_node.py
  scripts/human_to_robot_mapper_node.py
  scripts/safety_monitor_node.py
  scripts/telemetry_node.py
  scripts/robot_controller_node.py
  scripts/mock_landmark_publisher.py
  scripts/legacy/main.py
  scripts/legacy/monodepth2_node.py
  scripts/legacy/ros_camera_view.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
