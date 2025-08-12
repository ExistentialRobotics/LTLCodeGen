cmake_minimum_required(VERSION 3.0.2)
project(jackal_solar_sim VERSION 0.1.0)

find_package(catkin REQUIRED #
             COMPONENTS geometry_msgs roscpp rospy std_msgs tf2_ros sensor_msgs ros_ign_gazebo)

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS scripts/dynamic_tf_solar.py scripts/dynamic_tf.py #
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY models DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
