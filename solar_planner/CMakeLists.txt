cmake_minimum_required(VERSION 3.0.2)
project(solar_planner)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    rospy 
    std_msgs
    nav_msgs
    geometry_msgs 
    genmsg
    tf
    tf2
    tf2_ros
    tf2_geometry_msgs
)

# YAML-CPP
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# Boost
find_package(Boost COMPONENTS serialization iostreams system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Spot
include_directories(/usr/local/include/spot) # Include Spot headers

# Declare a catkin package
catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    std_msgs 
    nav_msgs 
    geometry_msgs 
    tf2
    tf2_ros
    tf2_geometry_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

# Define executables
add_executable(ltl_planner_node src/ltl_planner_node.cpp)


target_link_libraries(ltl_planner_node
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  spot
  bddx
  yaml-cpp
)


# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/tracker_node.py
  scripts/tracker_odom_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



