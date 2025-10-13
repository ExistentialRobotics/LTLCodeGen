cmake_minimum_required(VERSION 3.8)
project(solar_planner)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ament + ROS 2 deps
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Third-party deps
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization iostreams system filesystem)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

# Executable
add_executable(ltl_planner_node src/ltl_planner_node.cpp)

ament_target_dependencies(ltl_planner_node
        rclcpp
        std_msgs
        nav_msgs
        geometry_msgs
        tf2
        tf2_ros
        tf2_geometry_msgs
)

target_compile_definitions(ltl_planner_node PRIVATE JSON_DIAGNOSTICS=1)

target_include_directories(ltl_planner_node PRIVATE
        include
        ${EIGEN3_INCLUDE_DIRS}
        /usr/local/include/spot   # Spot headers
)

target_link_libraries(ltl_planner_node
        yaml-cpp
        nlohmann_json::nlohmann_json
        ${Boost_LIBRARIES}
        spot
        bddx
        Eigen3::Eigen
)

# Install C++ node
install(TARGETS ltl_planner_node
        DESTINATION lib/${PROJECT_NAME}
)

# (Optional) install launch files
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME}
)

ament_package()
