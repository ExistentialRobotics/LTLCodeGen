#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Quaternion.h>

#include <spot/parseaut/public.hh>
#include <spot/twa/bdddict.hh>

#include "solar_planner/grid_map.h"
#include "solar_planner/environments/planning_spot_2d.h"
#include "solar_planner/astar_nx.h"

#include <tf/transform_listener.h>

#include <cmath>

std::vector<geometry_msgs::Pose> computePath(const spot::twa_graph_ptr &automaton,
                                             const nav_msgs::Odometry &our_pose,
                                             nav_msgs::OccupancyGrid occ_map,
                                             nav_msgs::OccupancyGrid label_map)
{
    // TODO: Get map data
    const std::vector<double> &start = {our_pose.pose.pose.position.x, our_pose.pose.pose.position.y};
    double eps = 1.0;
    int width = occ_map.info.width;
    int height = occ_map.info.height;
    const std::vector<double> &mapres = {occ_map.info.resolution, occ_map.info.resolution};
    const std::vector<double> &mapmin = {occ_map.info.origin.position.y, occ_map.info.origin.position.x};
    const std::vector<double> &mapmax = {occ_map.info.origin.position.y + occ_map.info.resolution * occ_map.info.height,
                                         occ_map.info.origin.position.x + occ_map.info.resolution * occ_map.info.width};
    const std::vector<int> &mapdim = {height, width};

    std::cout << "Start: \n";
    for (double value : start)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl
              << std::endl;

    std::cout << "Epsilon: " << eps << std::endl
              << std::endl;

    std::cout << "mapmin: \n";
    for (double value : mapmin)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl
              << std::endl;

    std::cout << "mapmax: \n";
    for (double value : mapmax)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl
              << std::endl;

    std::cout << "mapres: \n";
    for (double value : mapres)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl
              << std::endl;

    std::cout << "mapdim: \n";
    for (int value : mapdim)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl
              << std::endl;

    //  // Initialize MAP
    std::cout << "Initializing Label Map..." << std::endl;
    std::unique_ptr<erl::GridMap<uint16_t>> MAP_ptr(new erl::GridMap<uint16_t>(mapmin, mapmax, mapres));

    //  Read label map 
    std::cout << "Reading Label Map Content:\n";
    Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> lmap(MAP_ptr->size()[0], MAP_ptr->size()[1]);

    for (int y = 0; y < MAP_ptr->size()[0]; ++y)
    {

        for (int x = 0; x < MAP_ptr->size()[1]; ++x)
        {
            // Map's data is stored in row-major order
            int index = y * width + x;
            int value = label_map.data[index];
            // Convert the map value to uint16_t (e.g., -1 for unknown becomes 0)
            lmap(y, x) = (value < 0) ? 0 : static_cast<uint16_t>(value);
        }
    }

    // Read occupancy map
    std::cout << "Reading Occupancy Map Content:\n";
    Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> omap(MAP_ptr->size()[0], MAP_ptr->size()[1]);

    for (int y = 0; y < MAP_ptr->size()[0]; ++y)
    {

        for (int x = 0; x < MAP_ptr->size()[1]; ++x)
        {
            // Map's data is stored in row-major order
            int index = y * width + x;
            int value = occ_map.data[index];

            // Convert the map value to uint16_t (e.g., -1 for unknown becomes 0)
            omap(y, x) = (value < 0) ? 0 : static_cast<uint16_t>(value); // Doubtful
        }
    }

    std::cout << "Initializing Environment..." << std::endl;
    erl::PlanningSpot2D ENV(lmap, omap, std::move(MAP_ptr), automaton);

    // Initialize planner
    std::array<int, 3> start_coord;
    start_coord[0] = erl::meters2cells(start[0], mapmin[0], mapres[0]);
    start_coord[1] = erl::meters2cells(start[1], mapmin[1], mapres[1]);
    start_coord[2] = automaton->get_init_state_number(); 
    erl::ARAStar<std::array<int, 3>> AA;

    // Plan path
    std::cout << "Starting computation..." << std::endl;
    auto t1 = erl::tic();
    auto output = AA.Astar(start_coord, ENV, eps);
    std::cout << "Computation done in " << erl::toc(t1) << " sec!" << std::endl;

    std::cout << "Plan cost = " << output.pcost << std::endl;
    std::cout << "Path length = " << output.path.size() << std::endl;
    std::cout << "action_idx.size() = " << output.action_idx.size() << std::endl;

    // Convert grid coordinates to poses
    std::vector<geometry_msgs::Pose> our_poses;
    for (auto &coord : output.path)
    {
        geometry_msgs::Pose pose;
        pose.position.x = erl::cells2meters(coord[0], mapmin[0], mapres[0]);
        pose.position.y = erl::cells2meters(coord[1], mapmin[1], mapres[1]);
        pose.position.z = 0.0;
        pose.orientation.w = 1.0;
        our_poses.push_back(pose);
    }

    return our_poses;
}

class LTLPlannerNode
{
public:
    LTLPlannerNode()
    {

        ROS_INFO("Setting up Planning!");
        // Occupancy map subscriber
        occ_map_sub_ = nh_.subscribe("/occupancy_map_2D", 1, &LTLPlannerNode::occMapCallback, this);

        // Label map subscriber
        label_map_sub_ = nh_.subscribe("/label_map", 1, &LTLPlannerNode::labelMapCallback, this);

        // Automaton subscriber
        automaton_sub_ = nh_.subscribe("/aut_str", 1, &LTLPlannerNode::automatonCallback, this);

        // Path publisher
        path_pub_ = nh_.advertise<nav_msgs::Path>("computed_path", 1);

        // Get the world frame id
        world_frame_id = "world";

        // Get the robot frame id
        robot_frame_id = "husky_1/base_link";
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber occ_map_sub_;
    ros::Subscriber label_map_sub_;
    ros::Subscriber automaton_sub_;
    ros::Publisher path_pub_;

    // These are modified for appropriate dimensions
    nav_msgs::OccupancyGrid current_occ_map_;
    nav_msgs::OccupancyGrid current_label_map_;

    // these are used to check if map is updated
    nav_msgs::OccupancyGrid none_modified_occ_map_;
    nav_msgs::OccupancyGrid none_modified_label_map_;

    nav_msgs::Odometry current_pose_;

    tf::TransformListener tf_listener;

    std::string world_frame_id;
    std::string robot_frame_id;

    spot::twa_graph_ptr automaton_;

    bool occ_map_received_ = false;
    bool label_map_received_ = false;
    bool automaton_received_ = false;
    bool pose_received_ = false;

    // Function to get nav_msgs::Odometry from TF
    nav_msgs::Odometry getOdomFromTF(const std::string &robot_frame_id)
    {
        nav_msgs::Odometry odom_msg;
        tf::StampedTransform transform;

        try
        {
            tf_listener.lookupTransform(world_frame_id, robot_frame_id, ros::Time(0), transform);
        }
        catch (tf::TransformException &ex)
        {
            ROS_ERROR("%s", ex.what());
            return odom_msg; // Return empty odometry if lookup fails
        }

        // Fill out the nav_msgs::Odometry message
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = world_frame_id;
        odom_msg.child_frame_id = robot_frame_id;

        // Set the position (translation)
        odom_msg.pose.pose.position.x = transform.getOrigin().x();
        odom_msg.pose.pose.position.y = transform.getOrigin().y();
        odom_msg.pose.pose.position.z = transform.getOrigin().z();

        // Set the orientation (rotation)
        odom_msg.pose.pose.orientation.x = transform.getRotation().x();
        odom_msg.pose.pose.orientation.y = transform.getRotation().y();
        odom_msg.pose.pose.orientation.z = transform.getRotation().z();
        odom_msg.pose.pose.orientation.w = transform.getRotation().w();

        pose_received_ = true;

        return odom_msg;
    }

    void occMapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        // Check if the map has already been received and if it matches the current map
        if (occ_map_received_ &&
            msg->info.width == none_modified_occ_map_.info.width &&
            msg->info.height == none_modified_occ_map_.info.height &&
            msg->info.resolution == none_modified_occ_map_.info.resolution &&
            msg->info.origin == none_modified_occ_map_.info.origin &&
            msg->data == none_modified_occ_map_.data)
        {
            return;
        }

        none_modified_occ_map_ = *msg;

        // Print the received map info
        ROS_INFO("Got a new occ map!");

        // Copy the received map to modify it
        nav_msgs::OccupancyGrid modified_map = *msg;

        // Check if width is even, if so, add a column of obstacles
        if (modified_map.info.width % 2 == 0)
        {
            ROS_INFO("Width is even. Adding a column of obstacles...");
            // Add a column of obstacles (value 100)
            for (int i = 0; i < modified_map.info.height; ++i)
            {
                modified_map.data.insert(modified_map.data.begin() + (i + 1) * modified_map.info.width + i, 100);
            }
            modified_map.info.width += 1; // Update the width
        }

        // Check if height is even, if so, add a row of obstacles
        if (modified_map.info.height % 2 == 0)
        {
            ROS_INFO("Height is even. Adding a row of obstacles...");
            // Add a row of obstacles (value 100)
            for (int i = 0; i < modified_map.info.width; ++i)
            {
                modified_map.data.push_back(100); // obstacles
            }
            modified_map.info.height += 1; // Update the height
        }

        // Now assign the modified map as the current map
        current_occ_map_ = modified_map;
        occ_map_received_ = true;
        ROS_INFO("Occ map modified and updated.");

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void labelMapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        // Check if the map has already been received and if it matches the current map
        if (label_map_received_ &&
            msg->info.width == none_modified_label_map_.info.width &&
            msg->info.height == none_modified_label_map_.info.height &&
            msg->info.resolution == none_modified_label_map_.info.resolution &&
            msg->info.origin == none_modified_label_map_.info.origin &&
            msg->data == none_modified_label_map_.data)
        {
            return;
        }

        none_modified_label_map_ = *msg;

        // Print the received map info
        ROS_INFO("Got a new map!");

        // Copy the received map to modify it
        nav_msgs::OccupancyGrid modified_map = *msg;

        // Check if width is even, if so, add a column of obstacles
        if (modified_map.info.width % 2 == 0)
        {
            ROS_INFO("Width is even. Adding a column of obstacles...");
            // Add a column of obstacles (value 100)
            for (int i = 0; i < modified_map.info.height; ++i)
            {
                modified_map.data.insert(modified_map.data.begin() + (i + 1) * modified_map.info.width + i, 0);
            }
            modified_map.info.width += 1; // Update the width
        }

        // Check if height is even, if so, add a row of obstacles
        if (modified_map.info.height % 2 == 0)
        {
            ROS_INFO("Height is even. Adding a row of obstacles...");
            // Add a row of obstacles (value 100)
            for (int i = 0; i < modified_map.info.width; ++i)
            {
                modified_map.data.push_back(0);
            }
            modified_map.info.height += 1; // Update the height
        }

        // Now assign the modified map as the current map
        current_label_map_ = modified_map;
        label_map_received_ = true;
        ROS_INFO("Label map modified and updated.");

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void automatonCallback(const std_msgs::String::ConstPtr &msg)
    {
        if (automaton_received_)
        {
            return;
        }

        spot::automaton_stream_parser *str_parser = new spot::automaton_stream_parser(msg->data.c_str(), "ROS_Str");
        spot::parsed_aut_ptr pa = str_parser->parse(spot::make_bdd_dict()); 

        if (pa->format_errors(std::cerr))
            return;
        // This cannot occur when reading a never claim, but
        // it could while reading a HOA file.
        if (pa->aborted)
        {
            std::cerr << "--ABORT-- read\n";
            ROS_INFO("Automaton aborted.");
            return;
        }

        automaton_ = pa->aut;
        automaton_received_ = true;

        ROS_INFO("Aut Type: [%d]", (int)pa->type); // 0 is HOA format

        delete str_parser;

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void attemptPathPlanning()
    {
        current_pose_ = getOdomFromTF(robot_frame_id);

        if (automaton_received_ && occ_map_received_ && label_map_received_ && pose_received_)
        {
            ROS_INFO("Attempting to compute path...");
            ////////////////////////////////////////////////
            // Get the current robot pose from TF
            current_pose_ = getOdomFromTF(robot_frame_id);
            double temp = current_pose_.pose.pose.position.x;
            current_pose_.pose.pose.position.x = current_pose_.pose.pose.position.y;
            current_pose_.pose.pose.position.y = temp;
            std::cout << "Current Pose: ";
            std::cout << current_pose_.pose.pose.position.x << " " << current_pose_.pose.pose.position.y << std::endl;

            std::vector<geometry_msgs::Pose> path = computePath(automaton_, current_pose_, current_occ_map_, current_label_map_);

            ROS_INFO("Path computed with %lu waypoints.", path.size());

            nav_msgs::Path path_msg = convertToPath(path, current_occ_map_.header.frame_id);

            // while ros is ok
            while (ros::ok())
            {
                path_pub_.publish(path_msg);
                // wait 1 second
                ros::Duration(1.0).sleep();
            }

            // Delay for 10 seconds before allowing a new path computation
            ROS_INFO("Waiting forever seconds before re-planning...");
            ros::Duration(100000.0).sleep();
        }
    }

    nav_msgs::Path convertToPath(const std::vector<geometry_msgs::Pose> &poses, const std::string &frame_id)
    {
        nav_msgs::Path path_msg;
        path_msg.header.stamp = ros::Time::now();
        // path_msg.header.frame_id = frame_id;
        path_msg.header.frame_id = "world";

        for (const auto &pose : poses)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time::now();
            pose_stamped.header.frame_id = "world";
            pose_stamped.pose = pose;
            double tmp = pose_stamped.pose.position.x;
            pose_stamped.pose.position.x = pose_stamped.pose.position.y;
            pose_stamped.pose.position.y = tmp;
            path_msg.poses.push_back(pose_stamped);
        }

        return path_msg;
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ltl_planner_node");

    LTLPlannerNode planner_node;
    ros::spin();

    return 0;
}
