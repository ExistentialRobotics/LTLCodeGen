#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <yaml-cpp/yaml.h>

#include <spot/parseaut/public.hh>
#include <spot/twa/bdddict.hh>

#include "solar_planner/grid_map.h"
#include "solar_planner/environments/planning_spot_2d.h"
#include "solar_planner/astar_nx.h"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <iostream>

void custom_print(std::ostream &out, const spot::twa_graph_ptr &aut)
{
    // We need the dictionary to print the BDDs that label the edges
    const spot::bdd_dict_ptr &dict = aut->get_dict();

    // Some meta-data...
    out << "Acceptance: " << aut->get_acceptance() << '\n';
    out << "Number of sets: " << aut->num_sets() << '\n';
    out << "Number of states: " << aut->num_states() << '\n';
    out << "Number of edges: " << aut->num_edges() << '\n';
    out << "Initial state: " << aut->get_init_state_number() << '\n';
    out << "Atomic propositions:";
    for (spot::formula ap : aut->ap())
        out << ' ' << ap << " (=" << dict->varnum(ap) << ')';
    out << '\n';

    // Arbitrary data can be attached to automata, by giving them
    // a type and a name.  The HOA parser and printer both use the
    // "automaton-name" to name the automaton.
    if (auto name = aut->get_named_prop<std::string>("automaton-name"))
        out << "Name: " << *name << '\n';

    // For the following prop_*() methods, the return value is an
    // instance of the spot::trival class that can represent
    // yes/maybe/no.  These properties correspond to bits stored in the
    // automaton, so they can be queried in constant time.  They are
    // only set whenever they can be determined at a cheap cost: for
    // instance an algorithm that always produces deterministic automata
    // would set the deterministic property on its output.  In this
    // example, the properties that are set come from the "properties:"
    // line of the input file.
    out << "Complete: " << aut->prop_complete() << '\n';
    out << "Deterministic: " << (aut->prop_universal() && aut->is_existential()) << '\n';
    out << "Unambiguous: " << aut->prop_unambiguous() << '\n';
    out << "State-Based Acc: " << aut->prop_state_acc() << '\n';
    out << "Terminal: " << aut->prop_terminal() << '\n';
    out << "Weak: " << aut->prop_weak() << '\n';
    out << "Inherently Weak: " << aut->prop_inherently_weak() << '\n';
    out << "Stutter Invariant: " << aut->prop_stutter_invariant() << '\n';

    // States are numbered from 0 to n-1
    unsigned n = aut->num_states();
    for (unsigned s = 0; s < n; ++s)
    {
        out << "State " << s << ":\n";

        // The out(s) method returns a fake container that can be
        // iterated over as if the contents was the edges going
        // out of s.  Each of these edges is a quadruplet
        // (src,dst,cond,acc).  Note that because this returns
        // a reference, the edge can also be modified.
        for (auto &t : aut->out(s))
        {
            out << "  edge(" << t.src << " -> " << t.dst << ")\n    label = ";
            spot::bdd_print_formula(out, dict, t.cond);
            out << "\n    acc sets = " << t.acc << '\n';
        }
    }
}

std::vector<geometry_msgs::msg::Pose> computePath(const spot::twa_graph_ptr &automaton,
                                             const nav_msgs::msg::Odometry &our_pose,
                                             nav_msgs::msg::OccupancyGrid occ_map,
                                             nav_msgs::msg::OccupancyGrid label_map)
{
    // TODO: Get map data
    custom_print(std::cout, automaton);
    // Get map data
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
    std::unique_ptr<erl::GridMap<uint16_t>> MAP_ptr(new erl::GridMap<uint16_t>(mapmin, mapmax, mapres, true));

    // Convert OccupancyGrid data to GridMap format
    std::vector<uint16_t> map_data(occ_map.info.width * occ_map.info.height);
    for (size_t i = 0; i < occ_map.data.size(); ++i)
    {
        // std::cout<< "data before: " << occ_map.data[i] << std::endl;
        map_data[i] = (occ_map.data[i] < 0) ? 0 : static_cast<uint16_t>(occ_map.data[i]);
        // std::cout<< "data after: " << map_data[i] << std::endl;
    }

    // Step 3: Update the GridMap object with the new data
    MAP_ptr->setMap(map_data);

    // Inflate the map before using it for planning
    double inflation_radius = occ_map.info.resolution * 1.5; // Set the desired inflation radius (in meters)
    std::cout << "Inflation radius: " << inflation_radius << std::endl;
    std::vector<uint16_t> inflated_map_data = erl::inflateMap2D(*MAP_ptr, inflation_radius);

    // Update the GridMap object with the inflated data
    bool map_succ = MAP_ptr->setMap(inflated_map_data);

    if (map_succ)
    {
        std::cout << "Map successfully initialized!\n";
    }
    else
    {
        std::cerr << "Failed to initialize map!\n";
    }

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
            // std::cout << "Data: " << occ_map.data[index];

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
    std::vector<geometry_msgs::msg::Pose> our_poses;
    for (auto &coord : output.path)
    {
        geometry_msgs::msg::Pose pose;
        pose.position.x = erl::cells2meters(coord[0], mapmin[0], mapres[0]);
        pose.position.y = erl::cells2meters(coord[1], mapmin[1], mapres[1]);
        pose.position.z = 0.0;
        pose.orientation.w = 1.0;
        our_poses.push_back(pose);
    }

    return our_poses;
}

class LTLPlannerNode : public rclcpp::Node
{
public:
    LTLPlannerNode() : rclcpp::Node("ltl_planner_node"),
                       tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "Setting up Planning!");
        // Occupancy map subscriber
        occ_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_map_2D", 10, std::bind(&LTLPlannerNode::occMapCallback, this, std::placeholders::_1));

        // Label map subscriber
        label_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("label_map", 10, std::bind(&LTLPlannerNode::labelMapCallback, this, std::placeholders::_1));

        // Automaton subscriber
        automaton_sub_ = this->create_subscription<std_msgs::msg::String>("aut_str", 10, std::bind(&LTLPlannerNode::automatonCallback, this, std::placeholders::_1));

        // Path publisher
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("computed_path", 10);

        // AP to Id dict publisher
        ap_id_pub_ = this->create_publisher<std_msgs::msg::String>("ap_id", 10);

        this->declare_parameter<std::string>("world_frame_id", "odom");
        this->declare_parameter<std::string>("robot_frame_id", "husky_1/base_link");
        this->declare_parameter<bool>("debug_mode", true);
        this->declare_parameter<double>("debug_pose_x", 0.0);
        this->declare_parameter<double>("debug_pose_y", 0.0);
        world_frame_id = this->get_parameter("world_frame_id").as_string();
        robot_frame_id = this->get_parameter("robot_frame_id").as_string();
        debug_mode_    = this->get_parameter("debug_mode").as_bool();
        debug_pose_x_  = this->get_parameter("debug_pose_x").as_double();
        debug_pose_y_  = this->get_parameter("debug_pose_y").as_double();

        RCLCPP_INFO(this->get_logger(), "world_frame_id = %s", world_frame_id.c_str());
        RCLCPP_INFO(this->get_logger(), "robot_frame_id = %s", robot_frame_id.c_str());
        RCLCPP_INFO(this->get_logger(), "debug_mode = %s", (debug_mode_ ? "true" : "false"));
    }

private:
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr label_map_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr automaton_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr ap_id_pub_;

    // These are modified for appropriate dimensions
    nav_msgs::msg::OccupancyGrid current_occ_map_;
    nav_msgs::msg::OccupancyGrid current_label_map_;

    // these are used to check if map is updated
    nav_msgs::msg::OccupancyGrid none_modified_occ_map_;
    nav_msgs::msg::OccupancyGrid none_modified_label_map_;

    nav_msgs::msg::Odometry current_pose_;

    // TF2
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::string world_frame_id;
    std::string robot_frame_id;

    spot::twa_graph_ptr automaton_;

    bool occ_map_received_ = false;
    bool label_map_received_ = false;
    bool automaton_received_ = false;
    bool pose_received_ = false;
    bool maps_same_dim_ = false;

    // debug settings
    bool debug_mode_;
    double debug_pose_x_;
    double debug_pose_y_;

    // Function to get nav_msgs::Odometry from TF
    nav_msgs::msg::Odometry getOdomFromTF(const std::string &robot_frame_id)
    {
        nav_msgs::msg::Odometry odom_msg;
        geometry_msgs::msg::TransformStamped tf;

        try
        {
            tf = tf_buffer_.lookupTransform(world_frame_id, robot_frame_id, tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
            return odom_msg; // Return empty odometry if lookup fails
        }

        // Fill out the nav_msgs::Odometry message
        odom_msg.header.stamp = this->get_clock()->now();
        odom_msg.header.frame_id = world_frame_id;
        odom_msg.child_frame_id = robot_frame_id;

        // Set the position (translation)
        odom_msg.pose.pose.position.x = tf.transform.translation.x;
        odom_msg.pose.pose.position.y = tf.transform.translation.y;
        odom_msg.pose.pose.position.z = tf.transform.translation.z;

        // Set the orientation (rotation)
        odom_msg.pose.pose.orientation = tf.transform.rotation;

        pose_received_ = true;

        return odom_msg;
    }

    void occMapCallback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr & msg)
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
        RCLCPP_INFO(this->get_logger(), "Got a new occ map!");

        // Copy the received map to modify it
        nav_msgs::msg::OccupancyGrid modified_map = *msg;

        // Check if width is even, if so, add a column of obstacles
        if (modified_map.info.width % 2 == 0)
        {
            RCLCPP_INFO(this->get_logger(), "Width is even. Adding a column of obstacles...");
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
            RCLCPP_INFO(this->get_logger(), "Height is even. Adding a row of obstacles...");
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
        RCLCPP_INFO(this->get_logger(), "Occ map modified and updated.");

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void labelMapCallback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr & msg)
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
        RCLCPP_INFO(this->get_logger(), "Got a new label map!");

        // Copy the received map to modify it
        nav_msgs::msg::OccupancyGrid modified_map = *msg;

        // Check if width is even, if so, add a column of obstacles
        if (modified_map.info.width % 2 == 0)
        {
            RCLCPP_INFO(this->get_logger(), "Width is even. Adding a column of zeros...");
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
            RCLCPP_INFO(this->get_logger(), "Height is even. Adding a row of zeros...");
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
        RCLCPP_INFO(this->get_logger(), "Label map modified and updated.");

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void automatonCallback(const std_msgs::msg::String::ConstSharedPtr & msg)
    {
        if (automaton_received_)
        {
            RCLCPP_INFO(this->get_logger(), "Received automaton again!");
            const spot::bdd_dict_ptr &dict = automaton_->get_dict();
            std::ostringstream oss;

            // Dictionary start
            oss << "{ ";

            for (spot::formula ap : automaton_->ap())
            {
                // Get ap and corresponding id from bdd_dict
                auto ap_str = ap.ap_name();
                auto id = dict->varnum(ap);
                // Write to dict
                oss << "\"" << ap_str << "\"" << " : " << "\"" << std::to_string(id) << "\"";
                oss << ", ";
            }

            // Close dictionary
            oss << "} ";

            std_msgs::msg::String out_msg;
            out_msg.data = oss.str();
            ap_id_pub_->publish(out_msg);

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
            RCLCPP_INFO(this->get_logger(), "Automaton aborted.");
            return;
        }

        automaton_ = pa->aut;
        automaton_received_ = true;

        RCLCPP_INFO(this->get_logger(), "Received automaton! Type=%d", static_cast<int>(pa->type));

        delete str_parser;

        // Proceed with path planning if all conditions are met
        attemptPathPlanning();
    }

    void attemptPathPlanning()
    {
        if (debug_mode_){
            current_pose_.pose.pose.position.x = debug_pose_x_;
            current_pose_.pose.pose.position.y = debug_pose_y_;
            pose_received_ = true;
        } else {
            current_pose_ = getOdomFromTF(robot_frame_id);
        }

        // need to check if map dimensions matches before planning
        maps_same_dim_ = (current_occ_map_.info.width  == current_label_map_.info.width &&
        current_occ_map_.info.height == current_label_map_.info.height);

        if (!maps_same_dim_) {
            RCLCPP_INFO(this->get_logger(), "Maps are not aligned in dimensions. Waiting until they match ...");
        }

        if (automaton_received_ && occ_map_received_ && label_map_received_ && pose_received_ && maps_same_dim_)
        {
            RCLCPP_INFO(this->get_logger(), "Attempting to compute path...");

            std::swap(current_pose_.pose.pose.position.x, current_pose_.pose.pose.position.y);
            RCLCPP_INFO(this->get_logger(), "Current Pose: %.3f %.3f", current_pose_.pose.pose.position.x, current_pose_.pose.pose.position.y);

            std::vector<geometry_msgs::msg::Pose> path = computePath(automaton_, current_pose_, current_occ_map_, current_label_map_);

            RCLCPP_INFO(this->get_logger(), "Path computed with %zu waypoints.", path.size());

            nav_msgs::msg::Path path_msg = convertToPath(path, current_occ_map_.header.frame_id);

            path_pub_->publish(path_msg);

            // Ensure maps are relatively up-to-date during next planning
            occ_map_received_ = false;
            label_map_received_ = false;

            // Delay for 10 seconds before allowing a new path computation
            RCLCPP_INFO(this->get_logger(), "Waiting 10 seconds before re-planning...");
            rclcpp::sleep_for(std::chrono::seconds(10));
        }
    }

    nav_msgs::msg::Path convertToPath(const std::vector<geometry_msgs::msg::Pose> &poses, const std::string &frame_id)
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->get_clock()->now();
        path_msg.header.frame_id = world_frame_id;

        for (const auto &pose : poses)
        {
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = this->get_clock()->now();
            pose_stamped.header.frame_id = world_frame_id;
            pose_stamped.pose = pose;
            std::swap(pose_stamped.pose.position.x, pose_stamped.pose.position.y);
            path_msg.poses.push_back(pose_stamped);
        }

        return path_msg;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LTLPlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
