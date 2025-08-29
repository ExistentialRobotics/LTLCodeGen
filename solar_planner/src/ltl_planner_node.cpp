#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Quaternion.h>

#include <spot/parseaut/public.hh>
#include <spot/twa/bdddict.hh>

#include "solar_planner/grid_map.h"
#include "solar_planner/environments/planning_spot_2d.h"
#include "solar_planner/astar_nx.h"

#include <tf/transform_listener.h>

#include <cmath>

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

std::vector<geometry_msgs::Pose> computePath(const spot::twa_graph_ptr &automaton,
                                             const nav_msgs::Odometry &our_pose,
                                             nav_msgs::OccupancyGrid occ_map,
                                             nav_msgs::OccupancyGrid label_map)
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
        ROS_INFO("Map successfully initialized!");
    }
    else
    {
        ROS_ERROR("Failed to initialize map!");
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
        path_pub_ = nh_.advertise<nav_msgs::Path>("computed_path", 1, true);

        // AP to Id dict publisher
        ap_id_pub_ = nh_.advertise<std_msgs::String>("ap_id", 1);

        // Private node handle for parameters (~param_name)
        ros::NodeHandle pnh("~");

        // Load frame ids and debug flag
        pnh.param("world_frame_id", world_frame_id, std::string("odom"));
        pnh.param("robot_frame_id", robot_frame_id, std::string("husky_1/base_link"));
        pnh.param("debug_mode", debug_mode_, true);
        pnh.param("debug_pose_x", debug_pose_x_, 0.0);
        pnh.param("debug_pose_y", debug_pose_y_, 0.0);

        ROS_INFO_STREAM("world_frame_id = " << world_frame_id);
        ROS_INFO_STREAM("robot_frame_id = " << robot_frame_id);
        ROS_INFO_STREAM("debug_mode = " << (debug_mode_ ? "true" : "false"));
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber occ_map_sub_;
    ros::Subscriber label_map_sub_;
    ros::Subscriber automaton_sub_;
    ros::Publisher path_pub_;
    ros::Publisher ap_id_pub_;

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
    bool maps_same_dim_ = false;

    // debug settings
    bool debug_mode_;
    double debug_pose_x_;
    double debug_pose_y_;

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
            ROS_INFO("Received automaton again!!!!!");
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

            std_msgs::String msg;
            msg.data = oss.str();

            ap_id_pub_.publish(msg);

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
        ROS_INFO("Received automaton!!!!!");

        ROS_INFO("Aut Type: [%d]", (int)pa->type); // 0 is HOA format

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
        ROS_INFO("Maps are not aligned in dimensions. Waiting until they match ...");
        }

        if (automaton_received_ && occ_map_received_ && label_map_received_ && pose_received_ && maps_same_dim_)
        {
            ROS_INFO("Attempting to compute path...");

            double temp = current_pose_.pose.pose.position.x;
            current_pose_.pose.pose.position.x = current_pose_.pose.pose.position.y;
            current_pose_.pose.pose.position.y = temp;
            std::cout << "Current Pose: ";
            std::cout << current_pose_.pose.pose.position.x << " " << current_pose_.pose.pose.position.y << std::endl;

            std::vector<geometry_msgs::Pose> path = computePath(automaton_, current_pose_, current_occ_map_, current_label_map_);

            ROS_INFO("Path computed with %lu waypoints.", path.size());

            nav_msgs::Path path_msg = convertToPath(path, current_occ_map_.header.frame_id);

            path_pub_.publish(path_msg);

            // Ensure maps are relatively up-to-date during next planning
            occ_map_received_ = false;
            label_map_received_ = false;

            // Delay for 10 seconds before allowing a new path computation
            ROS_INFO("Waiting 10 seconds before re-planning...");
            ros::Duration(10.0).sleep();
        }
    }

    nav_msgs::Path convertToPath(const std::vector<geometry_msgs::Pose> &poses, const std::string &frame_id)
    {
        nav_msgs::Path path_msg;
        path_msg.header.stamp = ros::Time::now();
        path_msg.header.frame_id = world_frame_id;

        for (const auto &pose : poses)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = ros::Time::now();
            pose_stamped.header.frame_id = world_frame_id;
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
