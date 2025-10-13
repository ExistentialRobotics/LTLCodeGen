#ifndef SEMANTIC_OCTOMAP_OCTOMAP_GENERATOR_ROS_H
#define SEMANTIC_OCTOMAP_OCTOMAP_GENERATOR_ROS_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Structure to hold RGB color and corresponding class ID
struct ColorClassMapping {
    int r, g, b;
    int class_id;
};

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "ssmi_mapping/semantic_octomap_node/octomap_generator.h"
#include "std_srvs/srv/empty.hpp"
#include "ssmi_interface/srv/get_rle.hpp"
#include "ssmi_interface/msg/ray_rle.hpp"
#include "octomap/octomap_types.h"
#include "octomap/Pointcloud.h"
#include "octomap/octomap.h"
#include "octomap_msgs/msg/octomap.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

#include "message_filters/subscriber.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/buffer.h"


struct coord_xy {
    double x, y;
    coord_xy(double x, double y) : x(x), y(y) {}
    bool operator==(const coord_xy& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template<> struct hash<coord_xy> {
        std::size_t operator()(const coord_xy& key) const {
            return std::hash<double>()(key.x) ^ (std::hash<double>()(key.y) << 1);
        }
    };
}

class OctomapGeneratorNode : public rclcpp::Node
{
public:
    /**
     * \brief Constructor
     */
    explicit OctomapGeneratorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    /// Desturctor
    ~OctomapGeneratorNode() override;
    /// Reset values to paramters from parameter server
    void reset();
    /**
     * \brief Callback to point cloud topic. Update the octomap and publish it in ROS
     * \param cloud ROS Pointcloud2 message in arbitrary frame (specified in the clouds header)
     */
    void insertCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);

    void publish2DOccupancyMap(const SemanticOctree* octomap,
                               const rclcpp::Time& stamp,
                               const std::string& frame_id);

    /**
     * \brief Save octomap to a file. NOTE: Not tested
     * \param filename The output filename
     */
    bool save(const char* filename) const;


protected:
    OctomapGeneratorBase<SemanticOctree>* octomap_generator_; ///<Octomap instance pointer
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr toggle_color_service_;  ///<ROS service to toggle semantic color display
    rclcpp::Service<ssmi_interface::srv::GetRLE>::SharedPtr RLE_service_;  ///<ROS service to querry RLE values
    void toggleUseSemanticColor(
        const std::shared_ptr<std_srvs::srv::Empty::Request> request,
        std::shared_ptr<std_srvs::srv::Empty::Response> response); ///<Function to toggle whether write semantic color or rgb color as when serializing octree
    void querry_RLE(
        const std::shared_ptr<ssmi_interface::srv::GetRLE::Request> request,
        std::shared_ptr<ssmi_interface::srv::GetRLE::Response> response);

    // Publishers
    rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr fullmap_pub_; ///<ROS publisher for octomap message
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_map_pub_; ///<ROS publisher for 2D occupancy map message
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_map_pub_; ///<ROS publisher for 2D occupancy map message semantic slice

    // TF and filters
    tf2_ros::Buffer tf_buffer_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> * pointcloud_sub_; ///<ROS subscriber for pointcloud message
    tf2_ros::TransformListener tf_listener_; ///<Listener for the transform between the camera and the world coordinates

    // Parameters
    std::string world_frame_id_; ///<Id of the world frame
    std::string pointcloud_topic_; ///<Topic name for subscribed pointcloud message
    float max_range_; ///<Max range for points to be inserted into octomap
    float raycast_range_; ///<Max range for points to perform raycasting to free unoccupied space
    float clamping_thres_max_; ///<Upper bound of occupancy probability for a node
    float clamping_thres_min_; ///<Lower bound of occupancy probability for a node
    float psi_; ///<Increment update value for a semantic class
    float phi_; ///<Decrement update value for a semantic class
    float resolution_; ///<Resolution of octomap
    float occupancy_thres_; ///<Minimum occupancy probability for a node to be considered as occupied
    float prob_hit_;  ///<Hit probability of sensor
    float prob_miss_; ///<Miss probability of sensor
    int class_id;
    bool publish_2d_map;
    double min_ground_z;
    double max_ground_z;
    double max_robot_z;
    bool enable_fuzzy_color_match_; ///<Enable fuzzy color matching using Euclidean distance
    int color_distance_threshold_; ///<Maximum squared Euclidean distance for color matching
    int max_color_logs_; ///<Maximum number of color query logs to print

    octomap_msgs::msg::Octomap map_msg_; //<ROS octomap message
    std::unordered_map<coord_xy,double> coordMap;
    int color_log_count_; ///<Counter for color query logs
    std::vector<ColorClassMapping> color_class_map_; ///<Cached color to class ID mappings

};

#endif //SEMANTIC_OCTOMAP_OCTOMAP_GENERATOR_ROS_H