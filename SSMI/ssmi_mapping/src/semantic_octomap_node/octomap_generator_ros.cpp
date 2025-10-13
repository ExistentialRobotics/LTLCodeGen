#include <ssmi_mapping/semantic_octomap_node/octomap_generator_ros.h>
#include <octomap_msgs/conversions.h>
#include <nav_msgs/msg/occupancy_grid.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <cmath>
#include <sstream>
#include <cstring> // For std::memcpy


OctomapGeneratorNode::OctomapGeneratorNode(const rclcpp::NodeOptions & options): Node("octomap_generator",
        rclcpp::NodeOptions(options)
            .allow_undeclared_parameters(true)
            .automatically_declare_parameters_from_overrides(true)),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_, this, false),
    color_log_count_(0)
{
    // Enable dedicated thread for TF buffer
    tf_buffer_.setUsingDedicatedThread(true);
    
    // Initiate octree
    RCLCPP_INFO(this->get_logger(), "Semantic octomap generated!");
    octomap_generator_ = new OctomapGenerator<PCLSemantics, SemanticOctree>();
    toggle_color_service_ = this->create_service<std_srvs::srv::Empty>("toggle_use_semantic_color", std::bind(&OctomapGeneratorNode::toggleUseSemanticColor, this, std::placeholders::_1, std::placeholders::_2));
    RLE_service_ = this->create_service<ssmi_interface::srv::GetRLE>("querry_RLE", std::bind(&OctomapGeneratorNode::querry_RLE, this, std::placeholders::_1, std::placeholders::_2));

    reset();
    fullmap_pub_ = this->create_publisher<octomap_msgs::msg::Octomap>("octomap_full", 1);
    occ_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_map_2D", 1);
    semantic_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("semantic_map_2D", 1);
    pointcloud_sub_ = new message_filters::Subscriber<sensor_msgs::msg::PointCloud2>(this, pointcloud_topic_);
    pointcloud_sub_->registerCallback(std::bind(&OctomapGeneratorNode::insertCloudCallback, this, std::placeholders::_1));
}

OctomapGeneratorNode::~OctomapGeneratorNode() {
    delete pointcloud_sub_;
    delete octomap_generator_;
}

void OctomapGeneratorNode::reset()
{
    // Use get_parameter_or with default values (parameters already auto-declared from YAML)
    pointcloud_topic_ = this->get_parameter_or("octomap.pointcloud_topic", std::string("pointcloud"));
    world_frame_id_ = this->get_parameter_or("octomap.world_frame_id", std::string("env_small"));
    resolution_ = this->get_parameter_or("octomap.resolution", 0.5);
    max_range_ = this->get_parameter_or("octomap.max_range", 15.0);
    raycast_range_ = this->get_parameter_or("octomap.raycast_range", 10.0);
    clamping_thres_min_ = this->get_parameter_or("octomap.clamping_thres_min", 1e-4);
    clamping_thres_max_ = this->get_parameter_or("octomap.clamping_thres_max", 1.0-1e-4);
    occupancy_thres_ = this->get_parameter_or("octomap.occupancy_thres", 0.5);
    prob_hit_ = this->get_parameter_or("octomap.prob_hit", 0.7);
    prob_miss_ = this->get_parameter_or("octomap.prob_miss", 0.4);
    psi_ = this->get_parameter_or("octomap.psi", 1.0);
    phi_ = this->get_parameter_or("octomap.phi", -0.1);
    publish_2d_map = this->get_parameter_or("octomap.publish_2d_map", true);
    min_ground_z = this->get_parameter_or("octomap.min_ground_z", 1.0);
    max_ground_z = this->get_parameter_or("octomap.max_ground_z", 3.5);
    max_robot_z = this->get_parameter_or("octomap.max_robot_z", 1.0);
    enable_fuzzy_color_match_ = this->get_parameter_or("octomap.enable_fuzzy_color_match", true);
    color_distance_threshold_ = this->get_parameter_or("octomap.color_distance_threshold", 1000);
    max_color_logs_ = this->get_parameter_or("octomap.max_color_logs", 50);

    octomap_generator_->setClampingThresMin(clamping_thres_min_);
    octomap_generator_->setClampingThresMax(clamping_thres_max_);
    octomap_generator_->setResolution(resolution_);
    octomap_generator_->setOccupancyThres(occupancy_thres_);
    octomap_generator_->setProbHit(prob_hit_);
    octomap_generator_->setProbMiss(prob_miss_);
    octomap_generator_->setPsi(psi_);
    octomap_generator_->setPhi(phi_);
    octomap_generator_->setRayCastRange(raycast_range_);
    octomap_generator_->setMaxRange(max_range_);

    // Debug: Log loaded color parameters
    auto param_names = this->list_parameters({}, 0).names;
    RCLCPP_INFO(this->get_logger(), "Total parameters loaded: %zu", param_names.size());
    int color_param_count = 0;
    color_class_map_.clear();
    
    for (const auto& name : param_names) {
        if (name.find("R") == 0 && name.find("G") != std::string::npos && name.find("B") != std::string::npos) {
            int id = this->get_parameter(name).as_int();
            
            // Parse RGB from parameter name
            size_t r_pos = 1, g_pos = name.find("G"), b_pos = name.find("B");
            int r = std::stoi(name.substr(r_pos, g_pos - r_pos));
            int g = std::stoi(name.substr(g_pos + 1, b_pos - g_pos - 1));
            int b = std::stoi(name.substr(b_pos + 1));
            
            // Add to color mapping cache
            color_class_map_.push_back({r, g, b, id});
            
            RCLCPP_INFO(this->get_logger(), "Loaded color parameter: %s = %d (RGB: %d, %d, %d)", name.c_str(), id, r, g, b);
            color_param_count++;
        }
    }
    RCLCPP_INFO(this->get_logger(), "Loaded %d color mapping parameters", color_param_count);
}

void OctomapGeneratorNode::toggleUseSemanticColor(const std::shared_ptr<std_srvs::srv::Empty::Request> request, std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
    octomap_generator_->setUseSemanticColor(!octomap_generator_->isUseSemanticColor());
    if(octomap_generator_->isUseSemanticColor())
        RCLCPP_INFO(this->get_logger(), "Using semantic color");
    else
        RCLCPP_INFO(this->get_logger(), "Using rgb color");
    if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_))
        fullmap_pub_->publish(map_msg_);
    else
        RCLCPP_ERROR(this->get_logger(), "Error serializing OctoMap");
}

void OctomapGeneratorNode::querry_RLE(const std::shared_ptr<ssmi_interface::srv::GetRLE::Request> request, std::shared_ptr<ssmi_interface::srv::GetRLE::Response> response)
{
    const octomap::point3d origin(request->origin.x, request->origin.y, request->origin.z);

    for (int i = 0; i < (int)request->end_points.size(); ++i)
    {
        const octomap::point3d endPoint(request->end_points[i].x, request->end_points[i].y, request->end_points[i].z);
        ssmi_interface::msg::RayRLE rayRLE_msg;
        if (octomap_generator_->get_ray_RLE(origin, endPoint, rayRLE_msg))
        {
            response->rle_list.push_back(rayRLE_msg);
        }
    }
}

void OctomapGeneratorNode::insertCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud_msg)
{
    // Voxel filter to down sample the point cloud
    // Create the filtering object
    pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*cloud_msg, *cloud);
    // Get tf transform
    geometry_msgs::msg::TransformStamped sensorToWorldTf;
    try {
        sensorToWorldTf = tf_buffer_.lookupTransform(
          world_frame_id_,
          cloud_msg->header.frame_id,
          cloud_msg->header.stamp,
          rclcpp::Duration(std::chrono::seconds(1))
        );
    } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR(this->get_logger(),
                     "Transform error of sensor data: %s", ex.what());
        return;
    }
    // Transform coordinate
    Eigen::Isometry3d iso = tf2::transformToEigen(sensorToWorldTf);
    Eigen::Matrix4f sensorToWorld = iso.matrix().cast<float>();

    octomap_generator_->insertPointCloud(cloud, sensorToWorld);

    // Publish full octomap
    map_msg_.header.frame_id = world_frame_id_;
    map_msg_.header.stamp = cloud_msg->header.stamp;
    if (octomap_msgs::fullMapToMsg(*octomap_generator_->getOctree(), map_msg_)) {
        fullmap_pub_->publish(map_msg_);
    } else {
        RCLCPP_ERROR(this->get_logger(), "Error serializing full OctoMap");
    }

    // Publish 2D occupancy map
    if (publish_2d_map) {
        publish2DOccupancyMap(octomap_generator_->getOctree(), cloud_msg->header.stamp, world_frame_id_);
    }
}

void OctomapGeneratorNode::publish2DOccupancyMap(const SemanticOctree* octomap,
                                                 const rclcpp::Time& stamp,
                                                 const std::string& frame_id)
{
    // Reset color log counter for each map publication
    color_log_count_ = 0;

    // get dimensions of octree
    double minX, minY, minZ, maxX, maxY, maxZ;
    octomap->getMetricMin(minX, minY, minZ);
    octomap->getMetricMax(maxX, maxY, maxZ);
    octomap::point3d minPt = octomap::point3d(minX, minY, minZ);

    unsigned int tree_depth = octomap->getTreeDepth();

    octomap::OcTreeKey paddedMinKey = octomap->coordToKey(minPt);

    nav_msgs::msg::OccupancyGrid::Ptr occupancy_map (new nav_msgs::msg::OccupancyGrid());
    nav_msgs::msg::OccupancyGrid::Ptr semantic_map (new nav_msgs::msg::OccupancyGrid());

    unsigned int width, height;
    double res;

    unsigned int ds_shift = tree_depth-16;

    occupancy_map->header.stamp = stamp;
    occupancy_map->header.frame_id = frame_id;
    occupancy_map->info.resolution = res = octomap->getNodeSize(16);
    occupancy_map->info.width = width = (maxX-minX) / res + 1;
    occupancy_map->info.height = height = (maxY-minY) / res + 1;
    occupancy_map->info.origin.position.x = minX  - (res / (float)(1<<ds_shift) ) + res;
    occupancy_map->info.origin.position.y = minY  - (res / (float)(1<<ds_shift) );

    occupancy_map->data.clear();
    occupancy_map->data.resize(width*height, -1);

    semantic_map->header.stamp = stamp;
    semantic_map->header.frame_id = frame_id;
    semantic_map->info.resolution = res = octomap->getNodeSize(16);
    semantic_map->info.width = width = (maxX - minX) / res + 1;
    semantic_map->info.height = height = (maxY - minY) / res + 1;
    semantic_map->info.origin.position.x = minX - (res / (float)(1 << ds_shift)) + res;
    semantic_map->info.origin.position.y = minY - (res / (float)(1 << ds_shift));

    semantic_map->data.clear();
    semantic_map->data.resize(width * height, -1); // init all cells to -1

    // traverse all leafs in the tree:
    unsigned int treeDepth = std::min<unsigned int>(16, octomap->getTreeDepth());
    for (typename SemanticOctree::iterator it = octomap->begin(treeDepth), end = octomap->end(); it != end; ++it)
    {

        double node_z = it.getZ();
        double node_half_side = pow(it.getSize(), 1 / 3) / 2;
        double top_side = node_z + node_half_side;
        double bottom_side = node_z - node_half_side;
        coord_xy xy_coordinates(it.getX(), it.getY());

        // ignore direct measurements of the ground (z = 0), and things that are too high
        if ((bottom_side >= min_ground_z && bottom_side <= max_ground_z) ||
            (top_side >= min_ground_z && top_side <= max_ground_z) ||
            (bottom_side <= min_ground_z && top_side >= max_ground_z))
        {
            bool occupied = octomap->isNodeOccupied(*it);
            int intSize = 1 << (16 - it.getDepth());

      octomap::OcTreeKey minKey=it.getIndexKey();

      for (int dx = 0; dx < intSize; dx++)
      {
        for (int dy = 0; dy < intSize; dy++)
        {
          int posX = std::max<int>(0, minKey[0] + dx - paddedMinKey[0]);
          posX>>=ds_shift;

          int posY = std::max<int>(0, minKey[1] + dy - paddedMinKey[1]);
          posY>>=ds_shift;

          int idx = width * posY + posX;

                    if (occupied)
                    {
                        // Height logic for occupancy map
                        if (node_z < max_robot_z && node_z >= resolution_)
                            occupancy_map->data[idx] = 100;

                        // Bird's eye view logic for semantic map
                        auto r = it->getSemantics().getSemanticColor().r;
                        auto g = it->getSemantics().getSemanticColor().g;
                        auto b = it->getSemantics().getSemanticColor().b;

                        // Try exact match first
                        std::string color_key = "R" + std::to_string(r) + "G" + std::to_string(g) + "B" + std::to_string(b);
                        class_id = this->get_parameter_or(color_key, -1);
                        
                        // If exact match fails and fuzzy matching is enabled, find closest color using Euclidean distance
                        if (class_id == -1 && enable_fuzzy_color_match_ && !color_class_map_.empty()) {
                            int min_distance = INT_MAX;
                            int best_class_id = -1;
                            
                            for (const auto& color_map : color_class_map_) {
                                // Calculate Euclidean distance in RGB space
                                int dr = (int)r - color_map.r;
                                int dg = (int)g - color_map.g;
                                int db = (int)b - color_map.b;
                                int distance = dr*dr + dg*dg + db*db;
                                
                                if (distance < min_distance) {
                                    min_distance = distance;
                                    best_class_id = color_map.class_id;
                                }
                            }
                            
                            // Only accept if within reasonable distance threshold
                            if (min_distance < color_distance_threshold_) {
                                class_id = best_class_id;
                            } else {
                                RCLCPP_DEBUG(this->get_logger(), "R%dG%dB%d No match found: min_distance=%d > color_distance_threshold_=%d.", (int)r, (int)g, (int)b, min_distance, color_distance_threshold_);
                            }
                        } else {
                            RCLCPP_WARN_ONCE(this->get_logger(), "Color matching failed to initiate: fuzzy_match=%s, color_map_size=%zu, class_id=%d.", 
                                            enable_fuzzy_color_match_ ? "true" : "false", color_class_map_.size(), class_id);
                        }
                        
                        
                        // Debug: Log color query with throttling
                        if (color_log_count_ < max_color_logs_ && class_id != -1) {
                            RCLCPP_DEBUG(this->get_logger(), "Color query: %s (R=%d, G=%d, B=%d) -> class_id=%d", color_key.c_str(), (int)r, (int)g, (int)b, class_id);
                            color_log_count_++;
                        }
                        
                        if (coordMap.find(xy_coordinates) == coordMap.end())
                        {
                            // if we have never seen this (X,Y) before, store the Z value
                            if (node_z < max_robot_z)
                                coordMap[xy_coordinates] = node_z;
                            // check if height is smaller than max_robot_z
                            if (node_z < max_robot_z && class_id != -1)
                                semantic_map->data[idx] = class_id;
                            // semantic_map->data[idx] = class_id;
                        }
                        else if (coordMap[xy_coordinates] <= node_z)
                        {
                            // if this is the highest Z value we've seen for this (X,Y), save it as the new max
                            if (node_z < max_robot_z)
                                coordMap[xy_coordinates] = node_z;
                            // check if height is smaller than max_robot_z
                            // if (node_z < max_robot_z)
                            if (class_id != -1)
                                semantic_map->data[idx] = class_id;
                        }
                    }

                    else if (occupancy_map->data[idx] == -1) // -1 is the default value for unseen cells
                    {
                        // if (node_z < max_robot_z)
                        occupancy_map->data[idx] = 0;
                        semantic_map->data[idx] = -128; // min-int to indicate no class
                    }
                }
            }
        } // if within bounds
    } // iterate over octree

    occ_map_pub_->publish(*occupancy_map);
    semantic_map_pub_->publish(*semantic_map);
} // end of function

bool OctomapGeneratorNode::save(const char* filename) const
{
    return octomap_generator_->save(filename);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OctomapGeneratorNode>());
    rclcpp::shutdown();

    return 0;
}