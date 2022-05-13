/* 
 * LIBRARIES
 */

/********** C++ **********/
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

/********** ROS **********/
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

/********** OpenCV **********/
#include <opencv2/opencv.hpp>

/********** PCL **********/
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

/* 
 * GLOBAL CONFIGURATIONS 
 */

/********** Definitions **********/

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER, LIVOX };

/********** General Variables **********/
//float cloudCurvature[400000];
//int cloudSortInd[400000];
//int cloudNeighborPicked[400000];
//int cloudLabel[400000];

/********** Class for handling global parameters**********/
class Params{
    public:
        ros::NodeHandle nh;

        //Topics
        string ptCloudTopic;
        
        //LiDAR Configuration
        SensorType sensor;
        int N_SCAN;
        int Horizon_SCAN;
        double SCAN_Period;
        float lidarMinRange;
        float lidarMaxRange;

        //Frames
        string lidarFrame;
        string odomFrame;

        //Fast Odometry Graph
        float DISTANCE_SQ_THRESHOLD;
        float NEARBY_SCAN;
        int skipFrameNum;

        Params(){
            //Topics
            nh.param<std::string>("tar_sam/ptCloudTopic", ptCloudTopic, "velodyne_points");
            
            //LiDAR Configurarion
            nh.param<int>("tar_sam/N_SCAN", N_SCAN, 16);
            nh.param<double>("tar_sam/SCAN_Period", SCAN_Period, 0.1);
            nh.param<int>("tar_sam/Horizon_SCAN", Horizon_SCAN, 1800);
            nh.param<float>("tar_sam/lidarMinRange", lidarMinRange, 1.0);
            nh.param<float>("tar_sam/lidarMaxRange", lidarMaxRange, 1000.0);
            nh.param<std::string>("tar_sam/lidarFrame", lidarFrame, "velodyne");
            nh.param<std::string>("tar_sam/odomFrame", odomFrame, "odom");

            //Fast Odometry Graph
            nh.param<float>("tar_sam/DISTANCE_SQ_THRESHOLD", DISTANCE_SQ_THRESHOLD, 25.0);
            nh.param<float>("tar_sam/NEARBY_SCAN", NEARBY_SCAN, 2.5);
            nh.param<int>("tar_sam/skipFrameNum", skipFrameNum, 5);
        } 
};

template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame){
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    //if (thisPub.getNumSubscribers() != 0)
    thisPub.publish(tempCloud);
    return tempCloud;
}

float pointDistance(PointType p){
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2){
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

//bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }








