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
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float64MultiArray.h>


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

/********** tf **********/
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

/********** OpenCV **********/
#include <opencv2/opencv.hpp>

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
        string imuTopic;
        string gpsTopic;
        
        //LiDAR Setup
        SensorType sensor;
        int N_SCAN;
        int Horizon_SCAN;
        double SCAN_Period;
        float lidarMinRange;
        float lidarMaxRange;

        // IMU
        float imuAccNoise;
        float imuGyrNoise;
        float imuAccBiasN;
        float imuGyrBiasN;
        float imuGravity;
        float imuRPYWeight;
        vector<double> extRotV;
        vector<double> extRPYV;
        vector<double> extTransV;
        Eigen::Matrix3d extRot;
        Eigen::Matrix3d extRPY;
        Eigen::Vector3d extTrans;
        Eigen::Quaterniond extQRPY;

        //Frames
        string baselinkFrame;
        string lidarFrame;
        string odomFrame;
        string mapFrame;

        //Fast Odometry Graph
        float DISTANCE_SQ_THRESHOLD;
        float NEARBY_SCAN;
        int skipFrameNum;

        //Map Optimization
        bool useImuHeadingInitialization;
        bool useGpsElevation;
        float gpsCovThreshold;
        float poseCovThreshold;

        bool savePCD;
        string savePCDDirectory;

        int edgeFeatureMinValidNum;
        int surfFeatureMinValidNum;

        float odometrySurfLeafSize;
        float mappingCornerLeafSize;
        float mappingSurfLeafSize ;

        float z_tollerance; 
        float rotation_tollerance;

        int numberOfCores;
        double mappingProcessInterval;

        float surroundingkeyframeAddingDistThreshold; 
        float surroundingkeyframeAddingAngleThreshold; 
        float surroundingKeyframeDensity;
        float surroundingKeyframeSearchRadius;
        
        bool  loopClosureEnableFlag;
        float loopClosureFrequency;
        int   surroundingKeyframeSize;
        float historyKeyframeSearchRadius;
        float historyKeyframeSearchTimeDiff;
        int   historyKeyframeSearchNum;
        float historyKeyframeFitnessScore;

        float globalMapVisualizationSearchRadius;
        float globalMapVisualizationPoseDensity;
        float globalMapVisualizationLeafSize;

        Params(){
            //Topics
            nh.param<std::string>("tar_sam/ptCloudTopic", ptCloudTopic, "velodyne_points");
            nh.param<std::string>("tar_sam/imuTopic", imuTopic, "imu_fix");
            nh.param<std::string>("lio_sam/gpsTopic", gpsTopic, "odometry/gps");
            
            //LiDAR Setup
            nh.param<int>("tar_sam/N_SCAN", N_SCAN, 16);
            nh.param<double>("tar_sam/SCAN_Period", SCAN_Period, 0.1);
            nh.param<int>("tar_sam/Horizon_SCAN", Horizon_SCAN, 1800);
            nh.param<float>("tar_sam/lidarMinRange", lidarMinRange, 1.0);
            nh.param<float>("tar_sam/lidarMaxRange", lidarMaxRange, 1000.0);

            //Frames
            nh.param<std::string>("tar_sam/baselinkFrame", baselinkFrame, "base_link");
            nh.param<std::string>("tar_sam/lidarFrame", lidarFrame, "velodyne");
            nh.param<std::string>("tar_sam/odomFrame", odomFrame, "odom");
            nh.param<std::string>("tar_sam/mapFrame", mapFrame, "map");

            //IMU Setup
            nh.param<float>("tar_sam/imuAccNoise", imuAccNoise, 0.01);
            nh.param<float>("tar_sam/imuGyrNoise", imuGyrNoise, 0.001);
            nh.param<float>("tar_sam/imuAccBiasN", imuAccBiasN, 0.0002);
            nh.param<float>("tar_sam/imuGyrBiasN", imuGyrBiasN, 0.00003);
            nh.param<float>("tar_sam/imuGravity", imuGravity, 9.80511);
            nh.param<float>("tar_sam/imuRPYWeight", imuRPYWeight, 0.01);
            nh.param<vector<double>>("tar_sam/extrinsicRot", extRotV, vector<double>());
            nh.param<vector<double>>("tar_sam/extrinsicRPY", extRPYV, vector<double>());
            nh.param<vector<double>>("tar_sam/extrinsicTrans", extTransV, vector<double>());
            extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
            extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
            extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
            extQRPY = Eigen::Quaterniond(extRPY).inverse();

            //Fast Odometry Graph
            nh.param<float>("tar_sam/DISTANCE_SQ_THRESHOLD", DISTANCE_SQ_THRESHOLD, 25.0);
            nh.param<float>("tar_sam/NEARBY_SCAN", NEARBY_SCAN, 2.5);
            nh.param<int>("tar_sam/skipFrameNum", skipFrameNum, 5);

            //Map Optimization

            nh.param<bool>("tar_sam/useImuHeadingInitialization", useImuHeadingInitialization, false);
            nh.param<bool>("tar_sam/useGpsElevation", useGpsElevation, false);
            nh.param<float>("tar_sam/gpsCovThreshold", gpsCovThreshold, 2.0);
            nh.param<float>("tar_sam/poseCovThreshold", poseCovThreshold, 25.0);

            nh.param<int>("tar_sam/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
            nh.param<int>("tar_sam/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

            nh.param<float>("tar_sam/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
            nh.param<float>("tar_sam/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

            nh.param<float>("tar_sam/z_tollerance", z_tollerance, FLT_MAX);
            nh.param<float>("tar_sam/rotation_tollerance", rotation_tollerance, FLT_MAX);

            nh.param<int>("tar_sam/numberOfCores", numberOfCores, 2);
            nh.param<double>("tar_sam/mappingProcessInterval", mappingProcessInterval, 0.15);

            nh.param<float>("tar_sam/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
            nh.param<float>("tar_sam/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
            nh.param<float>("tar_sam/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
            nh.param<float>("tar_sam/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

            nh.param<bool>("tar_sam/loopClosureEnableFlag", loopClosureEnableFlag, false);
            nh.param<float>("tar_sam/loopClosureFrequency", loopClosureFrequency, 1.0);
            nh.param<int>("tar_sam/surroundingKeyframeSize", surroundingKeyframeSize, 50);
            nh.param<float>("tar_sam/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
            nh.param<float>("tar_sam/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
            nh.param<int>("tar_sam/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
            nh.param<float>("tar_sam/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

            nh.param<float>("tar_sam/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
            nh.param<float>("tar_sam/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
            nh.param<float>("tar_sam/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);
        } 


        sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
        {
            sensor_msgs::Imu imu_out = imu_in;
            // rotate acceleration
            Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
            acc = extRot * acc;
            imu_out.linear_acceleration.x = acc.x();
            imu_out.linear_acceleration.y = acc.y();
            imu_out.linear_acceleration.z = acc.z();
            // rotate gyroscope
            Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
            gyr = extRot * gyr;
            imu_out.angular_velocity.x = gyr.x();
            imu_out.angular_velocity.y = gyr.y();
            imu_out.angular_velocity.z = gyr.z();
            // rotate roll pitch yaw
            Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
            Eigen::Quaterniond q_final = q_from * extQRPY;
            imu_out.orientation.x = q_final.x();
            imu_out.orientation.y = q_final.y();
            imu_out.orientation.z = q_final.z();
            imu_out.orientation.w = q_final.w();

            if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
            {
                ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
                ros::shutdown();
            }

            return imu_out;
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

template<typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}







