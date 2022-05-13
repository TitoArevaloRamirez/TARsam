#include "tar_sam/common.h"

#include <boost/smart_ptr/shared_ptr.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <eigen3/Eigen/Dense>

#include "lidarFactor.hpp"

using namespace message_filters;


class FastOdomGraph: public Params{
    public:

        //Subscribers and  Publishers
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud1;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud2;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud3;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud4;

        typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;

        typedef Synchronizer<MySyncPolicy> Sync;
        boost::shared_ptr<Sync> sync;

        ros::Publisher pub_cloudCornerLast;
        ros::Publisher pub_cloudSurfLast;
        ros::Publisher pub_fastOdom;
        ros::Publisher pub_fastOdom_path;

        //Point cloud containers
        pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
        pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
        pcl::PointCloud<PointType>::Ptr surfPointsFlat;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;
        
        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast;
        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast;

        pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
        pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
        pcl::PointCloud<PointType>::Ptr laserCloudFullRes;


        //Buffers
        //queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
        //queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
        //queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
        //queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
        //queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
        
        //General variables
        int corner_correspondence;
        int plane_correspondence;
        int laserCloudCornerLastNum;
        int laserCloudSurfLastNum;
        double syncTime;

        int frameCount;

        nav_msgs::Path laserPath;

        //Transformations
        Eigen::Quaterniond q_w_curr;
        Eigen::Vector3d t_w_curr;

        double para_q[4];
        double para_t[3];

        Eigen::Map<Eigen::Quaterniond> q_last_curr;
        Eigen::Map<Eigen::Vector3d> t_last_curr;


        //mutex
        std::mutex mtx;

        //Flags
        bool systemInited; 
        bool firstLaserFrame; 

        FastOdomGraph():kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>()), kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>()), corner_correspondence(0), plane_correspondence(0), laserCloudCornerLastNum(0), laserCloudSurfLastNum(0), q_w_curr(1, 0, 0, 0), t_w_curr(0, 0, 0), q_last_curr(para_q), t_last_curr(para_t) {

            //cloud1.subscribe(nh, "/laser_cloud_sharp", 1);
            //cloud2.subscribe(nh, "/laser_cloud_less_sharp", 1);
            //cloud3.subscribe(nh, "/laser_cloud_flat", 1);
            //cloud4.subscribe(nh, "laser_cloud_less_flat", 1);

            cloud1.subscribe(nh, "tar_sam/feature/cloudSharp", 1);
            cloud2.subscribe(nh, "tar_sam/feature/cloudLessSharp", 1);
            cloud3.subscribe(nh, "tar_sam/feature/cloudFlat", 1);
            cloud4.subscribe(nh, "tar_sam/feature/cloudLessFlat", 1);
            sync.reset(new Sync(MySyncPolicy(10), cloud1, cloud2, cloud3, cloud4));
            sync->registerCallback(boost::bind(&FastOdomGraph::featuresHandler, this, _1, _2, _3, _4));
            
            pub_cloudCornerLast = nh.advertise<sensor_msgs::PointCloud2> ("tar_sam/features/cornerLast", 1);
            pub_cloudSurfLast = nh.advertise<sensor_msgs::PointCloud2> ("tar_sam/features/surfLast", 1);
            pub_fastOdom = nh.advertise<nav_msgs::Odometry> ("tar_sam/fastOdom", 1);
            pub_fastOdom_path = nh.advertise<nav_msgs::Path> ("tar_sam/fastOdom_path", 1);

            para_q[0] = 0.0;
            para_q[1] = 0.0;
            para_q[2] = 0.0;
            para_q[3] = 1.0;

            para_t[0] = 0.0;
            para_t[1] = 0.0;
            para_t[2] = 0.0;

            frameCount = 0;
            
            //q_last_curr = new Eigen::Map<Eigen::Quaterniond>(para_q);
            //t_last_curr = new Eigen::Map<Eigen::Vector3d>(para_t);
            //
            syncTime = 0.0;

            systemInited = false;
            firstLaserFrame = false;

            allocateMemory();
            resetParameters();
        }

        void allocateMemory(){
            cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
            cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
            surfPointsFlat.reset(new pcl::PointCloud<PointType>());
            surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

            //kdtreeCornerLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());
            //kdtreeSurfLast.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

            laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
            laserCloudFullRes.reset(new pcl::PointCloud<PointType>());

        }

        void resetParameters(){
            cornerPointsSharp->clear();
            cornerPointsLessSharp->clear();
            surfPointsFlat->clear();
            surfPointsLessFlat->clear();

            //laserCloudCornerLast->clear();
            //laserCloudSurfLast->clear();
            //laserCloudFullRes->clear();
        }

        void featuresHandler(const sensor_msgs::PointCloud2ConstPtr &cloudSharp_msg, const sensor_msgs::PointCloud2ConstPtr &cloudLessSharp_msg, const sensor_msgs::PointCloud2ConstPtr &cloudFlat_msg, const sensor_msgs::PointCloud2ConstPtr &cloudLessFlat_msg){
            std::lock_guard<std::mutex> lock(mtx);
            //mtx_buffer.lock();
            //cornerSharpBuf.push(cloudSharp_msg);
            //cornerLessSharpBuf.push(cloudLessSharp_msg);
            //surfFlatBuf.push(cloudFlat_msg);
            //surfLessFlatBuf.push(cloudLessFlat_msg);
            //mtx_buffer.unlock();
            //
            syncTime = cloudFlat_msg->header.stamp.toSec();
            
            //Get feature clouds
            pcl::fromROSMsg(*cloudSharp_msg, *cornerPointsSharp); 
            pcl::fromROSMsg(*cloudLessSharp_msg, *cornerPointsLessSharp); 
            pcl::fromROSMsg(*cloudFlat_msg, *surfPointsFlat); 
            pcl::fromROSMsg(*cloudLessFlat_msg, *surfPointsLessFlat); 

            if (!systemInited){
                systemInited = true;
                ROS_INFO("\033[1;32m Fast Odometry Initialized \033[0m");
            }
            else{

                computeLaserOdom();
            }
            publishLaserOdom();

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            resetParameters();

            frameCount++;
        }

        void publishLaserOdom(){
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = odomFrame;
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(syncTime);
            laserOdometry.pose.pose.orientation.x = q_last_curr.x();
            laserOdometry.pose.pose.orientation.y = q_last_curr.y();
            laserOdometry.pose.pose.orientation.z = q_last_curr.z();
            laserOdometry.pose.pose.orientation.w = q_last_curr.w();
            laserOdometry.pose.pose.position.x = t_last_curr.x();
            laserOdometry.pose.pose.position.y = t_last_curr.y();
            laserOdometry.pose.pose.position.z = t_last_curr.z();
            pub_fastOdom.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = odomFrame;
            pub_fastOdom_path.publish(laserPath);

        }

        void computeLaserOdom(){
            int cornerPointsSharpNum = cornerPointsSharp->points.size();
            int surfPointsFlatNum = surfPointsFlat->points.size();

            for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter){
                corner_correspondence = 0;
                plane_correspondence = 0;

                //ceres::LossFunction *loss_function = NULL;
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;

                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(para_q, 4, q_parameterization);
                problem.AddParameterBlock(para_t, 3);

                pcl::PointXYZI pointSel;
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                // find correspondence for corner features
                for (int i = 0; i < cornerPointsSharpNum; ++i){
                    TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                    kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD){
                        closestPointInd = pointSearchInd[0];
                        int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j){
                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * (laserCloudCornerLast->points[j].x - pointSel.x) + (laserCloudCornerLast->points[j].y - pointSel.y) * (laserCloudCornerLast->points[j].y - pointSel.y) + (laserCloudCornerLast->points[j].z - pointSel.z) *(laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2){
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j){

                            // if in the same scan line, continue
                            if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                continue;

                            // if not in nearby scans, end the loop
                            if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *(laserCloudCornerLast->points[j].x - pointSel.x) + (laserCloudCornerLast->points[j].y - pointSel.y) *(laserCloudCornerLast->points[j].y - pointSel.y) + (laserCloudCornerLast->points[j].z - pointSel.z) * (laserCloudCornerLast->points[j].z - pointSel.z);

                            if (pointSqDis < minPointSqDis2){
                                // find nearer point
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    if (minPointInd2 >= 0) { // both closestPointInd and minPointInd2 is valid
                        Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                   cornerPointsSharp->points[i].y,
                                                   cornerPointsSharp->points[i].z);
                        Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                     laserCloudCornerLast->points[closestPointInd].y,
                                                     laserCloudCornerLast->points[closestPointInd].z);
                        Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                     laserCloudCornerLast->points[minPointInd2].y,
                                                     laserCloudCornerLast->points[minPointInd2].z);

                        double s;
                        //if (DISTORTION)
                        //    s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                        //else
                        //    s = 1.0;
                        s = 1.0;
                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        corner_correspondence++;
                    }
                }

                // find correspondence for plane features
                for (int i = 0; i < surfPointsFlatNum; ++i){
                    TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                    kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                    int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                    if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD){
                        closestPointInd = pointSearchInd[0];

                        // get closest point's scan ID
                        int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                        double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                        // search in the direction of increasing scan line
                        for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j){
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                    (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                    (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or lower scan line
                            if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2){
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            // if in the higher scan line
                            else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3){
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        // search in the direction of decreasing scan line
                        for (int j = closestPointInd - 1; j >= 0; --j){
                            // if not in nearby scans, end the loop
                            if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                break;

                            double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * (laserCloudSurfLast->points[j].x - pointSel.x) + (laserCloudSurfLast->points[j].y - pointSel.y) * (laserCloudSurfLast->points[j].y - pointSel.y) + (laserCloudSurfLast->points[j].z - pointSel.z) * (laserCloudSurfLast->points[j].z - pointSel.z);

                            // if in the same or higher scan line
                            if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2){
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                            else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3){
                                // find nearer point
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }

                        if (minPointInd2 >= 0 && minPointInd3 >= 0){

                            Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                        surfPointsFlat->points[i].y,
                                                        surfPointsFlat->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                            laserCloudSurfLast->points[closestPointInd].y,
                                                            laserCloudSurfLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                            laserCloudSurfLast->points[minPointInd2].y,
                                                            laserCloudSurfLast->points[minPointInd2].z);
                            Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                            laserCloudSurfLast->points[minPointInd3].y,
                                                            laserCloudSurfLast->points[minPointInd3].z);

                            double s;
                            //if (DISTORTION)
                            //    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                            //else
                            //    s = 1.0;
                            s = 1.0;
                            ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            plane_correspondence++;
                        }
                    }
                }

                if ((corner_correspondence + plane_correspondence) < 10){
                    ROS_WARN("Corner and Surface correspondences lower than 10\n");
                }

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

            }

            t_w_curr = t_w_curr + q_w_curr * t_last_curr;
            q_w_curr = q_w_curr * q_last_curr;

        }

        void TransformToStart(PointType const *const pi, PointType *const po){
            //interpolation ratio
            double s;
            //if (DISTORTION)
            //    s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
            //else
            //    s = 1.0;
            s = 1.0;
            Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
            Eigen::Vector3d t_point_last = s * t_last_curr;
            Eigen::Vector3d point(pi->x, pi->y, pi->z);
            Eigen::Vector3d un_point = q_point_last * point + t_point_last;
        
            po->x = un_point.x();
            po->y = un_point.y();
            po->z = un_point.z();
            po->intensity = pi->intensity;
        }



};



int main(int argc, char** argv){
    ros::init(argc, argv, "tar_sam");

    FastOdomGraph fastOdomGraph;

    ROS_INFO("\033[1;36m\n ---> Module: \033[0;36m Fast Odometry Graph \033[1;36m Started <--- \033[0m");
   
    ros::spin();

    return 0;
}
