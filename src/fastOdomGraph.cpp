#include "tar_sam/common.h"
#include "tar_sam/cloud_info.h"

#include <boost/smart_ptr/shared_ptr.hpp>
#include <gtsam/nonlinear/PriorFactor.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <eigen3/Eigen/Dense>

#include "lidarFactor.hpp"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

using namespace message_filters;


class FastOdomGraph: public Params{
    public:

        //Subscribers and  Publishers
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud1;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud2;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud3;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud4;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud5;
        typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
        typedef Synchronizer<MySyncPolicy> Sync;
        boost::shared_ptr<Sync> sync;

        ros::Subscriber subImu;
        ros::Subscriber subOdometry;

        ros::Publisher pub_cloudCornerLast;
        ros::Publisher pub_cloudSurfLast;
        ros::Publisher pub_fastOdom;
        ros::Publisher pub_fastOdom_path;
        ros::Publisher pub_ImuOdometry;
        ros::Publisher pubLaserCloudInfo;

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
        std::deque<sensor_msgs::Imu> imuQueOpt;
        std::deque<sensor_msgs::Imu> imuQueImu;
        
        //General variables
        int corner_correspondence;
        int plane_correspondence;
        int laserCloudCornerLastNum;
        int laserCloudSurfLastNum;
        double syncTime;

        double syncTime_arr[100];

        int frameCount;

        const double delta_t = 0;

        nav_msgs::Path laserPath;
        tar_sam::cloud_info cloudInfo;
        
        std_msgs::Header cloudHeader;

        //Transformations
        Eigen::Quaterniond q_w_curr;
        Eigen::Vector3d t_w_curr;

        double para_q[4];
        double para_t[3];

        Eigen::Map<Eigen::Quaterniond> q_last_curr;
        Eigen::Map<Eigen::Vector3d> t_last_curr;

        // T_bl: tramsform points from lidar frame to imu frame 
        gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
        // T_lb: tramsform points from imu frame to lidar frame
        gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

        //GTSAM models
        gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
        gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
        gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
        gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
        gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
        gtsam::Vector noiseModelBetweenBias;

        gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
        gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

        gtsam::Pose3 prevPose_;
        gtsam::Vector3 prevVel_;
        gtsam::NavState prevState_;
        gtsam::imuBias::ConstantBias prevBias_;

        gtsam::NavState prevStateOdom;
        gtsam::imuBias::ConstantBias prevBiasOdom;


        gtsam::ISAM2 optimizer;
        gtsam::NonlinearFactorGraph graphFactors;
        gtsam::Values graphValues;

        int key = 1;


        //mutex
        std::mutex mtx;

        //Flags
        bool systemInited; 
        bool firstLaserFrame; 

        bool doneFirstOpt = false;
        double lastImuT_imu = -1;
        double lastImuT_opt = -1;

        FastOdomGraph():kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>()), kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>()), corner_correspondence(0), plane_correspondence(0), laserCloudCornerLastNum(0), laserCloudSurfLastNum(0), q_w_curr(1, 0, 0, 0), t_w_curr(0, 0, 0), q_last_curr(para_q), t_last_curr(para_t) {

            //cloud1.subscribe(nh, "/laser_cloud_sharp", 1);
            //cloud2.subscribe(nh, "/laser_cloud_less_sharp", 1);
            //cloud3.subscribe(nh, "/laser_cloud_flat", 1);
            //cloud4.subscribe(nh, "laser_cloud_less_flat", 1);

            cloud1.subscribe(nh, "tar_sam/feature/cloudSharp", 1);
            cloud2.subscribe(nh, "tar_sam/feature/cloudLessSharp", 1);
            cloud3.subscribe(nh, "tar_sam/feature/cloudFlat", 1);
            cloud4.subscribe(nh, "tar_sam/feature/cloudLessFlat", 1);
            cloud5.subscribe(nh, "tar_sam/cloudDeskew", 1);
            sync.reset(new Sync(MySyncPolicy(10), cloud1, cloud2, cloud3, cloud4, cloud5));
            sync->registerCallback(boost::bind(&FastOdomGraph::featuresHandler, this, _1, _2, _3, _4, _5));

            subImu = nh.subscribe<sensor_msgs::Imu>  (imuTopic, 2000, &FastOdomGraph::imuHandler, this, ros::TransportHints().tcpNoDelay());
            subOdometry = nh.subscribe<nav_msgs::Odometry>("tar_sam/mapping/odometry_incremental", 5,    &FastOdomGraph::odometryHandler, this, ros::TransportHints().tcpNoDelay());
            
            pub_cloudCornerLast = nh.advertise<sensor_msgs::PointCloud2> ("tar_sam/features/cornerLast", 1);
            pub_cloudSurfLast = nh.advertise<sensor_msgs::PointCloud2> ("tar_sam/features/surfLast", 1);
            pub_fastOdom = nh.advertise<nav_msgs::Odometry> ("tar_sam/fastOdom", 1);
            pub_fastOdom_path = nh.advertise<nav_msgs::Path> ("tar_sam/fastOdom_path", 1);
            pub_ImuOdometry = nh.advertise<nav_msgs::Odometry> ("tar_sam/ImuOdom", 2000);
            pubLaserCloudInfo = nh.advertise<tar_sam::cloud_info> ("tar_sam/feature/cloud_info", 1);

            para_q[0] = 0.0;
            para_q[1] = 0.0;
            para_q[2] = 0.0;
            para_q[3] = 1.0;

            para_t[0] = 0.0;
            para_t[1] = 0.0;
            para_t[2] = 0.0;

            frameCount = 0;
            
            syncTime = 0.0;

            systemInited = false;
            firstLaserFrame = false;

            setupFactorGraph();
            allocateMemory();
            resetClouds();
        }

        void setupFactorGraph(){
            boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
            p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
            p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
            p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
            gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

            priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
            priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
            priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
            correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
            correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
            noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
            
            imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
            imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
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

        void resetClouds(){
            cornerPointsSharp->clear();
            cornerPointsLessSharp->clear();
            surfPointsFlat->clear();
            surfPointsLessFlat->clear();

            //laserCloudCornerLast->clear();
            //laserCloudSurfLast->clear();
            //laserCloudFullRes->clear();
        }

        void resetOptimization(){
            gtsam::ISAM2Params optParameters;
            optParameters.relinearizeThreshold = 0.1;
            optParameters.relinearizeSkip = 1;
            optimizer = gtsam::ISAM2(optParameters);

            gtsam::NonlinearFactorGraph newGraphFactors;
            graphFactors = newGraphFactors;

            gtsam::Values NewGraphValues;
            graphValues = NewGraphValues;
        }

        void resetGraphParams(){
            lastImuT_imu = -1;
            doneFirstOpt = false;
            systemInited = false;
        }

        void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg){
            std::lock_guard<std::mutex> lock(mtx);

            double currentCorrectionTime = ROS_TIME(odomMsg);

            for (int i = 0; i < 99; i++){
                if (currentCorrectionTime == syncTime_arr[i]){
                    float p_x = odomMsg->pose.pose.position.x;
                    float p_y = odomMsg->pose.pose.position.y;
                    float p_z = odomMsg->pose.pose.position.z;
                    float r_x = odomMsg->pose.pose.orientation.x;
                    float r_y = odomMsg->pose.pose.orientation.y;
                    float r_z = odomMsg->pose.pose.orientation.z;
                    float r_w = odomMsg->pose.pose.orientation.w;
                    bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;
                    gtsam::Pose3 priorPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));

                    gtsam::Pose3 curPose = priorPose.compose(lidar2Imu);
                    gtsam::PriorFactor<gtsam::Pose3> pose_factor_1(X(i), curPose, correctionNoise);
                    graphFactors.add(pose_factor_1);
                    ROS_INFO("correction");

                    syncTime_arr[i] = 0;
                    break;
                }
                syncTime_arr[i] = 0;
            }

        }


        void featuresHandler(const sensor_msgs::PointCloud2ConstPtr &cloudSharp_msg, const sensor_msgs::PointCloud2ConstPtr &cloudLessSharp_msg, const sensor_msgs::PointCloud2ConstPtr &cloudFlat_msg, const sensor_msgs::PointCloud2ConstPtr &cloudLessFlat_msg, const sensor_msgs::PointCloud2ConstPtr &cloudDeskew_msg){
            std::lock_guard<std::mutex> lock(mtx);
            //mtx_buffer.lock();
            //cornerSharpBuf.push(cloudSharp_msg);
            //cornerLessSharpBuf.push(cloudLessSharp_msg);
            //surfFlatBuf.push(cloudFlat_msg);
            //surfLessFlatBuf.push(cloudLessFlat_msg);
            //mtx_buffer.unlock();
            //

            syncTime = cloudFlat_msg->header.stamp.toSec();
            cloudHeader = cloudFlat_msg->header;            // new cloud header

            // make sure we have imu data to integrate
            if (imuQueOpt.empty())
                return;
            
            //Get feature clouds
            pcl::fromROSMsg(*cloudSharp_msg, *cornerPointsSharp); 
            pcl::fromROSMsg(*cloudLessSharp_msg, *cornerPointsLessSharp); 
            pcl::fromROSMsg(*cloudFlat_msg, *surfPointsFlat); 
            pcl::fromROSMsg(*cloudLessFlat_msg, *surfPointsLessFlat); 
            cloudInfo.cloud_deskewed = *cloudDeskew_msg;

            if (!systemInited){
                resetOptimization();

                // initial Pose
                gtsam::Pose3 initialPose = gtsam::Pose3(gtsam::Rot3::Quaternion(1, 0, 0, 0), gtsam::Point3(0, 0, 0));
                prevPose_ = initialPose.compose(lidar2Imu);
                gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
                graphFactors.add(priorPose);

                // initial velocity
                prevVel_ = gtsam::Vector3(0, 0, 0);
                gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
                graphFactors.add(priorVel);

                // initial bias
                prevBias_ = gtsam::imuBias::ConstantBias();
                gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
                graphFactors.add(priorBias);

                // add values
                graphValues.insert(X(0), prevPose_);
                graphValues.insert(V(0), prevVel_);
                graphValues.insert(B(0), prevBias_);

                // optimize once
                optimizer.update(graphFactors, graphValues);
                graphFactors.resize(0);
                graphValues.clear();

                imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
                imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

                syncTime_arr[0] = syncTime;
 
                key = 1;
                systemInited = true;
                ROS_INFO("\033[1;32m Fast Odometry Initialized \033[0m");
                updateLaserFeatures();
                return;
            }

            resetGraph4Speed();

            computeLaserOdom();
            updateLaserFeatures();
            publishClouds();
            resetClouds();

            integrateIMUData();

            addIMUFactor();
            addLiDARFactor();

            optimizeUpdateGraph();
            getInitialOdom();

            syncTime_arr[key] = syncTime;
            ++key;
            doneFirstOpt = true;

            publishLaserOdom();

            //frameCount++;
        }

        void getInitialOdom(){

            double t_x = prevPose_.translation().x();             //t_w_curr.x(); //
            double t_y = prevPose_.translation().y();             //t_w_curr.y(); //
            double t_z = prevPose_.translation().z();             //t_w_curr.z(); //
                                                                  
            double q_w = prevPose_.rotation().toQuaternion().w(); //q_w_curr.w(); //
            double q_x = prevPose_.rotation().toQuaternion().x(); //q_w_curr.x(); //
            double q_y = prevPose_.rotation().toQuaternion().y(); //q_w_curr.y(); //
            double q_z = prevPose_.rotation().toQuaternion().z(); //q_w_curr.z(); //
            double roll, pitch, yaw;
            tf::Quaternion orientation(q_x, q_y, q_z, q_w);
            tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

            cloudInfo.initialGuessX = t_x;
            cloudInfo.initialGuessY = t_y;
            cloudInfo.initialGuessZ = t_z;
            cloudInfo.initialGuessRoll  = roll;
            cloudInfo.initialGuessPitch = pitch;
            cloudInfo.initialGuessYaw   = yaw;

            cloudInfo.odomAvailable = true;
        }

        void publishClouds(){

            cloudInfo.header = cloudHeader;

            cloudInfo.cloud_corner  = publishCloud(pub_cloudCornerLast,  laserCloudCornerLast,  cloudHeader.stamp, lidarFrame);
            cloudInfo.cloud_surface = publishCloud(pub_cloudSurfLast, laserCloudSurfLast, cloudHeader.stamp, lidarFrame);
            
            // publish to mapOptimization
            pubLaserCloudInfo.publish(cloudInfo);
        }

        void optimizeUpdateGraph(){
            // insert predicted values
            gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
            graphValues.insert(X(key), propState_.pose());
            graphValues.insert(V(key), propState_.v());
            graphValues.insert(B(key), prevBias_);

            // optimize
            optimizer.update(graphFactors, graphValues);
            optimizer.update();
            graphFactors.resize(0);
            graphValues.clear();
            
            // Overwrite the beginning of the preintegration for the next step.
            gtsam::Values result = optimizer.calculateEstimate();
            prevPose_  = result.at<gtsam::Pose3>(X(key));
            prevVel_   = result.at<gtsam::Vector3>(V(key));
            prevState_ = gtsam::NavState(prevPose_, prevVel_);
            prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));

            //t_w_curr.x() = prevPose_.translation().x();
            //t_w_curr.y() = prevPose_.translation().y();
            //t_w_curr.z() = prevPose_.translation().z();

            //q_w_curr.w() = prevPose_.rotation().toQuaternion().w();
            //q_w_curr.x() = prevPose_.rotation().toQuaternion().x();
            //q_w_curr.y() = prevPose_.rotation().toQuaternion().y();
            //q_w_curr.z() = prevPose_.rotation().toQuaternion().z();

            // Reset the optimization preintegration object.
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

            // 2. after optiization, re-propagate imu odometry preintegration
            prevStateOdom = prevState_;
            prevBiasOdom  = prevBias_;
            // first pop imu message older than current correction data
            double lastImuQT = -1;
            while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < syncTime - delta_t)
            {
                lastImuQT = ROS_TIME(&imuQueImu.front());
                imuQueImu.pop_front();
            }
            // repropogate
            if (!imuQueImu.empty())
            {
                // reset bias use the newly optimized bias
                imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
                // integrate imu message from the beginning of this optimization
                for (int i = 0; i < (int)imuQueImu.size(); ++i)
                {
                    sensor_msgs::Imu *thisImu = &imuQueImu[i];
                    double imuTime = ROS_TIME(thisImu);
                    double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                            gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                    lastImuQT = imuTime;
                }
            }

        }

        void addLiDARFactor(){
            // add LiDAR factor
            gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(q_last_curr.w(), q_last_curr.x(), q_last_curr.y(), q_last_curr.z()), gtsam::Point3(t_last_curr.x(), t_last_curr.y(), t_last_curr.z()));
            gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
            gtsam::BetweenFactor<gtsam::Pose3> pose_factor(X(key-1), X(key), curPose, correctionNoise2);
            graphFactors.add(pose_factor);

            gtsam::Pose3 priorPose = gtsam::Pose3(gtsam::Rot3::Quaternion(q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z()), gtsam::Point3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z()));
            curPose = priorPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> pose_factor_1(X(key), curPose, correctionNoise);
            graphFactors.add(pose_factor_1);
        }

        void addIMUFactor(){
            // add imu factor to graph
            const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
            gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
            graphFactors.add(imu_factor);

            // add imu bias between factor
            graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                             gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

        }



        void resetGraph4Speed(){
            if (key == 100){
                // get updated noise before reset
                gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
                gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
                gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
                // reset graph
                resetOptimization();
                // add pose
                gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
                graphFactors.add(priorPose);
                // add velocity
                gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
                graphFactors.add(priorVel);
                // add bias
                gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
                graphFactors.add(priorBias);
                // add values
                graphValues.insert(X(0), prevPose_);
                graphValues.insert(V(0), prevVel_);
                graphValues.insert(B(0), prevBias_);
                // optimize once
                optimizer.update(graphFactors, graphValues);
                graphFactors.resize(0);
                graphValues.clear();

                key = 1;
            }

        }
        void integrateIMUData(){
            while (!imuQueOpt.empty()){
                // pop and integrate imu data that is between two optimizations
                sensor_msgs::Imu *thisImu = &imuQueOpt.front();
                double imuTime = ROS_TIME(thisImu);
                if (imuTime < syncTime - delta_t)
                {
                    double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                    imuIntegratorOpt_->integrateMeasurement(
                            gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                            gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                    
                    lastImuT_opt = imuTime;
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }

        }

        void updateLaserFeatures(){
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

        }

        void publishLaserOdom(){
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = odomFrame;
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(syncTime);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
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

        void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw){
            std::lock_guard<std::mutex> lock(mtx);

            sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

            imuQueOpt.push_back(thisImu);
            imuQueImu.push_back(thisImu);

            if (doneFirstOpt == false)
                return;

            double imuTime = ROS_TIME(&thisImu);
            double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
            lastImuT_imu = imuTime;

            // integrate this single imu message
            imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                    gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

            // predict odometry
            gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

            // publish odometry
            nav_msgs::Odometry odometry;
            odometry.header.stamp = thisImu.header.stamp;
            odometry.header.frame_id = odomFrame;
            odometry.child_frame_id = "odom_imu";

            // transform imu pose to ldiar
            gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
            gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

            odometry.pose.pose.position.x = lidarPose.translation().x();
            odometry.pose.pose.position.y = lidarPose.translation().y();
            odometry.pose.pose.position.z = lidarPose.translation().z();
            odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
            odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
            odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
            odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
            
            odometry.twist.twist.linear.x = currentState.velocity().x();
            odometry.twist.twist.linear.y = currentState.velocity().y();
            odometry.twist.twist.linear.z = currentState.velocity().z();
            odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
            odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
            odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
            pub_ImuOdometry.publish(odometry);
        }


};



int main(int argc, char** argv){
    ros::init(argc, argv, "tar_sam");

    FastOdomGraph fastOdomGraph;

    ROS_INFO("\033[1;36m\n ---> Module: \033[0;36m Fast Odometry Graph \033[1;36m Started <--- \033[0m");
   
    ros::spin();

    return 0;
}
