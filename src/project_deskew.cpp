// TODO: DESKEW section
#include "tar_sam/common.h"
#include "tar_sam/cloud_info.h"

struct genericPointXYZIRT{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    //float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (genericPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) 
    //(float, time, time)
)

using PointXYZIRT = genericPointXYZIRT;

class ProjectDeskew : public Params{
    private:

        //Subscribers and publishers
        ros::Subscriber sub_lidarCloud_raw;
        ros::Publisher  pub_lidarCloud_projectDeskew;
        ros::Publisher pub_laserCloudInfo;

        //Buffers
        std::deque<sensor_msgs::PointCloud2> cloudQueue;

        //Point cloud containers
        //pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
        pcl::PointCloud<PointType>::Ptr laserCloudIn;
        pcl::PointCloud<PointType>::Ptr fullCloud;
        pcl::PointCloud<PointType>::Ptr extractedCloud;
 

        sensor_msgs::PointCloud2 currentCloudMsg;

        //General porpose variables
        double timeScanCur;
        double timeScanEnd;
        std_msgs::Header cloudHeader;
        tar_sam::cloud_info cloudInfo;


        //flags
        int deskewFlag;

    public:
        ProjectDeskew():
            deskewFlag(0){
                sub_lidarCloud_raw = nh.subscribe<sensor_msgs::PointCloud2>(ptCloudTopic, 5, &ProjectDeskew::lidarCloudHandler, this, ros::TransportHints().tcpNoDelay());

                pub_lidarCloud_projectDeskew = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/lidarCloud_projectDeskew",1);
                pub_laserCloudInfo = nh.advertise<tar_sam::cloud_info> ("tar_sam/projectDeskew/cloud_info", 1);

                allocateMemory();
                resetParameters();

            }
        void allocateMemory(){
            //laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
            laserCloudIn.reset(new pcl::PointCloud<PointType>());
            fullCloud.reset(new pcl::PointCloud<PointType>());
            extractedCloud.reset(new pcl::PointCloud<PointType>());

            fullCloud->points.resize(N_SCAN*Horizon_SCAN);

            cloudInfo.startRingIndex.assign(N_SCAN, 0);
            cloudInfo.endRingIndex.assign(N_SCAN, 0);

            resetParameters();
        }
        void resetParameters()
        {
            laserCloudIn->clear();
            extractedCloud->clear();
        }
        ~ProjectDeskew(){};

        void lidarCloudHandler(const sensor_msgs::PointCloud2ConstPtr &lidarCloud_msg){

            if(!getRawCloud(lidarCloud_msg))
                return;

            projectPointCloud();
            publishClouds();
            resetParameters();
        }

        void publishClouds(){
            cloudInfo.header = cloudHeader;
            cloudInfo.cloud_deskewed = publishCloud(pub_lidarCloud_projectDeskew, extractedCloud, cloudHeader.stamp, odomFrame);
            pub_laserCloudInfo.publish(cloudInfo);

        }

        void projectPointCloud(){
            vector<int> scanStartInd(N_SCAN, 0);
            vector<int> scanEndInd(N_SCAN, 0);

            int cloudSize = laserCloudIn -> points.size();

            float startOri = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
            float endOri = -atan2(laserCloudIn->points[cloudSize - 1].y,
                                  laserCloudIn->points[cloudSize - 1].x) + 2 * M_PI;

            if (endOri - startOri > 3 * M_PI)
                endOri -= 2 * M_PI;
            
            else if (endOri - startOri < M_PI)
                endOri += 2 * M_PI;

            bool halfPassed = false;
            int count = cloudSize;
            vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCAN);

            for (int i = 0; i < cloudSize; ++i){
                PointType thisPoint;
                thisPoint.x = laserCloudIn->points[i].x;
                thisPoint.y = laserCloudIn->points[i].y;
                thisPoint.z = laserCloudIn->points[i].z;
                thisPoint.intensity = laserCloudIn->points[i].intensity;

                float angle = atan(thisPoint.z / sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                int scanID = 0;

                if (N_SCAN == 16){
                    scanID = int((angle + 15) / 2 + 0.5);
                    if (scanID > (N_SCAN - 1) || scanID < 0){
                        count--;
                        continue;
                    }
                }
                else{
                    ROS_ERROR("Wrong scan number, use 16");
                    ros::shutdown();

                }
                float ori = -atan2(thisPoint.y, thisPoint.x);
                if (!halfPassed){ 
                    if (ori < startOri - M_PI / 2)
                        ori += 2 * M_PI;
                    else if (ori > startOri + M_PI * 3 / 2)
                        ori -= 2 * M_PI;
                    if (ori - startOri > M_PI)
                        halfPassed = true;
                }
                else{
                    ori += 2 * M_PI;
                    if (ori < endOri - M_PI * 3 / 2)
                        ori += 2 * M_PI;
                    else if (ori > endOri + M_PI / 2)
                        ori -= 2 * M_PI;
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                //point.intensity = scanID + scanPeriod * relTime;
                laserCloudScans[scanID].push_back(thisPoint); 
            }

            //cloudSize = count;
            for (int i = 0; i < N_SCAN; i++){ 
                cloudInfo.startRingIndex[i] = extractedCloud->size() + 5;
                *extractedCloud += laserCloudScans[i];
                cloudInfo.endRingIndex[i] = extractedCloud->size() - 6;
            }


        } 

        bool getRawCloud(const sensor_msgs::PointCloud2ConstPtr lidarCloud_msg){
            
            //Point cloud msg to buffer
            cloudQueue.push_back(*lidarCloud_msg);
            //if (cloudQueue.size() <= 2)
            //    return false;

            //Convert ros point cloud to pcl point cloud 
            //(TODO: add support for other LiDAR sensors)
            currentCloudMsg = std::move(cloudQueue.front());
            cloudQueue.pop_front();

            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn); //Ouputs a warning if there is not a field match

            checkCloud(currentCloudMsg);

            cloudHeader = currentCloudMsg.header;
            timeScanCur = cloudHeader.stamp.toSec();

            //if (deskewFlag == 1){
            //    //uncomment the following lines if the velodyne message has each point time field
            //    //timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

            //    //comment the following lines if the velodyne message has each point time field
            //    ROS_ERROR("Edit genericPointXYZIRT: add time field");
            //    ros::shutdown(); 
            //}

            return true;
        }

        void checkCloud(sensor_msgs::PointCloud2 currentCloudMsg){
            //Check if cloud is dense 
            if (laserCloudIn->is_dense == false){
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points");
                ros::shutdown();
            }
            
            ////Check ring channel
            //static int ringFlag = 0;
            //if (ringFlag == 0){
            //    ringFlag = -1;
            //    for (auto &field : currentCloudMsg.fields){
            //        if (field.name == "ring"){
            //            ringFlag = 1;
            //            break;
            //        }
            //    }
            //    if (ringFlag == -1){
            //        ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
            //        ros::shutdown();
            //    }
            //}

            ////Check each point time
            //if (deskewFlag == 0){
            //    deskewFlag = -1;
            //    //std::cout << "Field: " <<std::endl;
            //    for (auto &field : currentCloudMsg.fields){
            //        //std::cout << "\t" << field.name <<std::endl;
            //        if (field.name == "time" || field.name == "t"){
            //            deskewFlag = 1;
            //            break;
            //        }
            //    }
            //    if (deskewFlag == -1)
            //        ROS_WARN("Each point timestamp not available, point's time will be added by brute force");
            //    
            //}

        }

};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "tar_sam");

    ProjectDeskew projectDeskew;

    
    ROS_INFO("\033[1;36m\n ---> Module: \033[0;36m Project and Deskew \033[1;36m Started <---\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}
