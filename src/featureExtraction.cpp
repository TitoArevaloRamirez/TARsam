#include "tar_sam/common.h"
#include "tar_sam/cloud_info.h"
#include <functional>

using std::atan2;
using std::cos;
using std::sin;

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

float cloudCurvature[400000];
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

class FeatureExtraction: public Params{
    public:
        //Subscribers and Publishers
        //ros::Subscriber sub_lidarCloudInfo;

        ros::Subscriber sub_lidarCloud_raw;
        ros::Publisher  pub_lidarCloud_projectDeskew;

        ros::Publisher pub_cornerPointsSharp;
        ros::Publisher pub_cornerPointsLessSharp;
        ros::Publisher pub_surfacePointsFlat;
        ros::Publisher pub_surfacePointsLessFlat;
        
        //Buffers
        std::deque<sensor_msgs::PointCloud2> cloudQueue;

        //Point cloud containers
        pcl::PointCloud<PointType>::Ptr laserCloud;

        pcl::PointCloud<PointType>::Ptr laserCloudIn;

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;

        sensor_msgs::PointCloud2 currentCloudMsg;


        //General variables
        double timeScanCur;
        double timeScanEnd;
        std_msgs::Header cloudHeader;

        std::vector<smoothness_t> cloudSmoothness;
        //float *cloudCurvature;
        int *cloudNeighborPicked;
        int *cloudLabel;
        int *cloudSortInd;

        //tar_sam::cloud_info cloudInfo;
        //std_msgs::Header cloudHeader;

        vector<int> scanStartInd;
        vector<int> scanEndInd;
        
        //flags
        int deskewFlag;

        FeatureExtraction():scanStartInd(N_SCAN, 0), scanEndInd(N_SCAN, 0){

            //sub_lidarCloudInfo = nh.subscribe<tar_sam::cloud_info>("tar_sam/projectDeskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

            sub_lidarCloud_raw = nh.subscribe<sensor_msgs::PointCloud2>(ptCloudTopic, 100, &FeatureExtraction::laserCloudHandler, this, ros::TransportHints().tcpNoDelay());

            pub_lidarCloud_projectDeskew = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/lidarCloud_projectDeskew",100);

            pub_cornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/feature/cloudSharp", 100);
            pub_cornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/feature/cloudLessSharp", 100);
            pub_surfacePointsFlat = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/feature/cloudFlat", 100);
            pub_surfacePointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("tar_sam/feature/cloudLessFlat", 100);

            allocateMemory();
            resetParameters();
        

        }

        void allocateMemory(){
            laserCloud.reset(new pcl::PointCloud<PointType>());
            laserCloudIn.reset(new pcl::PointCloud<PointType>());

            //cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
            //cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
            //surfPointsFlat.reset(new pcl::PointCloud<PointType>());
            //surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

            //cloudSmoothness.resize(400000);
            //cloudCurvature = new float[400000];
            cloudNeighborPicked = new int[400000];
            cloudLabel = new int[400000];


            //cloudCurvature = new float[400000];
             cloudSortInd = new int[400000];
            //cloudNeighborPicked = new int[400000];
            //cloudLabel = new int[400000];
        }

        void resetParameters(){
            laserCloud -> clear();
            laserCloudIn -> clear();

            //cornerPointsSharp -> clear();
            //cornerPointsLessSharp -> clear();
            //surfPointsFlat -> clear();
            //surfPointsLessFlat -> clear();
        }


        void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &lidarCloud_msg){

            pcl::fromROSMsg(*lidarCloud_msg, *laserCloudIn); //Ouputs a warning if there is not a field match
            //removeClosedPointCloud(*laserCloudIn, *laserCloudIn, 0.1);

            cloudHeader = lidarCloud_msg->header;
            timeScanCur = cloudHeader.stamp.toSec();

            projectPointCloud();
            calculateSmoothness();
            extractFeatures();
            publishFeatureClouds();
            resetParameters();

        }

        void projectPointCloud(){
            int cloudSize = laserCloudIn -> points.size();

            float startOri = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
            float endOri = -atan2(laserCloudIn->points[cloudSize - 1].y,
                                  laserCloudIn->points[cloudSize - 1].x) +
                           2 * M_PI;

            if (endOri - startOri > 3 * M_PI)
            {
                endOri -= 2 * M_PI;
            }
            else if (endOri - startOri < M_PI)
            {
                endOri += 2 * M_PI;
            }
            //printf("end Ori %f\n", endOri);

            bool halfPassed = false;
            int count = cloudSize;
            PointType point;
            std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCAN);
            for (int i = 0; i < cloudSize; i++)
            {
                point.x = laserCloudIn->points[i].x;
                point.y = laserCloudIn->points[i].y;
                point.z = laserCloudIn->points[i].z;
                point.intensity = laserCloudIn->points[i].intensity;

                float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
                int scanID = 0;

                if (N_SCAN == 16)
                {
                    scanID = int((angle + 15) / 2 + 0.5);
                    if (scanID > (N_SCAN - 1) || scanID < 0)
                    {
                        count--;
                        continue;
                    }
                }
                else if (N_SCAN == 32)
                {
                    scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
                    if (scanID > (N_SCAN - 1) || scanID < 0)
                    {
                        count--;
                        continue;
                    }
                }
                else if (N_SCAN == 64)
                {   
                    if (angle >= -8.83)
                        scanID = int((2 - angle) * 3.0 + 0.5);
                    else
                        scanID = N_SCAN / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                    // use [0 50]  > 50 remove outlies 
                    if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                    {
                        count--;
                        continue;
                    }
                }
                else
                {
                    printf("wrong scan number\n");
                    ROS_BREAK();
                }
                //printf("angle %f scanID %d \n", angle, scanID);

                float ori = -atan2(point.y, point.x);
                if (!halfPassed)
                { 
                    if (ori < startOri - M_PI / 2)
                    {
                        ori += 2 * M_PI;
                    }
                    else if (ori > startOri + M_PI * 3 / 2)
                    {
                        ori -= 2 * M_PI;
                    }

                    if (ori - startOri > M_PI)
                    {
                        halfPassed = true;
                    }
                }
                else
                {
                    ori += 2 * M_PI;
                    if (ori < endOri - M_PI * 3 / 2)
                    {
                        ori += 2 * M_PI;
                    }
                    else if (ori > endOri + M_PI / 2)
                    {
                        ori -= 2 * M_PI;
                    }
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                //point.intensity = scanID + scanPeriod * relTime;
                laserCloudScans[scanID].push_back(point); 
            }
            
            cloudSize = count;
            //printf("points size %d \n", cloudSize);

            //pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
            laserCloud -> clear();
            for (int i = 0; i < N_SCAN; i++)
            { 
                scanStartInd[i] = laserCloud->size() + 5;
                *laserCloud += laserCloudScans[i];
                scanEndInd[i] = laserCloud->size() - 6;
            }
        } 

        void calculateSmoothness(){

            int cloudSize = laserCloud->points.size();
            //cout << "Cloud Size: " << cloudSize << endl;

            for (int i = 5; i < cloudSize - 5; i++){ 
                float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;

                float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;

                float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

                cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
                cloudNeighborPicked[i] = 0;
                cloudLabel[i] = 0;
                cloudSortInd[i] = i;

                //cloudSmoothness[i].value = cloudCurvature[i];
                //cloudSmoothness[i].ind = i;
            }
        }

        void extractFeatures(){

            cornerPointsSharp.clear();
            cornerPointsLessSharp.clear();
            surfPointsFlat.clear();
            surfPointsLessFlat.clear();

            for (int i = 0; i < N_SCAN; i++)
            {
                if( scanEndInd[i] - scanStartInd[i] < 6)
                    continue;
                pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
                for (int j = 0; j < 6; j++)
                {
                    int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
                    int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;


                    //auto comp_func = std::bind(&FeatureExtraction::comp, this, std::placeholders::_1, std::placeholders::_2);
                    std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
                    //std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep+1, by_value());

                    int largestPickedNum = 0;
                    for (int k = ep; k >= sp; k--)
                    {
                        int ind = cloudSortInd[k]; 
                        //int ind = cloudSmoothness[k].ind;

                        if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] > 0.1)
                        {

                            largestPickedNum++;
                            if (largestPickedNum <= 2)
                            {                        
                                cloudLabel[ind] = 2;
                                cornerPointsSharp.push_back(laserCloud->points[ind]);
                                cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                            }
                            else if (largestPickedNum <= 20)
                            {                        
                                cloudLabel[ind] = 1; 
                                cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                            }
                            else
                            {
                                break;
                            }

                            cloudNeighborPicked[ind] = 1; 

                            for (int l = 1; l <= 5; l++)
                            {
                                float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                                float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                                float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                {
                                    break;
                                }

                                cloudNeighborPicked[ind + l] = 1;
                            }
                            for (int l = -1; l >= -5; l--)
                            {
                                float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                                float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                                float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                {
                                    break;
                                }

                                cloudNeighborPicked[ind + l] = 1;
                            }
                        }
                    }

                    int smallestPickedNum = 0;
                    for (int k = sp; k <= ep; k++)
                    {
                        int ind = cloudSortInd[k];
                        //int ind = cloudSmoothness[k].ind;

                        if (cloudNeighborPicked[ind] == 0 &&
                            cloudCurvature[ind] < 0.1)
                        {

                            cloudLabel[ind] = -1; 
                            surfPointsFlat.push_back(laserCloud->points[ind]);

                            smallestPickedNum++;
                            if (smallestPickedNum >= 4)
                            { 
                                break;
                            }

                            cloudNeighborPicked[ind] = 1;
                            for (int l = 1; l <= 5; l++)
                            { 
                                float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                                float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                                float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                {
                                    break;
                                }

                                cloudNeighborPicked[ind + l] = 1;
                            }
                            for (int l = -1; l >= -5; l--)
                            {
                                float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                                float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                                float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                {
                                    break;
                                }

                                cloudNeighborPicked[ind + l] = 1;
                            }
                        }
                    }

                    for (int k = sp; k <= ep; k++)
                    {
                        if (cloudLabel[k] <= 0)
                        {
                            surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                        }
                    }
                }

                pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
                pcl::VoxelGrid<PointType> downSizeFilter;
                downSizeFilter.setInputCloud(surfPointsLessFlatScan);
                downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
                downSizeFilter.filter(surfPointsLessFlatScanDS);

                surfPointsLessFlat += surfPointsLessFlatScanDS;
            }

        }

        void publishFeatureClouds(){
            //freeCloudInfoMemory();

            //sensor_msgs::PointCloud2 tempCloud;

            //tempCloud = publishCloud(pub_cornerPointsSharp, &cornerPointsSharp, cloudHeader.stamp, odomFrame);
            //tempCloud = publishCloud(pub_cornerPointsLessSharp, &cornerPointsLessSharp, cloudHeader.stamp, odomFrame);
            //tempCloud = publishCloud(pub_surfacePointsFlat, &surfPointsFlat, cloudHeader.stamp, odomFrame);
            //tempCloud = publishCloud(pub_surfacePointsLessFlat, &surfPointsLessFlat, cloudHeader.stamp, odomFrame);
            sensor_msgs::PointCloud2 laserCloudOutMsg;
            pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "camera_init";
            pub_lidarCloud_projectDeskew .publish(laserCloudOutMsg);

            sensor_msgs::PointCloud2 cornerPointsSharpMsg;
            pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
            cornerPointsSharpMsg.header.stamp = cloudHeader.stamp;
            cornerPointsSharpMsg.header.frame_id = "camera_init";
            pub_cornerPointsSharp.publish(cornerPointsSharpMsg);

            sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
            pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
            cornerPointsLessSharpMsg.header.stamp = cloudHeader.stamp;
            cornerPointsLessSharpMsg.header.frame_id = "camera_init";
            pub_cornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

            sensor_msgs::PointCloud2 surfPointsFlat2;
            pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
            surfPointsFlat2.header.stamp = cloudHeader.stamp;
            surfPointsFlat2.header.frame_id = "camera_init";
            pub_surfacePointsFlat.publish(surfPointsFlat2);

            sensor_msgs::PointCloud2 surfPointsLessFlat2;
            pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
            surfPointsLessFlat2.header.stamp = cloudHeader.stamp;
            surfPointsLessFlat2.header.frame_id = "camera_init";
            pub_surfacePointsLessFlat.publish(surfPointsLessFlat2);
        }

        //void freeCloudInfoMemory(){
        //    cloudInfo.startRingIndex.clear();
        //    cloudInfo.endRingIndex.clear();
        //    cloudInfo.pointColInd.clear();
        //    cloudInfo.pointRange.clear();
        //}

};

int main(int argc, char** argv){
    ros::init(argc, argv, "tar_sam");

    FeatureExtraction featureExtraction;

    ROS_INFO("\033[1;36m\n ---> Module: \033[0;36m Feature Extraction \033[1;36m Started <--- \033[0m");
   
    ros::spin();

    return 0;
}
