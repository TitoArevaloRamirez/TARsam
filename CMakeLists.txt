cmake_minimum_required(VERSION 3.0.2)
project(tar_sam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  #basic
  roscpp
  rospy
  cv_bridge
  tf
  #msgs
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  message_generation
  visualization_msgs
  message_filters
  #pcl 
  pcl_conversions
)

find_package(OpenMP REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED HINTS /home/usr3/Code/Opencv_latest/opencv_install)
find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED QUIET)

################################################
## Declare ROS messages, services and actions ##
################################################
add_message_files(
  DIRECTORY msg
  FILES
  cloud_info.msg
)


generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
)


################################################
## Declare ROS dynamic reconfigure parameters ##
################################################


###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  DEPENDS EIGEN3 PCL GTSAM
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs sensor_msgs message_runtime message_generation visualization_msgs message_filters
  
)

###########
## Build ##
###########
include_directories(
    include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
	${GTSAM_INCLUDE_DIR}
)
link_directories(
	include
	${PCL_LIBRARY_DIRS}
    ${OpenCV_LIBRARY_DIRS}
    ${GTSAM_LIBRARY_DIRS}
)

##Project and Deskew the raw LiDAR point cloud
#add_executable(${PROJECT_NAME}_projectDeskew src/project_deskew.cpp)
#add_dependencies(${PROJECT_NAME}_projectDeskew ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
#target_link_libraries(${PROJECT_NAME}_projectDeskew ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
#
##Extract features from the projected and deskewed LiDAR point cloud
#add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
#add_dependencies(${PROJECT_NAME}_featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
#target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME}_featureExtraction src/scanRegistration.cpp)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Fast Odometry Graph 
add_executable(${PROJECT_NAME}_fastOdomGraph src/fastOdomGraph.cpp)
target_link_libraries(${PROJECT_NAME}_fastOdomGraph ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES} gtsam)
add_dependencies(${PROJECT_NAME}_fastOdomGraph ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)

# Mapping Optimization
add_executable(${PROJECT_NAME}_mapOptimization src/mapOptimization.cpp)
add_dependencies(${PROJECT_NAME}_mapOptimization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_compile_options(${PROJECT_NAME}_mapOptimization PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_mapOptimization ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenMP_CXX_FLAGS} gtsam)

#############
## Install ##
#############


#############
## Testing ##
#############

