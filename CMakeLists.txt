cmake_minimum_required( VERSION 3.10)
project(SfM)

find_package(OpenCV 4)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4 not found.")
   endif()

set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")

# find_package(Eigen3 REQUIRED)

find_package(GTSAM REQUIRED)

# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable( SfM sfm_with_BA.cpp)
target_link_libraries(${PROJECT_NAME} 
    gtsam             # Link against the core GTSAM library
   #  GTSAM::gtsam_unstable    # If using GTSAM unstable features
    ${OpenCV_LIBS}           # Link against OpenCV libraries
)