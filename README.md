# Structure from Motion (SfM)

This repository implements a Structure from Motion (SfM) pipeline to reconstruct a 3D scene from a sequence of 2D images using computer vision techniques and bundle adjustment.
---

## Features

- **3D Reconstruction:** Recover the 3D structure of a scene from a set of 2D images.
- **Bundle Adjustment:** Optimize camera poses and 3D points for higher accuracy using the GTSAM library.
- *Visualization:* Generate a 3D point cloud viewable with Open3D.
---

### **Dependencies**
Ensure the following are installed:

- **OpenCV**: For image processing and feature matching.
- **GTSAM**: For bundle adjustment.
- **C++17 or later**: To compile the code.
- **Python**: For 3D point cloud visualization (requires Open3D).
---

## Installation Steps

1. Clone this repository:
```sh
git clone https://github.com/TejaswiniDilip18/Structure-from-Motion.git
cd Structure-from-Motion/
```

2. Download the dataset:
- Download and unzip south-building.zip into the project directory
- The dataset is available [here](https://colmap.github.io/datasets.html).

3. Compile the project:
```sh
mkdir build && cd build
cmake ..
make
```

4. Run the program:
```sh
./SfM
```

5. Visualize the point cloud: Run the Python visualization script
```sh
python3 pcd_vis.py
```
---

## **Implementation Details**

The pipeline consists of four main steps:

1. **Image Preprocessing**:
   - Detect keypoints and compute descriptors using OpenCV.
   - Match features between image pairs.

2. **Camera Pose Estimation**:
   - Estimate relative camera poses using epipolar geometry.
   - Recover sparse 3D points.

3. **Bundle Adjustment**:
   - Refine camera poses and 3D points using GTSAM to minimize reprojection error.

4. **Visualization**:
   - Export the optimized 3D points and visualize the reconstructed scene using Open3D.
---


## **Results**
A reconstructed 3D model after bundle adjustment:

![SfM](results/sfm_BA.gif)
---

## **Dataset**

### **Details**
- **Name**: South Building
- **Location**: UNC Chapel Hill
- **Number of Images**: 128
- **Camera**: Consistent intrinsic parameters
- **Provider**: Christopher Zach

This dataset is part of the COLMAP datasets collection, which also includes Gerrard Hall, Graham Hall, and Person Hall. More details are available [here](https://colmap.github.io/datasets.html).
---

## **Acknowledgements**

- **Original Implementation**: This project draws inspiration from [Nghia Ho's SfM Example](https://github.com/nghiaho12/SFM_example.git).
- **Dataset**: South Building dataset from the [COLMAP datasets](https://colmap.github.io/datasets.html) collection, provided by Christopher Zach.
---
