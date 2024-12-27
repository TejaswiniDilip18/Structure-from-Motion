#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <map>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <vector>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>

namespace fs = std::filesystem;

using namespace std;

const int IMAGE_DOWNSAMPLE = 1; // downsample images for faster processing
const double FOCAL_LENGTH = 2559.68 / IMAGE_DOWNSAMPLE; // focal length of camera in pixels (approx)
const int MIN_LANDMARK_SEEN = 3; // minimum number of camera views a landmark must be seen in to be considered

struct sfm_Helper{

    struct ImagePose{
        cv::Mat img; // downsampled image
        cv::Mat desc; // descriptors
        std::vector<cv::KeyPoint> kp; // keypoints

        cv::Mat T; // 4x4 transformation matrix
        cv::Mat P; // 3x4 projection matrix

        // alias to clarify map usage below
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // seypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
    };

    // 3D point
    struct Landmark{
        cv::Point3f pt;
        int seen = 0; // number of times this point is seen
        cv::Vec3f color;
    };
    std::vector<ImagePose> img_pose;
    std::vector<Landmark> landmark;
};

int main(){

    sfm_Helper sfm;

    // find matches
    {
        std::string imgDirectory = "../south-building/images/";
        std::string imgPrefix = "P1180";
        std::string imgFileType = ".JPG";
        int imgStartIndex = 141;
        int imgEndIndex = 225;

        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

        cv::namedWindow("img", cv::WINDOW_NORMAL);

        // extract features
        for (int imgIndex = imgStartIndex; imgIndex <= imgEndIndex; imgIndex++)
        {
            std::string imgFullFilename = imgDirectory + imgPrefix + std::to_string(imgIndex) + imgFileType;
            cv::Mat img = cv::imread(imgFullFilename);

            if (!img.empty()){  

                sfm_Helper::ImagePose a;
                cv::resize(img, img, img.size()/IMAGE_DOWNSAMPLE);
                a.img = img;
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

                akaze->detectAndCompute(img, cv::noArray(), a.kp, a.desc);

                sfm.img_pose.emplace_back(a);
            }
            else{
                std::cout << "Image " << imgFullFilename << " is empty!" << std::endl;
            }
        }

        std::cout << "Number of images = " << sfm.img_pose.size() << std::endl;

        // match features between images
        for(size_t i=0; i< sfm.img_pose.size()-1; i++){
            auto &img_pose_i = sfm.img_pose[i];
            
            // for(size_t j=i+1; j< sfm.img_pose.size(); j++){
            for(size_t j=i+1; j<= i+1; j++){
                auto &img_pose_j = sfm.img_pose[j];
                std::vector<std::vector<cv::DMatch>> matches;

                matcher->knnMatch(img_pose_i.desc, img_pose_j.desc, matches, 2, cv::noArray());

                std::vector<cv::Point2f> src, dst;
                std::vector<int> i_kp, j_kp;
                std::vector<uchar> mask;

                for(auto &m : matches){
                    if(m[0].distance < 0.7*m[1].distance){
                        src.push_back(img_pose_i.kp[m[0].queryIdx].pt);
                        dst.push_back(img_pose_j.kp[m[0].trainIdx].pt);

                        i_kp.push_back(m[0].queryIdx);
                        j_kp.push_back(m[0].trainIdx);
                    }
                }

                cv::findFundamentalMat(src, dst, cv::FM_RANSAC, 3.0, 0.99, mask);

                cv::Mat canvas = img_pose_i.img.clone();
                canvas.push_back(img_pose_j.img.clone());

                for(size_t k=0; k<mask.size(); k++){
                    if(mask[k]){
                        img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
                        img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

                        cv::line(canvas, src[k], dst[k] + cv::Point2f(0, img_pose_i.img.rows), cv::Scalar(0, 0, 255), 2);
                    }
                }

                int good_matches = cv::sum(mask)[0];
                assert(good_matches>=8);

                std::cout << "Feature matching " << i << " " << j << " ==> " << good_matches << "/" << matches.size() << std::endl;

                cv::resize(canvas, canvas, canvas.size()/2);

                cv::imshow("img", canvas);
                cv::waitKey(1);
            }
        }
    }
    // Recover motion between previous to current image and triangulate points
    {
        // Setup camera matrix

        // Principal point
        double cx = sfm.img_pose[0].img.size().width/2;
        double cy = sfm.img_pose[0].img.size().height/2;

        cv::Point2d pp(cx, cy);

        // Calibration matrix
        cv::Mat K = cv::Mat::eye(3,3, CV_64F);

        K.at<double>(0,0) = FOCAL_LENGTH;
        K.at<double>(1,1) = FOCAL_LENGTH;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        cout<< '\n'<< "Initial Camera Matrix: "<< '\n' << K << endl;

        sfm.img_pose[0].T = cv::Mat::eye(4, 4, CV_64F);
        sfm.img_pose[0].P = K*cv::Mat::eye(3, 4, CV_64F);

        for(size_t i=0; i<sfm.img_pose.size()-1; i++){
            auto &prev = sfm.img_pose[i];
            auto &curr = sfm.img_pose[i+1];

            std::vector<cv::Point2f> src, dst;
            std::vector<size_t> kp_used;

            for(size_t k=0; k<prev.kp.size(); k++){
                if(prev.kp_match_exist(k, i+1)){
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    src.push_back(prev.kp[k].pt);
                    dst.push_back(curr.kp[match_idx].pt);

                    kp_used.push_back(k);
                }
            }

            if (src.size() < 8 || dst.size() < 8) {
                std::cout << "Not enough matches between images " << i << " and " << (i+1) << std::endl;
                continue;
            }

            cv::Mat mask;

            cv::Mat E = cv::findEssentialMat(dst, src, FOCAL_LENGTH, pp, cv::RANSAC, 0.999, 1.0, mask);
            cv::Mat local_R, local_t;

            cv::recoverPose(E, dst, src, local_R, local_t, FOCAL_LENGTH, pp, mask);

            // local transform
            cv::Mat T = cv::Mat::eye(4,4, CV_64F);
            local_R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
            local_t.copyTo(T(cv::Range(0,3), cv::Range(3,4))); 

            // accumulate transformation
            curr.T = prev.T*T;

            // projection matrix
            cv::Mat R = curr.T(cv::Range(0,3), cv::Range(0,3));
            cv::Mat t = curr.T(cv::Range(0,3), cv::Range(3,4));

            cv::Mat P(3,4,CV_64F);
            P(cv::Range(0,3), cv::Range(0,3)) = R.t();
            P(cv::Range(0,3), cv::Range(3, 4)) = -R.t()*t;

            P = K*P;

            curr.P = P;

            cv::Mat points4D;
            cv::triangulatePoints(prev.P, curr.P, src, dst, points4D);

            // Scale the new 3d points to be similar to the existing 3d points (landmark)
            // Use ratio of distance between pairing 3d points
            if(i>0){
                double scale = 0;
                int count = 0;

                cv::Point3f prev_camera;

                prev_camera.x = prev.T.at<double>(0,3);
                prev_camera.y = prev.T.at<double>(1,3);
                prev_camera.z = prev.T.at<double>(2,3);

                std::vector<cv::Point3f> new_pts;
                std::vector<cv::Point3f> existing_pts;

                for(size_t j=0; j<kp_used.size(); j++){
                    size_t k = kp_used[j];
                    if(mask.at<uchar>(j) && prev.kp_match_exist(k, i+1) && prev.kp_3d_exist(k)){
                        cv::Point3f pt3d;

                        pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                        pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                        pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                        size_t idx = prev.kp_3d(k);
                        cv::Point3f avg_landmark = sfm.landmark[idx].pt / (sfm.landmark[idx].seen - 1);

                        new_pts.push_back(pt3d);
                        existing_pts.push_back(avg_landmark);             
                    }
                }

                for (size_t j = 0; j < new_pts.size() - 1; j++) {
                    for (size_t k = j + 1; k < new_pts.size(); k++) {
                        double norm_existing = norm(existing_pts[j] - prev_camera);
                        double norm_new = norm(new_pts[j] - prev_camera);

                        double s = norm_existing / norm_new;
                        scale += s;
                        count++;
                    }
                }

                assert(count>0);

                scale /= count;

                cout << "image " << (i+1) << " ==> " << i << " scale=" << scale << " count=" << count <<  endl;

                // apply scale and re-calculate T and P matrix
                local_t *= scale;

                // local tansform
                cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
                local_t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));

                // accumulate transform
                curr.T = prev.T*T;

                // make projection ,matrix
                R = curr.T(cv::Range(0, 3), cv::Range(0, 3));
                t = curr.T(cv::Range(0, 3), cv::Range(3, 4));

                cv::Mat P(3, 4, CV_64F);
                P(cv::Range(0, 3), cv::Range(0, 3)) = R.t();
                P(cv::Range(0, 3), cv::Range(3, 4)) = -R.t()*t;
                P = K*P;

                curr.P = P;

                cv::triangulatePoints(prev.P, curr.P, src, dst, points4D);

            }

            // Find good triangulated points
            for (size_t j=0; j < kp_used.size(); j++) {
                if (mask.at<uchar>(j)) {
                    size_t k = kp_used[j];
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    cv::Point3f pt3d;

                    pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                    pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                    pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                    cv::Vec3b color = curr.img.at<cv::Vec3b>(curr.kp[match_idx].pt);
                    // normalize color
                    cv::Vec3f colorf = cv::Vec3f(color[0], color[1], color[2]) / 255.0;

                    if(prev.kp_3d_exist(k)){
                        curr.kp_3d(match_idx) = prev.kp_3d(k);
                       
                        sfm.landmark[prev.kp_3d(k)].pt += pt3d;
                        sfm.landmark[prev.kp_3d(k)].color += colorf;
                        sfm.landmark[curr.kp_3d(match_idx)].seen++;
                    }
                    else{
                        // Add a new 3D point
                        sfm_Helper::Landmark landmark;

                        landmark.pt = pt3d;
                        landmark.seen = 2;
                        landmark.color = colorf;

                        sfm.landmark.push_back(landmark);

                        prev.kp_3d(k) = sfm.landmark.size() - 1;
                        curr.kp_3d(match_idx) = sfm.landmark.size() - 1;
                    } 
                }
            }
        }
        // Average out the landmark 3d position
        for (auto &l : sfm.landmark) {
            if (l.seen >= 3) {
                l.pt /= (l.seen - 1);
                l.color /= (l.seen - 1);
            }
        }
    }


        // Run GTSAM bundle adjustment
        gtsam::Values result;
        {
            using namespace gtsam;

            // Model the camera intrinsics
            double cx = sfm.img_pose[0].img.size().width/2;
            double cy = sfm.img_pose[0].img.size().height/2;

            Cal3_S2 K(FOCAL_LENGTH, FOCAL_LENGTH, 0, cx, cy); // Assuming 0 Skew 
            noiseModel::Isotropic::shared_ptr measurement_noise = noiseModel::Isotropic::Sigma(2, 2.0); // pixel error in (x,y)

            // Initialize the Factor Graph and Variables
            NonlinearFactorGraph graph;
            Values initial;

            // Add camera poses to graph
            for(size_t i=0; i<sfm.img_pose.size(); i++){
                auto &img_pose = sfm.img_pose[i];

                Rot3 R(
                    img_pose.T.at<double>(0,0),
                    img_pose.T.at<double>(0,1),
                    img_pose.T.at<double>(0,2),
                    
                    img_pose.T.at<double>(1,0),
                    img_pose.T.at<double>(1,1),
                    img_pose.T.at<double>(1,2),

                    img_pose.T.at<double>(2,0),
                    img_pose.T.at<double>(2,1),
                    img_pose.T.at<double>(2,2)
                );

                Point3 t;

                t(0) = img_pose.T.at<double>(0,3);
                t(1) = img_pose.T.at<double>(1,3);
                t(2) = img_pose.T.at<double>(2,3);

                Pose3 pose(R,t);

                // Add prior for the first image
                if(i==0){
                    noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.1)).finished());
                    graph.emplace_shared<PriorFactor<Pose3>>(Symbol('x', 0), pose, pose_noise); // add directly to graph
                }

                initial.insert(Symbol('x', i), pose);

                // landmark seen
                for(size_t k=0; k<img_pose.kp.size(); k++){
                    if(img_pose.kp_3d_exist(k)){
                        size_t landmark_id = img_pose.kp_3d(k);

                        if (sfm.landmark[landmark_id].seen >= MIN_LANDMARK_SEEN)
                        {
                            Point2 pt;

                            pt(0) = img_pose.kp[k].pt.x;
                            pt(1) = img_pose.kp[k].pt.y;

                            graph.emplace_shared<GeneralSFMFactor2<Cal3_S2>>(pt, measurement_noise, Symbol('x', i), Symbol('l', landmark_id), Symbol('K', 0));
                        }
                    }
                }
            }

            // Add a prior on the calibration
            initial.insert(Symbol('K', 0), K);

            noiseModel::Diagonal::shared_ptr cal_noise = noiseModel::Diagonal::Sigmas((Vector(5) << 100, 100, 0.01 /*skew*/, 100, 100).finished());
            graph.emplace_shared<PriorFactor<Cal3_S2>>(Symbol('K', 0), K, cal_noise);

             // Initialize estimate for landmarks
            bool init_prior = false;

            for (size_t i=0; i < sfm.landmark.size(); i++) {
                if (sfm.landmark[i].seen >= MIN_LANDMARK_SEEN) {
                    cv::Point3f &p = sfm.landmark[i].pt;

                    initial.insert<Point3>(Symbol('l', i), Point3(p.x, p.y, p.z));

                    if (!init_prior) {
                        init_prior = true;

                        noiseModel::Isotropic::shared_ptr point_noise = noiseModel::Isotropic::Sigma(3, 0.1);
                        Point3 p(sfm.landmark[i].pt.x, sfm.landmark[i].pt.y, sfm.landmark[i].pt.z);
                        graph.emplace_shared<PriorFactor<Point3>>(Symbol('l', i), p, point_noise);
                    }
                }
            }

            result = LevenbergMarquardtOptimizer(graph, initial).optimize();

            cout << endl;
            cout << "initial graph error = " << graph.error(initial) << endl;
            cout << "final graph error = " << graph.error(result) << endl;            
        

            // Save optimized 3D points to a file
            ofstream out("../3d_points_after_ba.txt");
            for (size_t i = 0; i < sfm.landmark.size(); i++) {
                if (sfm.landmark[i].seen >= MIN_LANDMARK_SEEN) {
                    // Check if the landmark exists in the result
                    if (result.exists(Symbol('l', i))) {
                        Point3 optimized_point = result.at<Point3>(Symbol('l', i)); // Get optimized landmark position
                        out << optimized_point.x() << " " << optimized_point.y() << " " << optimized_point.z() << " ";
                        out << sfm.landmark[i].color[0] << " " << sfm.landmark[i].color[1] << " " << sfm.landmark[i].color[2] << endl;
                    }
                }
            }
            out.close();
        }

    return 0;
}