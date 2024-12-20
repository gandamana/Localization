/*
 *  Author : koseng (Lintang)
*/

#pragma once

#include <ros/ros.h>
#include <ros/package.h>

#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <geometry_msgs/Point.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <dynamic_reconfigure/server.h>
//#include <v9_ball_detector/BallDetectorConfig.h>
#include <v9_ball_detector/BallDetectorParamsConfig.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>

#include <yaml-cpp/yaml.h>

// BUAT LOKALISASI

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include "v9_ball_detector/v9_amcl.h"

#include "vision_utils/vision_common.h"
#include "vision_utils/LineTip.h"
#include "vision_utils/FieldBoundary.h"
#include "vision_utils//LUT.h"
#include "vision_utils/localization_utils.h"
#include "vision_utils/fitcircle.h"

#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Imu.h>

#include <robotis_math/robotis_math.h>

#include <string.h>
#include <fstream>
#include <sstream>
#include <immintrin.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
// #define DEBUG

using Gandamana::FRAME_HEIGHT;
using Gandamana::FRAME_WIDTH;
using Gandamana::POINTS_MAP_H;
using Gandamana::POINTS_MAP_W;
using Field::PENALTY_MARK_DISTANCE;
using Field::GOAL_WIDTH;

class BallDetector: public AMCL{
private:
    ros::NodeHandle nh_;

    cv_bridge::CvImagePtr cv_img_ptr_subs_;
    cv_bridge::CvImage cv_img_pubs_;

    image_transport::ImageTransport it_;
    image_transport::Subscriber it_subs_;
    image_transport::Publisher it_pubs_;

    void imageCallback(const sensor_msgs::ImageConstPtr &_msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo &_msg);
    void publishImage();

    ros::Publisher update_params_pub_;

    cv::Mat in_img_;
    cv::Mat out_img_;

    sensor_msgs::CameraInfo cam_info_msg_;
    ros::Subscriber cam_info_sub_;
    ros::Publisher cam_info_pub_;

    unsigned int img_encoding_;

    geometry_msgs::Point ball_pos_;
    ros::Publisher ball_pos_pub_;

    ros::Time stamp_;
    std::string frame_id_;

    cv::Mat& setInputImage();
    void setOutputImage(const cv::Mat &_out_img);

    std::string ball_config_path;
    v9_ball_detector::BallDetectorParamsConfig config_;
    dynamic_reconfigure::Server<v9_ball_detector::BallDetectorParamsConfig> param_server_;
    dynamic_reconfigure::Server<v9_ball_detector::BallDetectorParamsConfig>::CallbackType param_cb_;
    void paramCallback(v9_ball_detector::BallDetectorParamsConfig &_config, uint32_t level);

    //====================
    

    cv::Mat LUT_data;
    std::string LUT_dir;
    ros::Subscriber LUT_sub_;
    void lutCallback(const vision_utils::LUTConstPtr &_msg);

    void ballRefCallback(const sensor_msgs::ImageConstPtr &_msg);
    image_transport::Subscriber it_bf_sub_;
    cv_bridge::CvImagePtr cv_bf_ptr_sub_;
    cv::Mat ball_ref_;
    cv::MatND ball_ref_hist_;
    float checkTargetHistogram(cv::Mat _target_roi);
    class HistParam{
    public:
        int channels[2];
        int hist_size[2];
        float h_ranges[2];
        float s_ranges[2];
        float v_ranges[2];
        const float* ranges[2];
        HistParam():channels{0,1},hist_size{32,32},
            h_ranges{0,255},s_ranges{0,255},v_ranges{0,255},
            ranges{s_ranges,s_ranges}{

        }
    }hist_param_;

    static const float MIN_FIELD_CONTOUR_AREA;
    static const float MIN_CONTOUR_AREA;

    cv::Mat cvtMulti(const cv::Mat &_ball_ref);

    cv::Mat in_hsv_;
    cv::Mat thresh_image_;

    ros::Publisher field_boundary_pub_;
    std::pair<cv::Mat, vision_utils::FieldBoundary > getFieldImage(const cv::Mat &_segmented_green);

    cv::Mat segmentColor(cv::Mat &_segmented_green, cv::Mat &_inv_segmented_green, cv::Mat &_segmented_ball_color, cv::Mat &_segmented_white);

    void filterContourData(std::vector<cv::Mat> &divided_roi, cv::Point top_left_pt,
                           std::vector<Points > &selected_data, cv::Mat *debug_mat, int sub_mode);

    int frame_mode_;
    ros::Subscriber frame_mode_subs_;
    void frameModeCallback(const std_msgs::Int8::ConstPtr &_msg);

    ros::Subscriber save_param_subs_;
    void saveParamCallback(const std_msgs::Empty::ConstPtr &_msg);

    ros::Subscriber pred_status_sub_;
    bool pred_status_;
    void predStatusCb(const std_msgs::Bool::ConstPtr &_msg){
        pred_status_ = _msg->data;
    }

    //Localization Utility
    image_transport::Publisher it_sw_pub_;
    image_transport::Publisher it_inv_sg_pub_;
    void localizationInputEnhance(cv::Mat &_input);
    void publishLocalizationUtils(const cv::Mat &_segmented_white,
                                             const cv::Mat &_inv_segmented_green,
                                             vision_utils::FieldBoundary _field_boundary);
    cv_bridge::CvImage cv_sw_pub_;
    cv_bridge::CvImage cv_inv_sg_pub_;

    void lineTipCallback(const vision_utils::LineTipConstPtr &_msg);
    ros::Subscriber line_tip_sub_;
    vision_utils::LineTip line_tip_;

    //Prediction Part
    Points regression_data_;
    std::vector<cv::Mat > est_trajectory_;
    std::vector<cv::Mat> getBallPosPrediction(const Points &_data);

    static const int MIN_LINE_INLIERS;
    static const int MIN_CIRCLE_INLIERS;
    static const int MIN_LINE_LENGTH;
    static const int MAX_LINE_MODEL;

    //test
    void integralImage(const cv::Mat &_input);

    //==============================AMCL LOKALISASI==============================
    inline float panAngleDeviation(float _pixel_x_pos);
    inline float tiltAngleDeviation(float _pixel_y_pos);
    inline float verticalDistance(float _tilt_dev);
    inline float horizontalDistance(float _distance_y, float _offset_pan);
    bool params_req_;
    
    Eigen::Affine3d NECK;
    Eigen::Affine3d HEAD;
    Eigen::Affine3d CAMERA_POSE;
    Eigen::Vector3d CAMERA_DIRECTION;
    Eigen::Vector3d CAMERA_ORIENTATION;
    Points target_points_return = {
        cv::Point(230, 428),
        cv::Point(235, 427),
        cv::Point(240, 424),
        cv::Point(250, 242),
        cv::Point(260, 235),
        cv::Point(270, 414),
        cv::Point(292, 259),
        cv::Point(324, 127),
        cv::Point(330, 183),
        cv::Point(350, 385),
        cv::Point(355, 385),
        cv::Point(367, 159),
        cv::Point(371, 155),
        cv::Point(373, 151),
        cv::Point(380, 207),
        cv::Point(381, 163),
        cv::Point(384, 139),
        cv::Point(390, 212),
        cv::Point(400, 215),
        cv::Point(400, 267),
        cv::Point(500, 246)
    };

    ros::Subscriber camera_info_sub_;
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr &_msg){
        camera_info_.D = _msg->D;
        camera_info_.K = _msg->K;
    }

    ros::Publisher particles_state_pub_;
    ros::Publisher features_pub_;
    ros::Publisher robot_state_pub_;
    ros::Publisher projected_ball_pub_;
    ros::Publisher projected_ball_stamped_pub_;

    ros::Subscriber js_sub_;
    ros::Subscriber imu_sub_;
    void imuCallback (const sensor_msgs::ImuConstPtr &imu_sub_);
    void jsCallback (const sensor_msgs::JointStatePtr &js_sub_);

    void sampleMotionModelOdometry(Particles &_particles_state,const geometry_msgs::Pose2D &_odometer_out);
    void measurementModel(Particles &_particle_state, vision_utils::Features _features_arg,float &_weight_avg);
    inline void arrangeTargetPoints(Points &_target_points);
    void initializeFieldFeaturesData();
    void genRadialPattern();
    void initializeFK(){
        NECK = Eigen::Affine3d(Eigen::Translation3d(Eigen::Vector3d(Gandamana::NECK_X, Gandamana::NECK_Y, Gandamana::NECK_Z)));
        HEAD = Eigen::Affine3d(Eigen::Translation3d(Eigen::Vector3d(Gandamana::NECK2HEAD_X, Gandamana::NECK2HEAD_Y, Gandamana::NECK2HEAD_Z)));
    }
    std::vector<std::vector<cv::Point2f> > landmark_pos_;
    std::vector<cv::Vec4f> line_segment_pos_; 

    vision_utils::FieldBoundary field_boundary_;

    void forwardKinematics(){
        NECK.linear() = Eigen::Matrix3d::Identity();
        HEAD.linear() = Eigen::Matrix3d::Identity();

        NECK.rotate(Eigen::AngleAxisd(roll_compensation_,Eigen::Vector3d(1,0,0)) *
                    Eigen::AngleAxisd(offset_head_,Eigen::Vector3d(0,1,0)));
                    // Eigen::AngleAxisd(0,Eigen::Vector3d(0,0,1)));
        HEAD.rotate(Eigen::AngleAxisd(pan_servo_angle_,Eigen::Vector3d(0,0,1)) *
                    Eigen::AngleAxisd(tilt_servo_angle_,Eigen::Vector3d(0,1,0)));
        CAMERA_POSE = NECK * HEAD;
        CAMERA_DIRECTION = CAMERA_POSE.translation();
        CAMERA_ORIENTATION = robotis_framework::convertRotationToRPY(CAMERA_POSE.linear());

            // std::cout << "R : " << roll_compensation_ * Math::RAD2DEG << " ; P : " << offset_head_ * Math::RAD2DEG << " ; Y : " << gy_heading_ << std::endl;
            // std::cout << "Robot Height : " << robot_height_ + CAMERA_DIRECTION.coeff(2) << std::endl;
            // std::cout << "PAN : " << pan_servo_angle_* Math::RAD2DEG << " ; TILT : " << tilt_servo_angle_ * Math::RAD2DEG << std::endl;
            // std::cout << "Camera Orientation : " << CAMERA_ORIENTATION * Math::RAD2DEG << std::endl;
            // std::cout << "Camera Direction : " << CAMERA_DIRECTION * 100 << std::endl;
    }

    float robot_height_;
    double shift_const_;
    float pan_servo_angle_;
    float tilt_servo_angle_;
    double pan_servo_offset_,tilt_servo_offset_;
    double H_FOV;
    double V_FOV;
    float TAN_HFOV_PER2;
    float TAN_VFOV_PER2;
    bool lost_features_;
    double z_offset_;
    double pan_rot_comp_;
    float offset_head_;
    float roll_compensation_;
    float gy_heading_;
    float last_gy_heading_;

    Eigen::Vector3d imu_data_;
    sensor_msgs::CameraInfo camera_info_;

    bool reset_particles_req_;
    double front_fall_limit_, behind_fall_limit_, right_fall_limit_, left_fall_limit_;
    double roll_offset_,pitch_offset_,yaw_offset_;
    double tilt_limit_;
    bool attack_dir_;
    double circle_cost;
    double inlier_error;
    double fx_;
    double fy_;
    std::vector<std::pair<int,int> > radial_pattern_;
 
    float calcShootDir(const cv::Point2f &_ball_pos);

    void getFeaturesModels(Points &_target_points, Vectors3 &_line_models, vision_utils::Features &_features_arg);
    void getFeaturesModels(Points &_target_points,Vectors3 &_line_models, vision_utils::Features &_features, vision_utils::LineTip &_line_tip);

    Points pointsProjection(const Points &_points, bool ball=false);

    void publishData();

public:
    BallDetector();
    void process();
    void processBaru();
    void saveParam();
    void loadParam();
};
