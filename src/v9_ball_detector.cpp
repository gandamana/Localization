#include "v9_ball_detector/v9_ball_detector.h"
#include <immintrin.h>

//TO RO LIST
/*
    1. Improvisasi deteksi garis
*/

#define FROM_VIDEO 0

const float BallDetector::MIN_CONTOUR_AREA = 100.0f;//100 -> Lomba ; 200 -> Bengkel
const float BallDetector::MIN_FIELD_CONTOUR_AREA = 1600.0f; //default: 1600.0f
// LOKALISASI
const int BallDetector::MIN_LINE_INLIERS = 5;
const int BallDetector::MIN_CIRCLE_INLIERS = 30;
const int BallDetector::MIN_LINE_LENGTH = 60;
const int BallDetector::MAX_LINE_MODEL = 10;
BallDetector::BallDetector():
    nh_(ros::this_node::getName()),
    it_(this->nh_),
    it_subs_(it_.subscribe("image_in", 1, &BallDetector::imageCallback, this)),
    it_pubs_(it_.advertise("image_out", 100)), //pub - aman
    cam_info_sub_(nh_.subscribe("camera_info_in", 100, &BallDetector::cameraInfoCallback, this)),
    cam_info_pub_(nh_.advertise<sensor_msgs::CameraInfo>("camera_info_out", 100)), //pub 
    frame_mode_subs_(nh_.subscribe("frame_mode", 1, &BallDetector::frameModeCallback, this)),
    save_param_subs_(nh_.subscribe("save_param", 1, &BallDetector::saveParamCallback, this)),
    LUT_sub_(nh_.subscribe("LUT_data", 1, &BallDetector::lutCallback, this)),
    it_bf_sub_(it_.subscribe("ball_ref", 1, &BallDetector::ballRefCallback, this)),
    it_sw_pub_(it_.advertise("segment_white", 10)), //pub
    it_inv_sg_pub_(it_.advertise("inv_segment_green", 10)), //pub
    line_tip_sub_(nh_.subscribe("line_tip",1,&BallDetector::lineTipCallback, this)),
    field_boundary_pub_(nh_.advertise<vision_utils::FieldBoundary > ("field_boundary", 10)), //pub
    ball_pos_pub_(nh_.advertise<geometry_msgs::Point > ("ball_pos", 100)), //pub
    update_params_pub_(nh_.advertise<std_msgs::Empty > ("update_params", 10)), //pub
    pred_status_sub_(nh_.subscribe("/alfarobi/prediction_status", 1, &BallDetector::predStatusCb, this)),
    frame_mode_(0),
    //===============================
    //         LOKALISASI START
    //==============================
    //sub
    js_sub_(nh_.subscribe("/robotis/goal_joint_states",1,&BallDetector::jsCallback,this)),
    camera_info_sub_(nh_.subscribe("camera_info_sub_",1,&BallDetector::cameraInfoCallback,this)),
    imu_sub_(nh_.subscribe("/robotis_op3/robotis/open_cr/imu",1,&BallDetector::imageCallback,this)),
    //pub
    robot_state_pub_(nh_.advertise<geometry_msgs::PoseStamped > ("robot_state", 10)),
    particles_state_pub_(nh_.advertise<vision_utils::Particles> ("particles_state", 10)),
    features_pub_(nh_.advertise<vision_utils::Features > ("field_features", 10)),
    projected_ball_pub_(nh_.advertise<geometry_msgs::Point>("projected_ball",10)),
    projected_ball_stamped_pub_(nh_.advertise<geometry_msgs::PointStamped >("stamped_projected_ball", 10)),
    //==============================
    reset_particles_req_(false),
    lost_features_(true),
    robot_height_(48.0f),
    gy_heading_(.0f),
    last_gy_heading_(.0f),
    imu_data_{.0, .0, .0},
    front_fall_limit_(.0),behind_fall_limit_(.0),
    right_fall_limit_(.0),left_fall_limit_(.0),
    pred_status_(false){
    
    nh_.param<double>("H_FOV",H_FOV, 61.25);
    nh_.param<double>("V_FOV",V_FOV, 47.88);
    nh_.param<double>("circle_cost", circle_cost, 6.0);
    nh_.param<double>("inlier_error", inlier_error, 1.0);
    nh_.param<double>("fx",fx_, 540.552005478);
    nh_.param<double>("fy",fy_, 540.571602012);
    nh_.param<double>("roll_offset",roll_offset_, .0);
    nh_.param<double>("pitch_offset",pitch_offset_, .0);
    nh_.param<double>("yaw_offset",yaw_offset_, .0);
    nh_.param<double>("tilt_limit",tilt_limit_, 30.0);
    nh_.param<double>("z_offset", z_offset_, .0);
    nh_.param<double>("pan_rot_comp", pan_rot_comp_, .0);
    nh_.param<double>("shift_const", shift_const_, -240.0);
    nh_.param<bool>("attack_dir", attack_dir_, false);

    roll_offset_ *= Math::DEG2RAD;
    pitch_offset_ *= Math::DEG2RAD;
    tilt_limit_ *= Math::DEG2RAD;
    H_FOV *= Math::DEG2RAD;
    V_FOV *= Math::DEG2RAD;
    pan_rot_comp_ *= Math::DEG2RAD;

    TAN_HFOV_PER2 = tan(H_FOV * 0.5);
    TAN_VFOV_PER2 = tan(V_FOV * 0.5);
    
    nh_.param<std::string>("ball_config_path", ball_config_path,
                           ros::package::getPath("v9_ball_detector") + "/config/saved_config.yaml");
    params_req_ = false;
    LUT_dir = ros::package::getPath("v9_ball_detector") + "/config/tabel_warna.xml";

    //===============================
    //         LOKALISASI
    //==============================
    loadParam();
    initializeParticles();
    initializeFieldFeaturesData();    
    genRadialPattern();
    initializeFK();

}

//===============================
//         LOKALISASI
//==============================
void BallDetector::jsCallback(const sensor_msgs::JointStatePtr &js_sub_) {
    pan_servo_angle_ = (js_sub_->position[0] * -1) + pan_servo_offset_;
    tilt_servo_angle_ = js_sub_->position[1] - tilt_servo_offset_;
}
void BallDetector::imuCallback(const sensor_msgs::ImuConstPtr &imu_sub_){
    Eigen::Quaterniond orientation;
    orientation.x() = imu_sub_->orientation.x;
    orientation.y() = imu_sub_->orientation.y;
    orientation.z() = imu_sub_->orientation.z;
    orientation.w() = imu_sub_->orientation.w;
    imu_data_ = robotis_framework::convertQuaternionToRPY(orientation);
    roll_compensation_ = -imu_data_.coeff(0) + roll_offset_;
    offset_head_ = imu_data_.coeff(1) + pitch_offset_;
//    offset_head_ = imu_data_.coeff(1);
    gy_heading_ = -imu_data_.coeff(2) * Math::RAD2DEG;
    if(gy_heading_ < 0)gy_heading_ = 360.0f + gy_heading_;
    odometer_out_.theta = (gy_heading_ - last_gy_heading_)*Math::DEG2RAD;
    std::cout <<  odometer_out_.theta << std::endl;
    last_gy_heading_ = gy_heading_;
}
//===============================
//         LOKALISASI
//===============================

void BallDetector::loadParam(){
    YAML::Node config_file;
    try{
        config_file = YAML::LoadFile(ball_config_path.c_str());
        // ROS_INFO("Cek file");
    }catch(const std::exception &e){
        ROS_ERROR("[v9_ball_detector] Unable to open config file: %s", e.what());
    }
    params_.num_particles = config_file["num_particles"].as<int>();
    params_.range_var = config_file["range_var"].as<double>();
    params_.beam_var = config_file["beam_var"].as<double>();
    params_.gy_var = config_file["gy_var"].as<double>();
    params_.alpha1 = config_file["alpha1"].as<double>();
    params_.alpha2 = config_file["alpha2"].as<double>();
    params_.alpha3 = config_file["alpha3"].as<double>();
    params_.alpha4 = config_file["alpha4"].as<double>();
    params_.short_term_rate = config_file["short_term_rate"].as<double>();
    params_.long_term_rate = config_file["long_term_rate"].as<double>();
    // std::cout << params_.num_particles << std::endl;
    // std::cout << params_.range_var << std::endl;
    // std::cout << params_.beam_var << std::endl;
    config_.score = config_file["score"].as<int>();
    config_.cost = config_file["cost"].as<int>();
    // ROS_INFO("score: %d", config_.score);
    // ROS_INFO("cost: %d", config_.cost);
    cv::FileStorage fs(LUT_dir.c_str(),cv::FileStorage::READ);
    fs["Tabel_Warna"] >> LUT_data;
    fs.release();

    ball_ref_ = cv::imread(ros::package::getPath("v9_ball_detector") + "/config/ball_ref.jpg");
    ball_ref_ = cvtMulti(ball_ref_);
    if(!ball_ref_.empty()){
        cv::calcHist(&ball_ref_, 1, hist_param_.channels, cv::Mat(), ball_ref_hist_, 2, hist_param_.hist_size, hist_param_.ranges);
        cv::normalize(ball_ref_hist_,ball_ref_hist_, .0, 1.0, cv::NORM_MINMAX);
    }
}

void BallDetector::saveParam(){
    YAML::Emitter yaml_out;
    yaml_out << YAML::BeginMap;
    yaml_out << YAML::Key << "score" << YAML::Value << config_.score;
    yaml_out << YAML::Key << "cost" << YAML::Value << config_.cost;
    yaml_out << YAML::EndMap;
    std::ofstream file_out(ball_config_path.c_str());
    file_out << yaml_out.c_str();
    file_out.close();
    cv::FileStorage fs(LUT_dir.c_str(), cv::FileStorage::WRITE);
    fs << "Tabel_Warna" << LUT_data;
    fs.release();
    // ROS_INFO("cek LUT");

    cv::imwrite(ros::package::getPath("v9_ball_detector") + "/config/ball_ref.jpg", ball_ref_);
}

void BallDetector::frameModeCallback(const std_msgs::Int8::ConstPtr &_msg){
    frame_mode_ = _msg->data;
}

void BallDetector::saveParamCallback(const std_msgs::Empty::ConstPtr &_msg){
    (void)_msg;
    saveParam();
}

void BallDetector::ballRefCallback(const sensor_msgs::ImageConstPtr &_msg){
    cv_bf_ptr_sub_ = cv_bridge::toCvCopy(_msg,_msg->encoding);
    ball_ref_ = cv_bf_ptr_sub_->image;
    ball_ref_ = cvtMulti(ball_ref_);
    cv::calcHist(&ball_ref_, 1, hist_param_.channels, cv::Mat(), ball_ref_hist_, 2, hist_param_.hist_size, hist_param_.ranges);
    cv::normalize(ball_ref_hist_, ball_ref_hist_, .0, 1.0 , cv::NORM_MINMAX);
}

void BallDetector::lutCallback(const vision_utils::LUTConstPtr &_msg){
//    uchar* LUT_ptr = LUT_data.data;
    for(size_t i = 0; i < _msg->color.size(); i++){
        int h = (int)_msg->color[i].x;
        int s = (int)_msg->color[i].y;
        LUT_data.at<uchar>(h,s) = (int) _msg->color_class.data;
        //ROS_INFO("h: %i: ", (int) _msg->color_class.data);
    //    std::cout << h << " , " << s << " ; " << (int) _msg->color_class.data << std::endl;
    //    LUT_ptr[s + h*256] = (int) _msg->color[i].z;
    }
//    cv::Mat diff = tempor != LUT_data;
//    if(cv::countNonZero(diff)==0)std::cout << "SAMA" << std::endl;
}

void BallDetector::lineTipCallback(const vision_utils::LineTipConstPtr &_msg){
    line_tip_.tip1 = _msg->tip1;
    line_tip_.tip2 = _msg->tip2;
}

void BallDetector::imageCallback(const sensor_msgs::ImageConstPtr &_msg){

    try{
        img_encoding_ = Gandamana::GRAY8Bit;
        if(_msg->encoding.compare(sensor_msgs::image_encodings::MONO8))
            img_encoding_ = Gandamana::GRAY8Bit;
#if FROM_VIDEO == 0
        if(_msg->encoding.compare(sensor_msgs::image_encodings::BGR8))
            img_encoding_ = Gandamana::BGR8Bit;
#else
        if(_msg->encoding.compare(sensor_msgs::image_encodings::RGB8))
            img_encoding_ = Gandamana::BGR8Bit;
#endif
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("[v9_ball_detector] cv bridge exception: %s",e.what());
        return;
    }

    cv_img_ptr_subs_ = cv_bridge::toCvCopy(_msg,_msg->encoding);
    this->stamp_ = _msg->header.stamp;
    this->frame_id_ = _msg->header.frame_id;
}

void BallDetector::cameraInfoCallback(const sensor_msgs::CameraInfo &_msg){
    //cam_info_msg_ = *_msg;
    
//    ROS_INFO("CHECK...");
}

void BallDetector::paramCallback(v9_ball_detector::BallDetectorParamsConfig &_config, uint32_t level){
    (void)level;
    this->config_ = _config;
}

void BallDetector::publishImage(){
    cv_img_pubs_.image = out_img_.clone();

    //Stamp
    cv_img_pubs_.header.seq++;
    cv_img_pubs_.header.stamp = this->stamp_;
    cv_img_pubs_.header.frame_id = this->frame_id_;

    //microsoft lifecam brightness setting only work when the camera is capturing
    //setting first to zero brightness after first 2 frame then set to desired value
    //3 April 2019
    if(cv_img_pubs_.header.seq == 2){
        std_msgs::Empty empty_msg;
        update_params_pub_.publish(empty_msg);
    }else if(cv_img_pubs_.header.seq == 4){
        std_msgs::Empty empty_msg;
        update_params_pub_.publish(empty_msg);
    }

    switch(img_encoding_){
        case Gandamana::GRAY8Bit:cv_img_pubs_.encoding = sensor_msgs::image_encodings::MONO8;break;
        case Gandamana::BGR8Bit:cv_img_pubs_.encoding = sensor_msgs::image_encodings::RGB8;break;
        default:cv_img_pubs_.encoding = sensor_msgs::image_encodings::RGB8;break;
    }

    it_pubs_.publish(cv_img_pubs_.toImageMsg());
    cam_info_pub_.publish(cam_info_msg_);
}

void BallDetector::publishLocalizationUtils(const cv::Mat &_segmented_white,
                                         const cv::Mat &_inv_segmented_green,
                                         vision_utils::FieldBoundary _field_boundary){
    // cek validasi -hkm
    if(_segmented_white.empty() || _inv_segmented_green.empty()){
    ROS_ERROR("Segmentasi gaada");
    return;
    }
    //end validasi

    cv_sw_pub_.image = _segmented_white.clone();
    cv_inv_sg_pub_.image = _inv_segmented_green.clone();
    // cv_sw_pub_.image = _segmented_white;
    // cv_inv_sg_pub_.image = _inv_segmented_green;
    // std::cout<<_inv_segmented_green<<std::endl; // ada
    // std::cout<<cv_inv_sg_pub_<<std::endl;
    // std::cout << "cv_inv_sg_pub_ size: " << cv_inv_sg_pub_.image.size() << std::endl;
    cv_sw_pub_.header.seq++;
    cv_inv_sg_pub_.header.seq++;
    _field_boundary.header.seq++;
    // std::cout<<cv_inv_sg_pub_<<std::endl;

    cv_sw_pub_.header.stamp = this->stamp_;
    cv_inv_sg_pub_.header.stamp = this->stamp_;
    _field_boundary.header.stamp = this->stamp_;
    // std::cout<<cv_inv_sg_pub_<<std::endl;

    cv_sw_pub_.header.frame_id = this->frame_id_;
    cv_inv_sg_pub_.header.frame_id = this->frame_id_;
    _field_boundary.header.frame_id = this->frame_id_;

    cv_sw_pub_.encoding = sensor_msgs::image_encodings::MONO8;
    cv_inv_sg_pub_.encoding = sensor_msgs::image_encodings::MONO8;

    // std::cout << "cv_inv_sg_pub_ size: " << cv_inv_sg_pub_.encoding.size() << std::endl;
    it_sw_pub_.publish(cv_sw_pub_.toImageMsg());
    it_inv_sg_pub_.publish(cv_inv_sg_pub_.toImageMsg());
    field_boundary_pub_.publish(_field_boundary);
    // std::cout<<cv_inv_sg_pub_.toImageMsg()<<std::endl;
    // ROS_INFO("Berhasil Publish Segmentasi"); //hkm
}

cv::Mat& BallDetector::setInputImage(){
    return in_img_;
}

void BallDetector::setOutputImage(const cv::Mat &_out_img){
    out_img_ = _out_img.clone();
    cv::imshow("out_image function", out_img_);
}

std::pair<cv::Mat, vision_utils::FieldBoundary > BallDetector::getFieldImage(const cv::Mat &_segmented_green){
    cv::Mat _field_contour = cv::Mat::zeros(_segmented_green.size(), CV_8UC1);
    vision_utils::FieldBoundary field_boundary;
    Points contour_points;
    std::vector<Points > contours;
    std::vector<cv::Vec4i > hierarchy;

    cv::findContours(_segmented_green, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < contours.size(); i++){ 
        if(cv::contourArea(contours[i]) > MIN_FIELD_CONTOUR_AREA){
            // ROS_INFO("tessss");
            contour_points.insert(contour_points.end(), contours[i].begin(), contours[i].end());
        }
    }

    if(contour_points.size()){
        std::vector<Points > contour(1);
        cv::convexHull(contour_points,contour.front());
        cv::Rect field_bound = cv::boundingRect(contour.front());
        drawContours(_field_contour, contour, 0, cv::Scalar(255), CV_FILLED); //cv::Scalar(255)
        //[HW] Scan from dual direction
        for(int i = field_bound.tl().x;
            i < field_bound.br().x; i++){
            geometry_msgs::Vector3 temp;
            temp.x = i;
            temp.y = -1;
            temp.z = field_bound.br().y - 1;
            for(int j = field_bound.tl().y;
                j < field_bound.br().y; j++){
                if(_field_contour.at<uchar >(j, i) > 0 &&
                        temp.y == -1){
                    temp.y = j;
                }else if(_field_contour.at<uchar >(j, i) == 0 &&
                         temp.y != -1){
                    temp.z = j - 1;
                    break;
                }
            }
            field_boundary.bound1.push_back(temp);
        }

        for(int i = field_bound.tl().y;
            i < field_bound.br().y; i++){
            geometry_msgs::Vector3 temp;
            temp.x = i;
            temp.y = -1;
            temp.z = field_bound.br().x - 1;
            for(int j = field_bound.tl().x;
                j < field_bound.br().x; j++){
                if(_field_contour.at<uchar >(i, j) > 0 &&
                        temp.y == -1){
                    temp.y = j;
                }else if(_field_contour.at<uchar >(i, j) == 0 &&
                         temp.y != -1){
                    temp.z = j - 1;
                    break;
                }
            }
            field_boundary.bound2.push_back(temp);
        }
    }

    std::pair<cv::Mat, vision_utils::FieldBoundary > result;
    result.first = _field_contour;
    result.second = field_boundary;
    return result;
}

void BallDetector::localizationInputEnhance(cv::Mat &_input){
    cv::Mat result = _input.clone();
    cv::dilate(result, result ,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)), cv::Point(), 1);

    std::vector<Points > contours;
    std::vector<cv::Vec4i > hierarchy;
    std::vector<double > contours_area;

    cv::findContours(result, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    
    result = cv::Mat::zeros(result.size(),CV_8UC1);

    for(size_t i=0;i<contours.size();i++){
        contours_area.emplace_back(cv::contourArea(contours[i]));
        if(contours_area[i] > MIN_CONTOUR_AREA && hierarchy[i][3] == -1){
            Points approx_curve;
            cv::approxPolyDP(contours[i], approx_curve, 0.002*cv::arcLength(contours[i],true),true);
            std::vector<Points > target_contour;
            target_contour.push_back(approx_curve);
            drawContours(result, target_contour, 0, cv::Scalar(255), CV_FILLED);
        }
    }

    for(size_t i=0;i<contours.size();i++){
        if(contours_area[i] > MIN_CONTOUR_AREA && hierarchy[i][3] > -1){
            Points approx_curve;
            cv::approxPolyDP(contours[i], approx_curve, 0.002*cv::arcLength(contours[i],true),true);
            std::vector<Points > target_contour;
            target_contour.push_back(approx_curve);
            drawContours(result, target_contour, 0, cv::Scalar(0), CV_FILLED);
        }
    }
    _input = result.clone();
}

cv::Mat BallDetector::segmentColor(cv::Mat &_segmented_green, cv::Mat &_inv_segmented_green,
                                    cv::Mat &_segmented_ball_color, cv::Mat &_segmented_white){
    //validasi input -hkm
    if(in_img_.empty()){
    ROS_ERROR("Input image gaada!");
    return cv::Mat();
    }
    //end

    cv::Mat blank = cv::Mat::zeros(Gandamana::FRAME_HEIGHT, Gandamana::FRAME_WIDTH, CV_8UC1);
    cv::Mat out_segment = cv::Mat::zeros(Gandamana::FRAME_HEIGHT, Gandamana::FRAME_WIDTH, CV_8UC3);

    cv::Mat segmented_green = blank.clone();
    cv::Mat segmented_ball_color = blank.clone();
    cv::Mat segmented_white = blank.clone();

    cv::cvtColor(in_img_,in_hsv_,CV_BGR2HSV);
    // cv::imshow("in_hsv", in_hsv_);

    // cv::Mat gray;
    // cv::cvtColor(in_img_,gray,CV_BGR2GRAY);
    // cv::medianBlur(gray,gray,3);
    // cv::Mat kernel = (cv::Mat_<double>(3,3) << 0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111);
    // cv::filter2D(gray,gray,CV_8UC1,kernel);
    // cv::Mat lutable = cv::Mat(1,256,CV_8U);
    // uchar *lutable_ptr = lutable.ptr();
    // for(int i=0;i<256;i++){
    //     lutable_ptr[i] = pow(i/255.0,0.2)*255.0;
    // }
    // cv::Mat resulttt = gray.clone();
    // cv::LUT(gray,lutable,resulttt);
    // cv::equalizeHist(gray,gray);
    // cv::adaptiveThreshold(gray,gray,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,11,1);
    // cv::imshow("GRAY",gray);
    // // cv::waitKey(1);

    int num_cols = Gandamana::FRAME_WIDTH;
    int num_rows = Gandamana::FRAME_HEIGHT;

    auto LUT_ptr = LUT_data.data;
    // std::cout << "Length of LUT: " << LUT_data.size() << std::endl; //aman 
    for(int i = 0; i < num_rows; i++){
        cv::Vec3b* in_hsv_ptr = in_hsv_.ptr<cv::Vec3b>(i);
        cv::Vec3b* out_segment_ptr = out_segment.ptr<cv::Vec3b>(i);
        uchar* sg_ptr = segmented_green.ptr<uchar>(i);
        uchar* sbc_ptr = segmented_ball_color.ptr<uchar>(i);
        uchar* sw_ptr = segmented_white.ptr<uchar>(i);
        for(int j = 0; j < num_cols; j++){
        //    std::cout << i << " , " << j << " ; " << (int)in_hsv_ptr[j][0] << " , " << (int)in_hsv_ptr[j][1] << std::endl;
            // ROS_INFO("value: i:%i j:%i", (int)in_hsv_ptr[j][0],  (int)in_hsv_ptr[j][1]);
            // if(LUT_data.at<uchar>(in_hsv_ptr[j][0], in_hsv_ptr[j][1]) == 1){
            // uchar pres_class = LUT_ptr[in_hsv_ptr[j][1] + in_hsv_ptr[j][0]*num_cols];
            uchar pres_class = LUT_data.at<uchar>(in_hsv_ptr[j][0], in_hsv_ptr[j][1]);
            // ROS_INFO("press_class: %i", pres_class);
            if(pres_class == 1){
                sg_ptr[j] = 255;
                out_segment_ptr[j][0] = 0;
                out_segment_ptr[j][1] = 200;
                out_segment_ptr[j][2] = 0;                
            }else if(pres_class == 2){
                sbc_ptr[j] = 255;
                out_segment_ptr[j][0] = 0;
                out_segment_ptr[j][1] = 140;
                out_segment_ptr[j][2] = 255;
            }else if(pres_class == 3){
                sw_ptr[j] = 255;
                out_segment_ptr[j][0] = 255;
                out_segment_ptr[j][1] = 255;
                out_segment_ptr[j][2] = 255;
            }
        }
    }

    cv::Mat inv_segmented_green;
    cv::bitwise_not(segmented_green,inv_segmented_green);
    // cv::imshow("inv green pertama", inv_segmented_green); //hkm
    // std::cout<<segmented_green<<std::endl;
    // std::cout<<inv_segmented_green<<std::endl;

    localizationInputEnhance(segmented_white); //hkm - default diatas clone
    localizationInputEnhance(inv_segmented_green);

    _segmented_green = segmented_green.clone();
    _inv_segmented_green = inv_segmented_green.clone();
    _segmented_ball_color = segmented_ball_color.clone();
    _segmented_white = segmented_white.clone();

    // cv::imshow("inv green kedua", inv_segmented_green); //hkm

    // Validasi sebelum return -hkm
    if(_segmented_white.empty() || _inv_segmented_green.empty()){
        ROS_ERROR("Segmentation failed - empty result!");
    } else {
        ROS_DEBUG("Segmentation successful");
    } //end

    return out_segment;
}

void BallDetector::filterContourData(std::vector<cv::Mat> &divided_roi, cv::Point top_left_pt,
                       std::vector<Points > &selected_data, cv::Mat *debug_mat, int sub_mode = 0){
    int num_roi_cols = divided_roi.front().cols;
    int num_roi_rows = divided_roi.front().rows;
    bool horizon_scan = (float)num_roi_rows/(float)num_roi_cols < .75f;
    cv::Point map_origin[4];
    map_origin[0].x = top_left_pt.x;
    map_origin[0].y = top_left_pt.y;
    map_origin[1].x = (sub_mode == 2)?top_left_pt.x:top_left_pt.x + divided_roi.front().cols;
    map_origin[1].y = (sub_mode == 2)?top_left_pt.y + divided_roi.front().rows:top_left_pt.y;
    map_origin[2].x = top_left_pt.x;
    map_origin[2].y = top_left_pt.y + num_roi_rows;
    map_origin[3].x = top_left_pt.x + num_roi_cols;
    map_origin[3].y = top_left_pt.y + num_roi_rows;
    for(size_t idx = 0; idx < divided_roi.size(); idx++){

        int scan_mode = idx;

        switch(idx){
        case 0:scan_mode = (sub_mode == 1) ? 0 : (sub_mode == 2) ? 2 : horizon_scan ? 0 : 2;break;
        case 1:scan_mode = (sub_mode == 1) ? 1 : (sub_mode == 2) ? 3 : horizon_scan ? 1 : 2;break;
        case 2:scan_mode = horizon_scan ? 0 : 3;break;
        case 3:scan_mode = horizon_scan ? 1 : 3;break;
        }

        switch(scan_mode){
        case 0:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=0;j<num_roi_cols;j++){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 1:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=num_roi_cols-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==num_roi_cols-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 2:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=0;j<num_roi_rows;j++){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;
        case 3:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=num_roi_rows-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==num_roi_rows-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;

        }
    }
}

cv::Mat BallDetector::cvtMulti(const cv::Mat &_ball_ref){
    // cv::Mat hsv;
    cv::Mat yuv; // default
    // cv::Mat ycrcb;

//    cv::cvtColor(_ball_ref,hsv,CV_BGR2HSV);
    cv::cvtColor(_ball_ref,yuv,CV_BGR2YUV); //default
//    cv::cvtColor(_ball_ref,ycrcb,CV_BGR2YCrCb);
//    cv::Mat temp;
//    hsv.convertTo(hsv,CV_32F);
//    yuv.convertTo(yuv,CV_32F);
//    ycrcb.convertTo(ycrcb,CV_32F);
//    cv::multiply(hsv,yuv,temp);
//    cv::multiply(temp,ycrcb,temp);
//    cv::normalize(temp,temp,0,1,cv::NORM_MINMAX);
//    temp *=255;
//    temp.convertTo(temp,CV_8U);
    return yuv.clone(); //default
    // return hsv.clone();
    // return ycrcb.clone();

}

float BallDetector::checkTargetHistogram(cv::Mat _target_roi){

    if(ball_ref_.empty()){
        ROS_ERROR("[v9_ball_detector] Ball reference not found !!!");
        return -1;
    }
    _target_roi = cvtMulti(_target_roi);
    cv::MatND target_hist;
    cv::calcHist(&_target_roi, 1, hist_param_.channels, cv::Mat(), target_hist, 2, hist_param_.hist_size, hist_param_.ranges);
    cv::normalize(target_hist,target_hist, .0, 1.0, cv::NORM_MINMAX);

    return cv::compareHist(ball_ref_hist_, target_hist, 5);
}

std::vector<cv::Mat > BallDetector::getBallPosPrediction(const Points &_data){
    int total_smp = 0;
    int total_smp2 = 0;
    int total_smp3 = 0;
    int total_smp4 = 0;

    int total_x=0;
    int total_smpx=0;
    int total_smp2x=0;

    int total_y=0;
    int total_smpy=0;
    int total_smp2y=0;
    for(size_t i=0;i < _data.size();i++){

        total_smp += i;
        int smp2 = i*i;
        int smp3 = smp2*i;
        total_smp2 += smp2;
        total_smp3 += smp3;
        total_smp4 += smp3*i;

        total_x += _data[i].x;
        total_smpx += i*_data[i].x;
        total_smp2x += smp2*_data[i].x;

        total_y += _data[i].y;
        total_smpy += i*_data[i].y;
        total_smp2y += smp2*_data[i].y;

    }
    cv::Mat A = (cv::Mat_<double>(3,3) << total_smp4, total_smp3, total_smp2, total_smp3, total_smp2, total_smp, total_smp2, total_smp, _data.size());
    cv::Mat bx = (cv::Mat_<double>(3,1) << total_smp2x,total_smpx,total_x);
    cv::Mat by = (cv::Mat_<double>(3,1) << total_smp2y,total_smpy,total_y);
    cv::Mat A_inv = A.inv();
    cv::Mat xpoly_const = A_inv*bx;
    cv::Mat ypoly_const = A_inv*by;
    cv::Mat first_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(0), ypoly_const.at<double>(0));
    cv::Mat second_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(1), ypoly_const.at<double>(1));
    cv::Mat third_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(2), ypoly_const.at<double>(2));
    std::vector<cv::Mat > result;
    result.push_back(first_term);
    result.push_back(second_term);
    result.push_back(third_term);

    return result;

}

void BallDetector::process(){

    if(cv_img_ptr_subs_ == nullptr)return;
    // auto t1 = boost::chrono::high_resolution_clock::now();
    static geometry_msgs::Point last_ball_pos_; //Ang
    setInputImage() = cv_img_ptr_subs_->image; //Ang
    // std::cout << last_ball_pos_ << std::endl;
    // cv::medianBlur(in_img_,in_img_,3);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    cv::Mat output_view = in_img_.clone(); //Ang
    cv::Mat segmented_green,segmented_ball_color,segmented_white, inv_segmented_green; //Ang
    // cv::imshow("Segmented_green: ", segmented_green);
    cv::Mat thresh_image = segmentColor(segmented_green, inv_segmented_green, segmented_ball_color, segmented_white); //Ang
    // cv::imshow("thresh image", thresh_image);
    // cv::imshow("ball_ref",ball_ref_);
    // cv::Mat hsv, mask; //hkm
    // cv::cvtColor(in_img_, hsv, cv::COLOR_BGR2HSV); //hkm
    // cv::inRange(hsv, cv::Scalar(10,100,100), cv::Scalar(25,255,255), mask); //hkm
    // cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));//hkm
    // cv::dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));//hkm

//    integralImage(in_hsv_); //default

    // cv::imshow("SBC",segmented_ball_color);
    // std::cout<<inv_segmented_green<<std::endl; //bisa
    // std::cout<<segmented_white<<std::endl; //bisa
    cv::imshow("ISG",inv_segmented_green); //Ang
    cv::imshow("SW",segmented_white); //Ang
    // cv::imshow("Real Image ", output_view);
    // cv::imshow("Segmentasi Green ", segmented_green);
    cv::waitKey(1); //Ang

    cv::Mat field_contour; //Ang
    // cv::Mat field_contour_second;
    std::pair<cv::Mat, vision_utils::FieldBoundary > field_prop = getFieldImage(segmented_green); //Ang
    field_contour = field_prop.first; //Ang
//     publishLocalizationUtils(segmented_white,inv_segmented_green,field_prop.second); //Ang

//     cv::Mat ball_inside_field; //Ang
//     cv::bitwise_and(segmented_ball_color,field_contour,ball_inside_field); //Ang
//     // cv::imshow("ball inside field", ball_inside_field);
//     // cv::imshow("field contour", field_contour);
//     std::vector<Points > contours; //Ang
//     //Approx NONE to get the authentic contours
//     cv::findContours(ball_inside_field, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //Ang
//     cv::Mat ball_inside_field_copy = ball_inside_field.clone(); //Ang

   
//     int count = 0; //Ang

//     cv::Vec4f best_candidate(-1.0f, -1.0f , std::numeric_limits<float>::max(), std::numeric_limits<float>::max()); //Ang
//     cv::Mat ROI; //Ang
//     // cv::imshow("ROI",ROI); //cause error please stay comment this line
//     // cv::waitKey(1);
//     cv::Rect ball_roi; //Ang
//     // ROS_INFO("best_candidate: k1=%f k2=%f", best_candidate[0], best_candidate[1]);
//     for(size_t i = 0; i < contours.size(); i++){ //Ang
//         float contour_area = cv::contourArea(contours[i]); //Ang
//         // std::string text = "contour size : " + std::to_string(contour_area);
//         // cv::Point text_position(10,30);
//         cv::drawContours(ball_inside_field_copy, contours, i, cv::Scalar(255,0,3)); //debugging entar - alvi//Ang
//         // // cv::putText(
//         //     ball_inside_field_copy,
//         //     text,
//         //     text_position,
//         //     cv::FONT_HERSHEY_SIMPLEX,
//         //     1.0,
//         //     cv::Scalar(0,255,0),
//         //     2
//         // );
//         // cv::imshow("contour", ball_inside_field_copy); // -alvi
//         // ROS_INFO("kontur area sebelum if: %f", contour_area);
//         if(contour_area > MIN_CONTOUR_AREA){ //Ang
//             // ROS_INFO("Setelah : %f", contour_area);
//             cv::Rect rect_rough_roi = cv::boundingRect(contours[i]); //Ang

//             cv::Mat roi_hsv_ = cv::Mat(in_hsv_,rect_rough_roi); //Ang
//             //  cv::imshow("ROI_HSV",roi_hsv_);
//             //  cv::waitKey(1);

//             float histogram_score = checkTargetHistogram(roi_hsv_); //Ang
//             // ROS_INFO("Histogram Score: %f",histogram_score);
//             cv::Rect bbox = cv::boundingRect(contours[i]); //Ang
//             cv::rectangle(in_img_, bbox, cv::Scalar(0, 255, 0), 2); //Ang
// #ifdef DEBUG
//             // std::cout << "=============================================" << std::endl;
//             // std::cout << count << ". HIST SCORE : " << histogram_score << std::endl;
//             // std::cout << "CONTOUR AREA : " << contour_area << std::endl;
// #endif
//             if(histogram_score < (float)config_.score / 10.0f)continue; //Ang
 
//            cv::rectangle(output_view,rect_rough_roi,cv::Scalar(255,255,255),2); //Ang

//             cv::Point tl_pt = rect_rough_roi.tl(); //Ang
//            cv::Point br_pt = rect_rough_roi.br(); //Ang

//             cv::Mat frame_rough_roi(ball_inside_field, rect_rough_roi); //Ang

//             float roi_ratio = (rect_rough_roi.width < rect_rough_roi.height)?(float)rect_rough_roi.width/(float)rect_rough_roi.height: //Ang
//                                                                              (float)rect_rough_roi.height/(float)rect_rough_roi.width; //Ang
//             // ROS_INFO("roi_ratio: %f", roi_ratio);
//             cv::Vec4f circle_param(-1.0f, -1.0f , std::numeric_limits<float>::max(), std::numeric_limits<float>::max()); //Ang

//             std::vector<cv::Mat> sub_frame; //Ang
// #ifdef DEBUG
//             // std::cout << "ROI Ratio : " << roi_ratio << " ; Recip : " << 1.0f/roi_ratio << std::endl;
// #endif

//             if(roi_ratio >= 0.55f && 1.0f/roi_ratio <= 1.45f){ //Ang
//                 if(contour_area < 5000){ //Ang
//                     Points outer_circle; //Ang
//                     cv::convexHull(contours[i],outer_circle); //Ang
//                     cv::Moments moment; //Ang
//                     moment = cv::moments(outer_circle,true); //Ang
//                     cv::Point ball_com(moment.m10/moment.m00, moment.m01/moment.m00); //Ang
//                     cv::Point2f ctr; //Ang
//                     float radius; //Ang
//                     cv::minEnclosingCircle(outer_circle,ctr,radius); //Ang
//                     cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(outer_circle, Gandamana::FIT_CIRCLE_MAX_STEPS, Gandamana::FIT_CIRCLE_EPS);//Ang
// //                    sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
//                 //    std::cout << sub_circle_param << std::endl;
//                     //    ROS_INFO("sub_circle_param: %f", sub_circle_param[2]); //aman
//                 //     if(std::abs(ball_com.x - sub_circle_param[0]) >= 2)
//                 //         std::cout << "Param 1 : " << std::abs(ball_com.x - sub_circle_param[0]) << std::endl;
//                 //     if(std::abs(ball_com.y - sub_circle_param[1]) >= 2)
//                 //         std::cout << "Param 2 : " << std::abs(ball_com.y - sub_circle_param[1]) << std::endl;
//                 //     if(std::abs(sub_circle_param[0] - ctr.x) >= 2)
//                 //         std::cout << "Param 3 : " << std::abs(sub_circle_param[0] - ctr.x) << std::endl;
//                 //     if(std::abs(sub_circle_param[1] - ctr.y) >= 2)
//                 //         std::cout << "Param 4 : " << std::abs(sub_circle_param[1] - ctr.y) << std::endl;
//                 //     std::cout << "Contour2Circle : " << cv::contourArea(outer_circle)/(CV_PI*radius*radius) << std::endl;
//                     if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost*5 && //Ang
//                             std::fabs(ball_com.x - sub_circle_param[0]) <= 2 && //Ang
//                             std::fabs(ball_com.y - sub_circle_param[1]) <= 2 && //Ang
//                             std::fabs(sub_circle_param[0] - ctr.x) <= 2 &&  //Ang
//                             std::fabs(sub_circle_param[1] - ctr.y) <= 2 && //Ang
//                             cv::contourArea(outer_circle)/(Math::PI*radius*radius) > 0.75) //Ang
// //                            sub_circle_param[0] > tl_pt.x && sub_circle_param[0] < br_pt.x &&
// //                            sub_circle_param[1] > tl_pt.y && sub_circle_param[1] < br_pt.y)
//                         circle_param=sub_circle_param; //Ang
                    
//                 }else{ //Ang
//                     sub_frame.resize(4); //Ang
//                     sub_frame[0] = cv::Mat(frame_rough_roi,cv::Rect(0, 0, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1)); //Ang
//                     sub_frame[1] = cv::Mat(frame_rough_roi,cv::Rect(rect_rough_roi.width >> 1, 0, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1)); //Ang
//                     sub_frame[2] = cv::Mat(frame_rough_roi,cv::Rect(0, rect_rough_roi.height >> 1, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1)); //Ang
//                     sub_frame[3] = cv::Mat(frame_rough_roi,cv::Rect(rect_rough_roi.width >> 1, rect_rough_roi.height >> 1, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1)); //Ang

// //                    cv::line(output_view,cv::Point(tl_pt.x + rect_rough_roi.width/2,tl_pt.y),cv::Point(tl_pt.x + rect_rough_roi.width/2,tl_pt.y + rect_rough_roi.height),cv::Scalar(255,0,0),2);
// //                    cv::line(output_view,cv::Point(tl_pt.x,tl_pt.y+rect_rough_roi.height/2),cv::Point(tl_pt.x+rect_rough_roi.width,tl_pt.y+rect_rough_roi.height/2),cv::Scalar(255,0,0),2);

//                     std::vector<Points > selected_data(4); //Ang

//                     cv::Mat sub_sample[4]; //Ang
//                     sub_sample[0] = cv::Mat::zeros(sub_frame[0].size(), CV_8UC1); //Ang
//                     sub_sample[1] = cv::Mat::zeros(sub_frame[1].size(), CV_8UC1); //Ang
//                     sub_sample[2] = cv::Mat::zeros(sub_frame[2].size(), CV_8UC1); //Ang
//                     sub_sample[3] = cv::Mat::zeros(sub_frame[3].size(), CV_8UC1); //Ang

//                     filterContourData(sub_frame, tl_pt, selected_data, sub_sample, 0); //Ang

//                     // cv::imshow("CEKK1",sub_frame[0]);
//                     // cv::imshow("CEKK2",sub_frame[1]);
//                     // cv::imshow("CEKK3",sub_frame[2]);
//                     // cv::imshow("CEKK4",sub_frame[3]);

//                     for(int j = 0; j < 4; j++){ //Ang

//                         cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data[j], Gandamana::FIT_CIRCLE_MAX_STEPS, Gandamana::FIT_CIRCLE_EPS); //Ang

// //                        sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
// //                        int axis_dir = frame_rough_roi.cols > frame_rough_roi.rows?1:0;
// //                        cv::Vec2f interval_ctr = axis_dir?cv::Vec2f(tl_pt.x, br_pt.x):cv::Vec2f(tl_pt.y,br_pt.y);
// //                        std::cout << j << ". " << sub_circle_param << std::endl;
//                         if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost) //Ang
//                                 //sub_circle_param[axis_dir] > interval_ctr[0] && sub_circle_param[axis_dir] < interval_ctr[1])
// //                                sub_circle_param[0] > 1.0*(float)tl_pt.x && sub_circle_param[0] < 1.0*(float)br_pt.x &&
// //                                sub_circle_param[1] > 1.0*(float)tl_pt.y && sub_circle_param[1] < 1.0*(float)br_pt.y)
// //                            sub_circle_param[2] > (float)std::min(sub_frame[0].cols,sub_frame[0].rows) &&
// //                                    sub_circle_param[2] < 1.2*(float)std::max(sub_frame[0].cols,sub_frame[0].rows) )
//                             circle_param=sub_circle_param; //Ang
//                     }
//                 }

//             }else if(roi_ratio > 0.2f){ //Ang
                
//                 sub_frame.resize(2); //Ang
//                 int sub_mode=0; //Ang
//                 if(rect_rough_roi.width > rect_rough_roi.height){ //Ang
//                     sub_frame[0] = cv::Mat(frame_rough_roi, cv::Rect(0, 0, rect_rough_roi.width >> 1, rect_rough_roi.height)); //Ang
//                     sub_frame[1] = cv::Mat(frame_rough_roi, cv::Rect(rect_rough_roi.width >> 1, 0, rect_rough_roi.width >> 1, rect_rough_roi.height)); //Ang
//                     sub_mode=1; //Ang
// //                    cv::line(output_view,cv::Point(tl_pt.x+rect_rough_roi.width/2,tl_pt.y),cv::Point(tl_pt.x+rect_rough_roi.width/2,tl_pt.y+rect_rough_roi.height),cv::Scalar(255,0,0),2);
//                 }else{
//                     sub_frame[0] = cv::Mat(frame_rough_roi, cv::Rect(0, 0, rect_rough_roi.width, rect_rough_roi.height >> 1)); //Ang
//                     sub_frame[1] = cv::Mat(frame_rough_roi, cv::Rect(0, rect_rough_roi.height >> 1, rect_rough_roi.width, rect_rough_roi.height >> 1)); //Ang
//                     sub_mode=2; //Ang
// //                    cv::line(output_view,cv::Point(tl_pt.x,tl_pt.y+rect_rough_roi.height/2),cv::Point(tl_pt.x+rect_rough_roi.width,tl_pt.y+rect_rough_roi.height/2),cv::Scalar(255,0,0),2);
//                 }
//                 cv::Mat sub_sample[4]; //Ang
//                 sub_sample[0] = cv::Mat::zeros(sub_frame[0].size(), CV_8UC1); //Ang
//                 sub_sample[1] = cv::Mat::zeros(sub_frame[1].size(), CV_8UC1); //Ang
//                 std::vector<Points > selected_data(2); //Ang

//                 filterContourData(sub_frame,tl_pt, selected_data, sub_sample, sub_mode); //Ang
// //                float area_ratio = contour_area/(rect_rough_roi.width*rect_rough_roi.height);
// //                if(area_ratio > 0.7 && area_ratio < 0.8){
// //                    selected_data[0].insert(selected_data[0].end(),selected_data[1].begin(), selected_data[1].end());
// //                    selected_data.resize(1);
// //                }

//                 for(size_t j = 0; j < selected_data.size(); j++){ //Ang
//                     cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data[j], Gandamana::FIT_CIRCLE_MAX_STEPS, Gandamana::FIT_CIRCLE_EPS);//Ang
// //                    sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
// //                    int axis_dir = frame_rough_roi.cols > frame_rough_roi.rows?1:0;
// //                    cv::Vec2f interval_ctr = axis_dir?cv::Vec2f(tl_pt.x, br_pt.x):cv::Vec2f(tl_pt.y,br_pt.y);
// //                    std::cout << j << ". " << sub_circle_param << std::endl;
//                     if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost && //Ang
//                             sub_circle_param[2] > std::max(frame_rough_roi.cols,frame_rough_roi.rows) >> 2) //Ang
//                             //sub_circle_param[axis_dir] > interval_ctr[0] && sub_circle_param[axis_dir] < interval_ctr[1])
// //                            sub_circle_param[2] >= (float)std::max(sub_frame[0].cols,sub_frame[0].rows))
//                         //                           2*sub_circle_param[2] <= 1.333*(float)std::max(sub_frame[0].cols,sub_frame[0].rows))
//                         circle_param=sub_circle_param; //Ang
//                 }
//             }

//         //    std::cout << " Cost : " << circle_param[3] << std::endl;
//             float ball_percentage = (float)cv::countNonZero(frame_rough_roi)/(frame_rough_roi.cols*frame_rough_roi.rows); //Ang
//             // ROS_INFO("Ball percentage");
// #ifdef DEBUG
//            std::cout << "Ball Percentage : " << ball_percentage << std::endl;
// #endif
//                                                         // Maximum Circle Radius
//             // std::cout << circle_param << std::endl;
//            constexpr float MAX_BALL_RAD = static_cast<float>(Gandamana::FRAME_WIDTH >> 1); //Ang
//         //    std::cout << "MAX BALL RAD : " << MAX_BALL_RAD << std::endl;
//             if(circle_param[2] > .0f && circle_param[2] < MAX_BALL_RAD && ball_percentage > .35f){ //Ang
//                 count++; //Ang

//                 float circle_radius = circle_param[2]; //Ang
//                 int tl_roi_x = std::min(std::max(0,int(circle_param[0] - circle_radius)), output_view.cols-1); //Ang
//                 int tl_roi_y = std::min(std::max(0,int(circle_param[1] - circle_radius)), output_view.rows-1); //Ang
//                 int br_roi_x = std::min(std::max(0,int(circle_param[0] + circle_radius)), output_view.cols-1); //Ang
//                 int br_roi_y = std::min(std::max(0,int(circle_param[1] + circle_radius)), output_view.rows-1); //Ang

//                 cv::Rect roi_region(tl_roi_x,tl_roi_y, (br_roi_x - tl_roi_x), (br_roi_y - tl_roi_y)); //Ang
//                 ball_roi = roi_region; //Ang
//                 cv::Mat green_percentage(segmented_green, rect_rough_roi); //Ang
//                 float green_percent = (float)cv::countNonZero(green_percentage)/(rect_rough_roi.area()); //Ang
// #ifdef DEBUG
//                 // std::cout << "Green Percent : " << green_percent << std::endl;
// #endif
//                 if(green_percent > .001f){ // Minimum 0.1% Green //Ang
//                     if(circle_param[3] < best_candidate[3]){ //Ang
//                         best_candidate = circle_param; //Ang
//                         ROI = cv::Mat(ball_inside_field,rect_rough_roi);//Ang
//                     }
//                 }
//             }
//         }
//     }
//     static int next_idx=11; //Ang
//     ball_pos_.x = best_candidate[0]; //Ang
//     ball_pos_.y = best_candidate[1]; //Ang
//     ball_pos_.z = best_candidate[2]; //Ang
//     if(best_candidate[0] > 0){ //Ang
//     //    static int count_img = 0;

//         next_idx=11; //Ang
//         static int sample_count = 0; //Ang
//         regression_data_.emplace_back(cv::Point(best_candidate[0],best_candidate[1])); //Ang
//         sample_count++; //Ang
//         if(sample_count >= 10){ //Ang
//             sample_count=0; //Ang
//             est_trajectory_ = getBallPosPrediction(regression_data_); //Ang

//             regression_data_.clear(); //Ang
//         } //Ang

// //        std::stringstream file_name;
// //        file_name << "/media/koseng/New Volume/temp4/" << frame_id_ << "_" << count_img << ".jpg";
// //        cv::imwrite(file_name.str().c_str(),output_view);
// //        count_img++;
//     //    cv::imshow("ROI", ROI); //aman
//     //    cv::waitKey(0);
//     //    cv::destroyAllWindows();
//     }else if(next_idx < 16 && est_trajectory_.size() > 0){ //Ang
//         if(pred_status_){ //Ang
//             cv::Mat pred_pos = (next_idx*next_idx) * est_trajectory_[0] + //Ang
//                                 next_idx * est_trajectory_[1] + //Ang
//                                 est_trajectory_[2]; //Ang
//             ball_pos_.x = pred_pos.at<double>(0); //Ang
//             ball_pos_.y = pred_pos.at<double>(1);   //Ang         
//         }else{ //Ang
//             ball_pos_ = last_ball_pos_; //Ang
//         }
//         next_idx++; //Ang
//     }else{ //Ang
//         if(regression_data_.size() > 0)regression_data_.clear(); //Ang
//     }
//     last_ball_pos_ = ball_pos_; //Ang
//     // ROS_INFO("ball posisi: x=%f y=%f", ball_pos_.x, ball_pos_.y);
//     ball_pos_pub_.publish(ball_pos_); //Ang

    //For purpose GUI only -
//    cv::waitKey(1);
    // std::cout << "Best candidate : " << best_candidate[0] << ", " << best_candidate[1] << ", " << best_candidate[2] << std::endl;
    switch(frame_mode_){// 144 - 408 microseconds //Ang
//    case 0:cvtColor(in_img_,in_img_,CV_BGR2RGB);setOutputImage(in_img_);break;
    case 1:setOutputImage(in_hsv_);break; //Ang
    case 2:setOutputImage(thresh_image);break; //Ang
    case 3:{ //Ang
        cv::cvtColor(field_contour,field_contour,CV_GRAY2BGR); //Ang
        for(int i=0;i<line_tip_.tip1.size();i++){ //Ang
                cv::Point tip1(line_tip_.tip1[i].x, line_tip_.tip1[i].y); //Ang
                cv::Point tip2(line_tip_.tip2[i].x, line_tip_.tip2[i].y); //Ang
                cv::line(output_view,cv::Point(line_tip_.tip1[i].x, line_tip_.tip1[i].y),//Ang
                        cv::Point(line_tip_.tip2[i].x, line_tip_.tip2[i].y), cv::Scalar(255,0,255), 3); //Ang
                cv::circle(output_view, tip1, 7, cv::Scalar(100,50,100), CV_FILLED); //Ang
                cv::circle(output_view, tip2, 7, cv::Scalar(100,50,100), CV_FILLED); //Ang
        }
        line_tip_.tip1.clear(); //Ang
        line_tip_.tip2.clear(); //Ang
        cv::bitwise_and(output_view,field_contour,output_view); //Ang
        cvtColor(output_view,output_view,CV_BGR2RGB); //Ang
        setOutputImage(output_view);
        break; //Ang
        // cv::Mat ball_region(output_view_roi.size(),CV_8UC3,cv::Scalar(0,255,255));//Ang
        // cv::addWeighted(output_view_roi, .5, ball_region, .5, .0,output_view_roi); //Ang
        //cv::circle(output_view,cv::Point(best_candidate[0],best_candidate[1]),best_candidate[2],cv::Scalar(0,255,255),2); //Ang --ini fix in

        // for(int idx = 0; idx < 16 && est_trajectory_.size() > 0 && //Ang
        //     regression_data_.size() > 0; idx++){ //Ang
        //     cv::Mat pred_pos = (idx*idx) * est_trajectory_[0] + //Ang
        //                         idx * est_trajectory_[1] + //Ang
        //                         est_trajectory_[2]; //Ang
        //     cv::Point pred_pos_pt(pred_pos.at<double>(0), pred_pos.at<double>(1)); //Ang
        //     cv::circle(output_view, pred_pos_pt, //Ang
        //                4, cv::Scalar(idx<10?255:0,idx<10?255:0,idx<10?255:0), CV_FILLED); //Ang
        // }
        // for(int i=0;i<line_tip_.tip1.size();i++){ //Ang
        //     cv::Point tip1(line_tip_.tip1[i].x, line_tip_.tip1[i].y); //Ang
        //     cv::Point tip2(line_tip_.tip2[i].x, line_tip_.tip2[i].y); //Ang
        //     cv::line(output_view,cv::Point(line_tip_.tip1[i].x, line_tip_.tip1[i].y),//Ang
        //              cv::Point(line_tip_.tip2[i].x, line_tip_.tip2[i].y), cv::Scalar(255,0,255), 3); //Ang
        //     cv::circle(output_view, tip1, 7, cv::Scalar(100,50,100), CV_FILLED); //Ang
        //     cv::circle(output_view, tip2, 7, cv::Scalar(100,50,100), CV_FILLED); //Ang
        // }
    }
    default://cvtColor(in_img_,in_img_,CV_BGR2RGB); //Ang
        setOutputImage(in_img_);break; //Ang
    }
//    setOutputImage(in_img_);
    // =================================EXPERIMENTAL==================================

    // publishImage();//Ang
    // auto t2 = boost::chrono::high_resolution_clock::now();
    // auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();
    // std::cout << "Elapsed Time : " << elapsed_time << std::endl;
    // std::cout << it_sw_pub_ << "SW" << std::endl;
    // std::cout << it_inv_sg_pub_ << "SG" << std::endl;
    // std::cout << field_boundary_pub_ << "FB" << std::endl;

    //=================================================================================================
    //=================================================================================================
    //                               LOKALISASI 
    //=================================================================================================
    //=================================================================================================
    
//     vision_utils::LineTip tip_points;
//     field_boundary_ = field_prop.second;
//     constexpr float FRAME_AREA = FRAME_WIDTH*FRAME_HEIGHT;
//     bool blind = (1.0f - (float)cv::countNonZero(inv_segmented_green)/FRAME_AREA) < 0.01f;
//     // std::cout <<"Buta: " << blind << std::endl;
//     auto t1 = boost::chrono::high_resolution_clock::now();
//     lost_features_ = true;

//     if(tilt_servo_angle_ < tilt_limit_ && offset_head_ < 40.0f && fabs(roll_offset_) < 22.0f){ //robot condition for feature observation
//         Points target_points;
//         Vectors3 line_models;

// //         //LocalizationUtils::getInstance()->scanLinePoints(invert_green_, segmented_white_, field_boundary_, target_points);
//         LocalizationUtils::getInstance()->scanLinePoints(inv_segmented_green, segmented_white, field_boundary_, target_points);
//         arrangeTargetPoints(target_points);
//         getFeaturesModels(target_points,line_models,features_,tip_points);
//         cv::Mat check_tp = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC1);
//             for(size_t i = 0;i < target_points.size();i++){
//                 check_tp.at<uchar>(target_points[i].y, target_points[i].x) = 255;
//             }
//             // std::cout << "CHECKKKK TP " << target_points.size() << std::endl; //aman
//             // std::cout << "CHECKKKK TP " << target_points << std::endl; //aman

//         for(size_t i=0;i<tip_points.tip1.size();i++){
//             cv::line(check_tp, cv::Point(tip_points.tip1[i].x, tip_points.tip1[i].y),
//             cv::Point(tip_points.tip2[i].x, tip_points.tip2[i].y),
//             cv::Scalar(150),2);
//         }
//         line_tip_ = tip_points;
//         cv::imshow("CHECK_TOK",check_tp); //aman
//         // std::cout << "CEEEEEEEEEEEEEEEEEEKKKKKKK " << check_tp << std::endl; //aman lag tapi
//     }
//     update(); 
//     if(blind){
//         robot_state_.x = 999.0;
//         robot_state_.y = 999.0;
//     }

//     if(odometer_buffer_.size() > 0) {
//             odometer_buffer_.erase(odometer_buffer_.begin());
//     }

//     publishData();

//     auto t2 = boost::chrono::high_resolution_clock::now();
//     auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();

//     static auto sum_et = .0;

// // #ifdef DEBUG
// //     std::cout << "Elapsed Time : " << elapsed_time << " ms : SUM ET : " << sum_et << std::endl;
// // #endif

//     if(lost_features_)
//         sum_et += elapsed_time;

//     if(sum_et > 200.0 || (!lost_features_ && sum_et > .0)){//hold for 200 ms
//         features_present_ = 0;
//         features_.feature.clear();
    //     sum_et = .0;
    // }
    
    // actionForMonitor();
    
// #ifdef DEBUG
//     if(!debug_viz_.empty())
//         cv::imshow("DEBUG_VIZ", debug_viz_);
//     cv::waitKey(1);
// #endif
}

void BallDetector::integralImage(const cv::Mat &_input){
    cv::Mat ch1_ii = cv::Mat::zeros(Gandamana::FRAME_HEIGHT, Gandamana::FRAME_WIDTH, CV_32FC1);
    cv::Mat ch2_ii = ch1_ii.clone();
    cv::Mat ch3_ii = ch1_ii.clone();

    int bound_y{0}, bound_x{0};
    int prev_y{0}, prev_x{0};

    for(int i=0;i<_input.rows;i++){
        bound_y = (int)(i > 0);
        prev_y = bound_y ? i-1 : 0;
        for(int j=0;j<_input.cols;j++){
            bound_x = (int)(j > 0);
            prev_x = bound_x ? j-1 : 0;
            ch1_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[0] +
                                    ch1_ii.at<float>(i,prev_x) * bound_x +
                                    ch1_ii.at<float>(prev_y,j) * bound_y;
            ch2_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[1] +
                                    ch2_ii.at<float>(i,prev_x) * bound_x +
                                    ch2_ii.at<float>(prev_y,j) * bound_y;
            ch3_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[2] +
                                    ch3_ii.at<float>(i,prev_x) * bound_x +
                                    ch3_ii.at<float>(prev_y,j) * bound_y;
            // std::cout << j << " , " << i << " ; " << ch1_ii.at<float>(i,j) << " ; " << ch2_ii.at<float>(i,j) << " ; " << ch1_ii.at<float>(i,j) << std::endl;
        }
    }
}
//=================================================================================================
//                               IMPLEMENTASI PROSES LOKALISASI
//=================================================================================================
void BallDetector::sampleMotionModelOdometry(Particles &_particles_state,const geometry_msgs::Pose2D &_odometer_out) {
    bool idle = odometer_buffer_.size() == 0;
    
    float diff_x = .0f;
    float diff_y = .0f;
    
    if(!idle){
        diff_x = odometer_buffer_.front().x;
        diff_y = -odometer_buffer_.front().y;
    }

    float drot1 = atan2(diff_x, -diff_y);// + gy_heading_*Math::DEG2RAD;
    float dtrans = sqrt(diff_x*diff_x + diff_y*diff_y);
    float drot2 = _odometer_out.theta; //- drot1;

    //temporarily not yet use motion noise
//    float drot1_sqr = drot1*drot1;
//    float dtrans_sqr = dtrans*dtrans;
//    float drot2_sqr = drot2*drot2;

    double noise_std_dev = features_present_ > 0 ? 1.75 : .0;//std::min(3,features_present_);

    for(Particles::iterator it=_particles_state.begin();
        it!=_particles_state.end();it++){
//        if(it->z < 0)it->z = 360 + it->z;
        float drot1_hat = drot1 ;//- sampleNormal(params_.alpha1*drot1_sqr + params_.alpha2*dtrans_sqr);
        float dtrans_hat = dtrans ;//- sampleNormal(params_.alpha3*dtrans_sqr + params_.alpha4*drot1_sqr + params_.alpha4*drot2_sqr);
        float drot2_hat = drot2 ;//- sampleNormal(params_.alpha1*drot2_sqr + params_.alpha2*dtrans_sqr);
        drot1_hat *= Math::RAD2DEG;
        drot2_hat *= Math::RAD2DEG;

//        if(dtrans_hat > 0.0)std::cout << "ROT1_HAT : " << drot1_hat << " ; TRANS_HAT : " << dtrans_hat << " ; ROT2_HAT : " << drot2_hat << std::endl;
        float tetha = (it->z /*+ drot1_hat*/)*Math::DEG2RAD;
        //Posterior Pose
//        it->x = it->x + dtrans_hat*cos(tetha) + features_present_ * (idle ? genNormalDist(0,0.5) : 0);
//        it->y = it->y + dtrans_hat*sin(tetha) + features_present_ * (idle ? genNormalDist(0,0.5) : 0);
//        it->z = it->z + /*drot1_hat +*/ drot2_hat + features_present_ * (idle ? genNormalDist(0,1) : 0);
        it->x = it->x + dtrans_hat*cos(tetha) + genNormalDist(.0, noise_std_dev);
        it->y = it->y + dtrans_hat*sin(tetha) + genNormalDist(.0, noise_std_dev);
        it->z = it->z + /*drot1_hat +*/ drot2_hat + genNormalDist(.0, noise_std_dev);
        if(it->z < .0)it->z = 360.0 + it->z;
    }

    odometer_out_.theta = .0;
}

void BallDetector::measurementModel(Particles &_particles_state, vision_utils::Features _features_arg, float &_weight_avg){
    std::cout << "FEATURE SIZE ARG: " << _features_arg.feature.size() << std::endl;
    if(_features_arg.feature.size() == 0){
//        float uniform_weight = 1.0/params_.num_particles;
//        for(Particles::iterator it = _particles_state.begin();
//            it != _particles_state.end(); it++){
//            it->w = uniform_weight;
//        }
        // _weight_avg = uniform_weight;
        _weight_avg = last_weight_avg_;
        return;
    }

    vision_utils::Features _features = _features_arg;
    for(std::vector<vision_utils::Feature>::iterator it = _features.feature.begin();it != _features.feature.end(); it++){
        it->param1 *= .01f;
        it->param2 *= .01f;
        it->param3 *= .01f;
        it->param4 *= .01f;

//        std::cout << "Param 4 : " << it->param4 << std::endl;
        // std::cout << "ORIENTATION : "  << it->orientation  << std::endl;
        it->orientation *= Math::DEG2RAD;
    }

    float total_weight = .0f;
//     float minimum_weight = std::numeric_limits<float>::max();
//    float max_weight = std::numeric_limits<float>::min();
//    int num_features = _features.feature.size();
//    float max_weight = 0;
    cv::Vec3f top3_weight = {.0f, .0f, .0f};
    bool acquisition[3] = {false, false, false};
    //Range, Beam, Correspondence
    Vecs4 weight_param(_features.feature.size());
    for(Particles::iterator it = _particles_state.begin();
        it != _particles_state.end(); it++){
//        std::cout << it->z << std::endl;        

//        bool segline_used[11] = {false,false,false,false,false,
//                                 false,false,false,false,false,false};
        float pos_x = it->x * .01f;
        float pos_y = it->y * .01f;
        float tetha =  it->z * Math::DEG2RAD;
        float c_t = cos(tetha);
        float s_t = sin(tetha);
        for(std::pair<std::vector<vision_utils::Feature>::iterator,Vecs4::iterator > it_pair(_features.feature.begin(), weight_param.begin());
            it_pair.first != _features.feature.end();
            it_pair.first++, it_pair.second++){

            int feature_type = it_pair.first->feature_type;
            // std::cout << "Feature type: " << feature_type << std::endl;
            (*it_pair.second)[0] = FIELD_LENGTH;
            (*it_pair.second)[1] = Math::PI_TWO;
            (*it_pair.second)[2] = .0f;
            (*it_pair.second)[3] = -1.0f; // Unknown Feature

            if(feature_type < 4){//L, T, X, circle landmark                
                float optimal_diff = std::numeric_limits<float>::max();
                for(size_t j = 0; j< landmark_pos_[feature_type].size(); j++){
                    float delta_x = (landmark_pos_[feature_type][j].x - pos_x);
                    float delta_y = (landmark_pos_[feature_type][j].y - pos_y);
                    float feature_dist = sqrt(delta_x*delta_x + delta_y*delta_y);
                    float diff = std::fabs(feature_dist - it_pair.first->param4);
                    if(diff < optimal_diff){                        
                        (*it_pair.second)[0] = diff;
                        float beam_dev = atan2(delta_y,delta_x);
                        int map_tetha = tetha > Math::PI ? (tetha-Math::TWO_PI) : tetha;
                        beam_dev = beam_dev - map_tetha;
//                        if(beam_dev < 0)beam_dev = Math::TWO_PI + beam_dev;
//                        beam_dev = tetha-beam_dev;
//                        beam_dev = tetha-beam_dev;
//                        int dir=1;
//                        if(fabs(beam_dev) > 180)dir =- 1;
//                        beam_dev = std::min(beam_dev, 360 - beam_dev) * dir;
//                        (*it_pair.second)[1] = fabs((beam_dev < Math::PI ? beam_dev : Math::TWO_PI - beam_dev) - it_pair.first->orientation);
                        beam_dev = std::fabs(beam_dev - it_pair.first->orientation);
                        (*it_pair.second)[1] = beam_dev;
                        (*it_pair.second)[2] = .0f;
                        (*it_pair.second)[3] = -1.0f;
//                        std::cout << (*it_pair.second) << std::endl;
                        optimal_diff = diff;
                    }

                }
            }else{
                float optimal_diff = std::numeric_limits<float>::max();

                cv::Point2f a(pos_x + it_pair.first->param2*c_t - it_pair.first->param1*s_t,
                            pos_y + it_pair.first->param2*s_t + it_pair.first->param1*c_t);
                cv::Point2f b(pos_x + it_pair.first->param4*c_t - it_pair.first->param3*s_t,
                            pos_y + it_pair.first->param4*s_t + it_pair.first->param3*c_t);
//                float segline_len = sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
                float orientation2V = std::min(std::fabs(Math::PI_TWO - tetha),
                                               std::fabs(Math::THREE_PI_TWO - tetha));
                float orientation2H = std::min(std::fabs(tetha),
                                      std::min(std::fabs(Math::TWO_PI - tetha),
                                               std::fabs(Math::PI - tetha)));

                float diffV = std::fabs(orientation2V - it_pair.first->orientation);
                float diffH = std::fabs(orientation2H - it_pair.first->orientation);

                for(size_t j = 0;j < line_segment_pos_.size(); j++){
//                    if(segline_used[j])continue;
                    if(j < 5){//Vertical Index
//                        if(a.y > line_segment_pos_[j][1] && b.y < line_segment_pos_[j][3]){
//                        float refline_len = (line_segment_pos_[j][3] - line_segment_pos_[j][1]);
//                        if(segline_len <= refline_len &&
//                            (std::min(a.y,b.y) - line_segment_pos_[j][1]) > -MAX_DIFF_OFFSET && // maximum 25 cm offset
//                            (std::max(a.y,b.y) - line_segment_pos_[j][3]) < MAX_DIFF_OFFSET){
                        float diff_offset1 = std::max(.0f, line_segment_pos_[j][1] - std::min(a.y,b.y));
                        float diff_offset2 = std::max(.0f, std::max(a.y,b.y) - line_segment_pos_[j][3]);
//                        float diff = std::fabs(a.x - line_segment_pos_[j][2]) + std::fabs(b.x - line_segment_pos_[j][0]);
                        float diff = std::fabs((std::max(a.x,b.x) - line_segment_pos_[j][2]) + (std::min(a.x,b.x) - line_segment_pos_[j][0]));
//                        float diff = std::fabs(std::max(a.x,b.x) - line_segment_pos_[j][2]) - std::fabs(std::min(a.x,b.x) - line_segment_pos_[j][0]);
                        float temp_diff = diff + diff_offset1 + diff_offset2 + diffV;
                        if(temp_diff < optimal_diff){
                            optimal_diff = temp_diff;
                            (*it_pair.second)[0] = diff + diff_offset1 + diff_offset2;
//                            float beam_dev = std::fabs(orientation_p2ref - it_pair.first->orientation);
                            (*it_pair.second)[1] = diffV;//fabs(beam_dev < 90.0 ? beam_dev : 180-beam_dev);
                            (*it_pair.second)[2] = diff_offset1 + diff_offset2;
                            (*it_pair.second)[3] = j;
//                            }
                        }
                    }else{//Horizontal Remains
//                        if(a.x > line_segment_pos_[j][0] && b.x < line_segment_pos_[j][2]){
//                        float refline_len = (line_segment_pos_[j][2] - line_segment_pos_[j][0]);
//                        if(segline_len <= refline_len ){
                        float diff_offset1 = std::max(.0f, line_segment_pos_[j][0] - std::min(a.x,b.x));
                        float diff_offset2 = std::max(.0f, std::max(a.x,b.x) - line_segment_pos_[j][2]);
//                        float diff = std::fabs(a.y - line_segment_pos_[j][3]) + std::fabs(b.y - line_segment_pos_[j][1]);
                        float diff = std::fabs((std::max(a.y,b.y) - line_segment_pos_[j][3]) + (std::min(a.y,b.y) - line_segment_pos_[j][1]));
//                        float diff = std::fabs(std::max(a.y,b.y) - line_segment_pos_[j][3]) - std::fabs(std::min(a.y,b.y) - line_segment_pos_[j][1]);
                        float temp_diff = diff + diff_offset1 + diff_offset2 + diffH;
                        if(temp_diff < optimal_diff){
                            optimal_diff = temp_diff;
                            (*it_pair.second)[0] = diff + diff_offset1 + diff_offset2;
//                            float beam_dev = std::fabs(orientation - it_pair.first->orientation);
                            (*it_pair.second)[1] = diffH;//fabs(beam_dev < 180.0 ? 90 - beam_dev : 270 - beam_dev);
                            (*it_pair.second)[2] = diff_offset1 + diff_offset2;
                            (*it_pair.second)[3] = j;
                        }
//                        }
                    }
                }
//                if((*it_pair.second)[2]>0)segline_used[(int)(*it_pair.second)[2]] = true;
            }

        }

        int features_used = 0;

        for(Vecs4::const_iterator it2 = weight_param.begin();
            it2 != weight_param.end(); it2++){
//            std::cout << (*it2)[0] << " ; " << (*it2)[1] << " ; " << (*it2)[3] << std::endl;
            float weight = (pdfRange((*it2)[0])*pdfBeam((*it2)[1]));//*probDensityFunc((*it2)[2],1.0));

            if(weight > top3_weight[0] && !acquisition[0]){
                acquisition[0] = (*it2)[3] == -1.0f;
                top3_weight[1] = top3_weight[0];
                top3_weight[0] = weight;
                ++features_used;
            }else if(weight > top3_weight[1] && !acquisition[1]){
                acquisition[1] = (*it2)[3] == -1.0f;
                top3_weight[2] = top3_weight[1];
                top3_weight[1] = weight;
                ++features_used;
            }else if(weight > top3_weight[2] && !acquisition[2]){
                acquisition[2] = (*it2)[3] == -1.0f;
                top3_weight[2] = weight;
                ++features_used;
            }

//            updated_weight += weight;
//            it->w = updated_weight;
        }
        features_used = std::max(1, std::min(features_used, 3));

        float gy_dev = fabs(tetha - gy_heading_ * Math::DEG2RAD);
        gy_dev = gy_dev > Math::PI ? Math::TWO_PI - gy_dev : gy_dev;
        float updated_weight = (top3_weight[0]*(int)(features_used > 0) +
                          top3_weight[1]*(int)(features_used > 1) +
                          top3_weight[2]*(int)(features_used > 2))/features_used;
//        std::cout << "VIS WEIGHT : " << updated_weight << std::endl;
        // float gy_weight = probDensityFunc(gy_dev,params_.gy_var);
        float gy_weight = expWeight(gy_dev, params_.gy_var);
//        std::cout << "GY Weight : " << gy_weight << std::endl;
//         updated_weight *= gy_weight;

//         it->w = updated_weight;

// //        if(updated_weight > max_weight)max_weight = updated_weight;

//         total_weight += updated_weight;

//         if(updated_weight < minimum_weight)
//             minimum_weight = updated_weight;

//        if(updated_weight > max_weight){
//            max_weight = updated_weight;
//        }

        top3_weight[0] = top3_weight[1] = top3_weight[2] = .0f;
        acquisition[0] = acquisition[1] = acquisition[2] = false;
    }
#ifdef DEBUG
//    std::cout << "MAX Weight : " << max_weight << std::endl;
//    std::cout << "Minimum Weight : " << minimum_weight << std::endl;
#endif

    float weight_avg = .0f;
//    if(max_weight == 0.0)max_weight = 1.0;
    for(Particles::iterator it = _particles_state.begin();
        it != _particles_state.end(); it++){
//        weight_avg += it->w/max_weight;
        weight_avg += it->w;
        it->w = it->w / total_weight;        
    }    
//    if(weight_avg==0)weight_avg=last_weight_avg_;
//    _weight_avg = weight_avg == 0.0 ? last_weight_avg_ : weight_avg/params_.num_particles;

    _weight_avg = weight_avg/params_.num_particles;
}

inline void BallDetector::arrangeTargetPoints(Points &_target_points){
    for(size_t i = 0; i < _target_points.size(); i++){
        for(size_t j = i+1; j < _target_points.size(); j++){
            if(_target_points[i].x > _target_points[j].x){
                _target_points[i] = _target_points[i] + _target_points[j];
                _target_points[j] = _target_points[i] - _target_points[j];
                _target_points[i] = _target_points[i] - _target_points[j];
            }else if(_target_points[i].x == _target_points[j].x){
                if(_target_points[i].y > _target_points[j].y){
                    _target_points[i] = _target_points[i] + _target_points[j];
                    _target_points[j] = _target_points[i] - _target_points[j];
                    _target_points[i] = _target_points[i] - _target_points[j];
                }
            }
        }
    }
}

void BallDetector::getFeaturesModels(Points &_target_points,
                                 Vectors3 &_line_models,
                                 vision_utils::Features &_features_arg){
    vision_utils::Features _features;
    cv::Mat points_map = cv::Mat::zeros(POINTS_MAP_H, POINTS_MAP_W, CV_8UC1);
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;
    cv::Point remap_origin(remap_x, remap_y);
    cv::Vec4f best_circle(-1.0f, -1.0f, std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    constexpr float CENTER_CIRCLE_RADIUS = (CENTER_CIRCLE_DIAMETER >> 1);

    for(size_t i = 0; i < _target_points.size(); i++){
        points_map.at<uchar>(remap_origin.y + _target_points[i].y, remap_origin.x + _target_points[i].x) = 255;
    }
}

void BallDetector::getFeaturesModels(Points &_target_points,
                                 Vectors3 &_line_models,
                                 vision_utils::Features &_features, vision_utils::LineTip &_line_tip){

    cv::Mat points_map = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC1);
    cv::Mat visualize = cv::Mat::zeros(FRAME_HEIGHT,FRAME_WIDTH, CV_8UC3);
    cv::Point remap_origin(FRAME_WIDTH >> 1,0);

    std::vector<Points > segline_inliers;

    Points target_pt;
    Vectors3 original_line_models;
   std::cout << "=============" << std::endl;
    for(int i = MAX_LINE_MODEL; i--;){
        Points inliers;
        geometry_msgs::Vector3 model;
      std::cout << "Size target point: " << _target_points.size() << std::endl;
        //Unsufficent data
        if(_target_points.size() < MIN_LINE_INLIERS)
            break;

        //Get line models & Update target points
        LocalizationUtils::getInstance()->RANSAC(_target_points, model, Gandamana::RANSAC_NUM_SAMPLES, Gandamana::RANSAC_MAX_ITER, inlier_error, MIN_LINE_INLIERS, inliers);
    //    std::cout << model.x << " ; " << model.y << std::endl;
        //Unable to get line model
        if(model.x == 0 || model.y == 0)
            continue;
        
        original_line_models.push_back(model);
        cv::Point tip1(inliers.front().x,model.x + inliers.front().x*model.y);
        cv::Point tip2(inliers.back().x,model.x + inliers.back().x*model.y);
        target_pt.push_back(tip1);
        target_pt.push_back(tip2);
        
        segline_inliers.push_back(inliers);

        geometry_msgs::Point tip_1,tip_2;
        tip_1.x = tip1.x;
        tip_1.y = tip1.y;
        tip_2.x = tip2.x;
        tip_2.y = tip2.y;
        _line_tip.tip1.push_back(tip_1);
        _line_tip.tip2.push_back(tip_2);

        cv::line(points_map,tip1,tip2,cv::Scalar(100),2);

    }

    Points projected_points = pointsProjection(target_pt);
    std::cout << "projected_points size: " << projected_points.size() << std::endl; //aman -hkm
    target_pt.clear();
    for(size_t i=0;i<projected_points.size()/2;i++){
        int idx1 = 2*i;
        int idx2 = 2*i+1;        

        vision_utils::Feature feature_data;
        // std::cout << "feature_data " << feature_data << std::endl; //masih nol -hkm
        feature_data.param1 = projected_points[idx1].x;
        feature_data.param2 = projected_points[idx1].y;
        feature_data.param3 = projected_points[idx2].x;
        feature_data.param4 = projected_points[idx2].y;
        float orientation = atan2(feature_data.param4-feature_data.param2, feature_data.param3-feature_data.param1)*Math::RAD2DEG;
        feature_data.orientation = orientation < 0 ? 180+orientation:orientation;
        feature_data.feature_type = 4;
        _features.feature.push_back(feature_data);
        
        geometry_msgs::Vector3 line_model;
        line_model.y = (projected_points[idx2].y - projected_points[idx1].y)/(projected_points[idx2].x - projected_points[idx1].x + 1e-6);
        line_model.x = projected_points[idx1].y - line_model.y*projected_points[idx1].x;
        _line_models.push_back(line_model);  
        

        cv::line(visualize,cv::Point(remap_origin.x + feature_data.param1, remap_origin.y + feature_data.param2),
                cv::Point(remap_origin.x + feature_data.param3, remap_origin.y + feature_data.param4),
                cv::Scalar(255,0,255),2);
    }
    
    //merge smiliar line
//     std::cout << "==================" << std::endl;
    for(size_t i=0;i<_line_models.size();i++){
        for(size_t j=i+1;j<_line_models.size();j++){
            float grad_ratio = fabs(_line_models[i].y) > fabs(_line_models[j].y) ?
                        fabs(_line_models[j].y/_line_models[i].y) : fabs(_line_models[i].y/_line_models[j].y);
//             std::cout << "Grad Ratio : " << grad_ratio << std::endl;
             bool dominant = segline_inliers[i].size() >= segline_inliers[j].size();
             int idx1 = dominant ? i : j;
             int idx2 = dominant ? j : i;
             float tip1_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param1 - _features.feature[idx2].param2 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
             float tip2_dist = fabs(_line_models[idx1].y*_features.feature[idx2].param3 - _features.feature[idx2].param4 + _line_models[idx1].x)/sqrt(_line_models[idx1].y*_line_models[idx1].y + 1);
//             std::cout << "Bias Diff : " << tip1_dist + tip2_dist << std::endl;
            if(grad_ratio > 0.35 &&
                    (tip1_dist + tip2_dist) < 50){
                // _line_models[i].y = (_line_models[i].y + _line_models[j].y)/2;
                _line_models[i] = dominant ? _line_models[i] : _line_models[j];
                _features.feature[i].param1 = std::min(_features.feature[i].param1,_features.feature[j].param1);
                _features.feature[i].param2 = _line_models[i].x + _line_models[i].y*_features.feature[i].param1;
                _features.feature[i].param3 = std::max(_features.feature[i].param3,_features.feature[j].param3);
                _features.feature[i].param4 = _line_models[i].x + _line_models[i].y*_features.feature[i].param3;                                

                float orientation = atan2(_features.feature[i].param4-_features.feature[i].param2,
                                          _features.feature[i].param3-_features.feature[i].param1)*Math::RAD2DEG;
                _features.feature[i].orientation = orientation < 0 ? 180+orientation:orientation;
                _features.feature.erase(_features.feature.begin() + j);

                _line_models.erase(_line_models.begin() + j);
                
                original_line_models[i] = dominant ? original_line_models[i] : original_line_models[j];
                _line_tip.tip1[i].x = std::min(_line_tip.tip1[i].x,_line_tip.tip1[j].x);
                _line_tip.tip1[i].y = original_line_models[i].x + original_line_models[i].y * _line_tip.tip1[i].x;
                _line_tip.tip2[i].x = std::max(_line_tip.tip2[i].x,_line_tip.tip2[j].x);
                _line_tip.tip2[i].y = original_line_models[i].x + original_line_models[i].y * _line_tip.tip2[i].x;

                _line_tip.tip1.erase(_line_tip.tip1.begin() + j);
                _line_tip.tip2.erase(_line_tip.tip2.begin() + j);

                original_line_models.erase(original_line_models.begin() + j);

                segline_inliers[i].insert(segline_inliers[i].end(),segline_inliers[j].begin(),segline_inliers[j].end());
                segline_inliers.erase(segline_inliers.begin() + j);

                j--;
            }
        }
    }
    
    for(int i=0;i<(int)_line_models.size();i++){
        int diff_x = _features.feature[i].param1 - _features.feature[i].param3;
        int diff_y = _features.feature[i].param2 - _features.feature[i].param4;
        if(sqrt(diff_x*diff_x + diff_y*diff_y) < MIN_LINE_LENGTH){
            _features.feature.erase(_features.feature.begin() + i);
            _line_models.erase(_line_models.begin() + i);
            segline_inliers.erase(segline_inliers.begin() + i);
            i--;
            continue;
        }
    }
    
    features_present_ = _features.feature.size();
    std::cout << "feature present: " << features_present_ << std::endl;
    // cv::imshow("VIZ",visualize);
    // cv::imshow("PM",points_map);

}

Points BallDetector::pointsProjection(const Points &_points, bool ball){
    Points projected_point;
    // std::cout << "projected_point " << projected_point << std::endl; // kosong -hkm
    cv::Mat points_map = cv::Mat::zeros(POINTS_MAP_H, POINTS_MAP_W, CV_8UC1);
#ifdef DEBUG
    cv::Mat target_points_viz = cv::Mat::zeros(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    cv::Mat unrotated_pm = points_map.clone();
    cv::Mat compensated = target_points_viz.clone();
    cv::Mat distorted_tp = target_points_viz.clone();
#endif
    constexpr int remap_x = POINTS_MAP_W >> 1;
    constexpr int remap_y = POINTS_MAP_H >> 2;

    constexpr float ctr_frame_x = FRAME_WIDTH >> 1;
    constexpr float ctr_frame_y = FRAME_HEIGHT >> 1;
    cv::Point2f center_frame(ctr_frame_x, ctr_frame_y);
    float K1 = camera_info_.D[0];
    float K2 = camera_info_.D[1];
    float K3 = camera_info_.D[4];
    float P1 = camera_info_.D[2];
    float P2 = camera_info_.D[3];
    float cx = camera_info_.K[2];
    float cy = camera_info_.K[5];
    float fx = camera_info_.K[0];
    float fy = camera_info_.K[4];
    float ctr_x = (center_frame.x - cx)/fx;
    float ctr_y = (center_frame.y - cy)/fy;

    // std::cout << "Kamera K1: " << K1 << std::endl;

//    const float PI_2 = CV_PI/2;
    forwardKinematics();
    float c_roll_comp = cos(CAMERA_ORIENTATION.coeff(0));
    float s_roll_comp = sin(CAMERA_ORIENTATION.coeff(0));

    float c_pan_servo = cos(pan_servo_angle_ + pan_rot_comp_);//cos(CAMERA_ORIENTATION.coeff(2));
    float s_pan_servo = sin(pan_servo_angle_ + pan_rot_comp_);//sin(CAMERA_ORIENTATION.coeff(2));

    float shift_pixel = (shift_const_ + ctr_frame_y)*(1 - c_roll_comp);
    for(Points::const_iterator it=_points.begin(); it != _points.end(); it++){        

        float xn = (it->x - cx)/fx;
        float yn = (it->y - cy)/fy;
        float diff_x = xn - ctr_x;
        float diff_y = yn - ctr_y;
        float rd_2 = diff_x*diff_x + diff_y*diff_y;
        float rd_4 = rd_2*rd_2;
        float rd_6 = rd_4*rd_2;
        float radial_distort = (1.0f + K1*rd_2 + K2*rd_4 + K3*rd_6);

        float undistort_x = xn*(radial_distort) +
                2.0f*P1*xn*yn + P2*(rd_2 + 2.0f*xn*xn);

        float undistort_y = yn*(radial_distort) +
                P1*(rd_2 + 2.0f*yn*yn) + 2.0f*P2*xn*yn;

        undistort_x = fx*undistort_x + cx;
        undistort_y = fy*undistort_y + cy;

        float trans_x = undistort_x - center_frame.x;
        float trans_y = undistort_y - center_frame.y;

    //   float trans_x = it->x - center_frame.x;
    //   float trans_y = it->y - center_frame.y;

        //projection start here

//        float compensated_x = center_frame.x + trans_x*cos(roll_compensation_) + trans_y*sin(roll_compensation_);
//        float compensated_y = center_frame.y - trans_x*sin(roll_compensation_) + trans_y*cos(roll_compensation_);

//        float roll_comp = (1 - fabs(pan_servo_angle_)/PI_2)*roll_compensation_ + (pan_servo_angle_/PI_2) * offset_head_;
//        float roll_comp = cos(pan_servo_angle_)*roll_compensation_ + sin(pan_servo_angle_)*offset_head_;

        float compensated_x = center_frame.x + trans_x*c_roll_comp + trans_y*s_roll_comp + shift_pixel;
        float compensated_y = center_frame.y - trans_x*s_roll_comp + trans_y*c_roll_comp + shift_pixel;

//        float offset_pan = panAngleDeviation(undistort_x);
//        float offset_tilt = tiltAngleDeviation(undistort_y);

        float offset_pan = panAngleDeviation(compensated_x);
        float offset_tilt = tiltAngleDeviation(compensated_y);

//        float distance_y = verticalDistance(offset_tilt);
        float distance_y = verticalDistance(offset_tilt);
        float distance_x = horizontalDistance(distance_y, offset_pan);
        //experimen
        float area = distance_x*distance_x + distance_y*distance_y;
        // std::cout << "OP : " << offset_pan << std::endl;
        // std::cout << " X : " << distance_x << " ; Y : " << distance_y << std::endl;
        std::cout << "Area Feature: " << area << std::endl;
        ball = true;
        if(sqrt(distance_x*distance_x + distance_y*distance_y) < 55.0 && !ball)//ignore feature in less than 55 cm
            continue;

        float rotated_x = distance_x*c_pan_servo + distance_y*s_pan_servo;
        float rotated_y = -distance_x*s_pan_servo + distance_y*c_pan_servo;

        // Robot local coordinate in real world

        cv::Point local_coord(rotated_x, rotated_y);

        int mapped_x = remap_x + local_coord.x;
        int mapped_y = remap_y + local_coord.y;
// #ifdef DEBUG
//         target_points_viz.at<uchar>(it->y,it->x) = 255;
//         if(compensated_x > 0 && compensated_x < target_points_viz.cols &&
//            compensated_y > 0 && compensated_y < target_points_viz.rows &&
//            debug_viz_mode == 4)
//             compensated.at<uchar>(compensated_y,compensated_x) = 255;
// #endif
        // std::cout << "Masuk MIN" << std::endl;
        //debug 
        // std::cout << "Nilai dibawah harus lebih 0" <<std::endl;
        // std::cout << "mapped_x: " << mapped_x<<std::endl;
        // std::cout << "mapped_y: " << mapped_y<<std::endl;
        // std::cout << "points_map: " << points_map.at<uchar>(mapped_y, mapped_x)<<std::endl;
        if(mapped_x < 0 || mapped_x >= POINTS_MAP_W || mapped_y < 0 || mapped_y >= POINTS_MAP_H || points_map.at<uchar>(mapped_y, mapped_x) > 0){
            continue;
        }

// #ifdef DEBUG
//         int map_unrotate_x = remap_x + distance_x;
//         int map_unrotate_y = remap_y + distance_y;
//         if(map_unrotate_x > 0 && map_unrotate_x < POINTS_MAP_W &&
//            map_unrotate_y > 0 && map_unrotate_y < POINTS_MAP_H &&
//            debug_viz_mode == 3)
//             unrotated_pm.at<uchar>(map_unrotate_y, map_unrotate_x) = 255;
            
//         if(undistort_x > 0 && undistort_x < target_points_viz.cols &&
//            undistort_y > 0 && undistort_y < target_points_viz.rows &&
//            debug_viz_mode == 5)
//             distorted_tp.at<uchar>(undistort_y,undistort_x) = 255;
// #endif
        std::cout << "Local cord: " <<local_coord.x <<std::endl;
        projected_point.push_back(local_coord);
        points_map.at<uchar>(mapped_y, mapped_x) = 255;
        
    //    cv::circle(draw,cv::Point(,),1,cv::Scalar(255),CV_FILLED);
    //    cv::imshow("draw: ", draw);
    }
// #ifdef DEBUG
//     if(!ball){
//         switch(debug_viz_mode){
//             case 0:debug_viz_ = target_points_viz.clone();break;
//             case 1:debug_viz_ = points_map.clone();break;
//             case 3:debug_viz_ = unrotated_pm.clone();break;
//             case 4:debug_viz_ = compensated.clone();break;
//             case 5:debug_viz_ = distorted_tp.clone();break;
//             default:break;
//         }
//     }
// #endif
    // cv::imshow("Projected Points",points_map);
    // cv::imshow("Target Points",target_points_viz);


    return projected_point;
}
inline float BallDetector::panAngleDeviation(float _pixel_x_pos){
    return atan((2.0f * _pixel_x_pos/FRAME_WIDTH - 1.0f) * TAN_HFOV_PER2);
}

inline float BallDetector::tiltAngleDeviation(float _pixel_y_pos){
    return atan((2.0f * _pixel_y_pos/FRAME_HEIGHT - 1.0f) * TAN_VFOV_PER2);
}

inline float BallDetector::verticalDistance(float _tilt_dev){
    float total_tilt = Math::PI_TWO - (CAMERA_ORIENTATION.coeff(1) + _tilt_dev);
    return (robot_height_ + CAMERA_DIRECTION.coeff(2) + z_offset_) * tan(total_tilt);
}

inline float BallDetector::horizontalDistance(float _distance_y, float _offset_pan){
    return _distance_y * tan(_offset_pan);
}


void BallDetector::publishData(){
    vision_utils::Particles particles_msg;
    geometry_msgs::PoseStamped robot_state_msg;
    robot_state_msg.pose.position.x = robot_state_.x ;
    robot_state_msg.pose.position.y = robot_state_.y ;
    robot_state_msg.pose.orientation.z = robot_state_.theta;
    robot_state_msg.header.stamp = this->stamp_;
    robot_state_msg.header.seq++;

    // std::cout << "POS X = " << robot_state_msg.pose.position.x << std::endl;//stuck 35
    // std::cout << "POS Y = " << robot_state_msg.pose.position.y << std::endl;//stuck 35
    // std::cout << "POS X 2 = " << robot_state_.x  << std::endl;//stuck 35
    // std::cout << "POS Y 2 = " << robot_state_.y  << std::endl; //stuck 35
    // std::cout << "POS Z = " << robot_state_.theta << std::endl; // stuck 0
    // std::cout << "HEADER = " << robot_state_msg.header.stamp << std::endl; // aman

    particles_msg.particle = particles_state_;
    particles_msg.header.stamp = this->stamp_;
    particles_msg.header.seq++;
    // std::cout << "Particles msg = " << particles_msg.particle.size() << std::endl; //gada
    // std::cout << "Header = " << particles_msg.header.stamp << std::endl; //ada
    // std::cout << "Header seq++ = " << particles_msg.header.seq++ << std::endl; //aman

    features_.header.stamp = this->stamp_;
    features_.header.seq++;

    geometry_msgs::PointStamped proj_ball_stamped_msg;
    proj_ball_stamped_msg.header.stamp = this->stamp_;
    proj_ball_stamped_msg.header.seq++;
    geometry_msgs::Point ball_pos_msg;
    ball_pos_msg.z = -1.0;
    proj_ball_stamped_msg.point = ball_pos_msg;

    if(ball_pos_.x != -1.0 &&
        ball_pos_.y != -1.0){
        Points ball_pos;
        ball_pos.emplace_back(cv::Point(ball_pos_.x,ball_pos_.y));
        ball_pos = pointsProjection(ball_pos, true);

        if(ball_pos.size() > 0){

            float c_t = cos(robot_state_.theta * Math::DEG2RAD);
            float s_t = sin(robot_state_.theta * Math::DEG2RAD);
            float shoot_dir = calcShootDir(cv::Point2f(robot_state_.x + ball_pos.front().y*c_t - ball_pos.front().x*s_t,
                                     robot_state_.y + ball_pos.front().y*s_t + ball_pos.front().x*c_t));

            ball_pos_msg.x = ball_pos.front().y;
            ball_pos_msg.y = -ball_pos.front().x;
            ball_pos_msg.z = shoot_dir;
            projected_ball_pub_.publish(ball_pos_msg);

            proj_ball_stamped_msg.point.x = ball_pos.front().x;
            proj_ball_stamped_msg.point.y = ball_pos.front().y;
            proj_ball_stamped_msg.point.z = shoot_dir;
        }
    }

    robot_state_pub_.publish(robot_state_msg);
    particles_state_pub_.publish(particles_msg);
    features_pub_.publish(features_);
    projected_ball_stamped_pub_.publish(proj_ball_stamped_msg);
}

float BallDetector::calcShootDir(const cv::Point2f &_ball_pos){//global ball pos
    // cv::Point2f goal_post1((attack_dir_?landmark_pos_[1][0]:landmark_pos_[1][4]) * 100.0f);
    // cv::Point2f goal_post2((attack_dir_?landmark_pos_[1][1]:landmark_pos_[1][5]) * 100.0f);
    constexpr float goal_postx = FIELD_LENGTH + BORDER_STRIP_WIDTH;
    constexpr float goal_posty1 = BORDER_STRIP_WIDTH + (FIELD_WIDTH - GOAL_WIDTH) * .5f;
    constexpr float goal_posty2 = BORDER_STRIP_WIDTH + (FIELD_WIDTH + GOAL_WIDTH) * .5f;
    cv::Point2f goal_post1(goal_postx, goal_posty1);
    cv::Point2f goal_post2(goal_postx, goal_posty2);
    constexpr float END_OF_XMONITOR = (float)(FIELD_LENGTH + (BORDER_STRIP_WIDTH << 1));
    constexpr float center_goal_y = (goal_posty1 + goal_posty2) * .5f;
    cv::Point2f center_goal(END_OF_XMONITOR, center_goal_y);

    constexpr float GK_OCCUPANCY = 80.0f;
    constexpr float HALF_GK_OCCUPANCY = GK_OCCUPANCY * .5f;
    cv::Point2f zero_dir_area1_tl(goal_post1.x - PENALTY_MARK_DISTANCE, goal_post1.y);
    cv::Point2f zero_dir_area1_br(goal_post1.x, center_goal.y - HALF_GK_OCCUPANCY);

    cv::Point2f zero_dir_area2_tl(goal_post2.x - PENALTY_MARK_DISTANCE, center_goal.y + HALF_GK_OCCUPANCY);
    cv::Point2f zero_dir_area2_br(goal_post2.x, goal_post2.y);

//    std::cout << zero_dir_area1_tl << " ; " << zero_dir_area1_br << std::endl;
//    std::cout << zero_dir_area2_tl << " ; " << zero_dir_area2_br << std::endl;

    if((_ball_pos.x > zero_dir_area1_tl.x && _ball_pos.x < zero_dir_area1_br.x &&
        _ball_pos.y > zero_dir_area1_tl.y && _ball_pos.y < zero_dir_area1_br.y) ||
            (_ball_pos.x > zero_dir_area2_tl.x && _ball_pos.x < zero_dir_area2_br.x &&
             _ball_pos.y > zero_dir_area2_tl.y && _ball_pos.y < zero_dir_area2_br.y) ||
            resetting_particle_){
        return 360.0f;
    }

    cv::Point2f gk_avoidance_area_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y - HALF_GK_OCCUPANCY);
    cv::Point2f gk_avoidance_area_br(center_goal.x - BORDER_STRIP_WIDTH, center_goal.y + HALF_GK_OCCUPANCY);

//    std::cout << gk_avoidance_area_tl << " ; " << gk_avoidance_area_br << std::endl;

    if((_ball_pos.x >= gk_avoidance_area_tl.x && _ball_pos.x <= gk_avoidance_area_br.x &&
        _ball_pos.y >= gk_avoidance_area_tl.y && _ball_pos.y <= gk_avoidance_area_br.y)){

        float robot_theta = robot_state_.theta;

        if(robot_theta < .0f)robot_theta = 360.0f + robot_theta;
        bool target_cond = robot_theta > 180.0 && robot_theta < 360.0;
        constexpr float HALF_GOAL_WIDTH = GOAL_WIDTH * .5f;
        cv::Point2f target_goal(center_goal.x - BORDER_STRIP_WIDTH, center_goal.y + (target_cond ? -HALF_GOAL_WIDTH : HALF_GOAL_WIDTH));

        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
        if(target_dir < .0f)target_dir = 360.0f + target_dir;
        return target_dir;
    }

    //Shooting Dir First Style
//    cv::Point2f avoidance_gk_area1_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y - (GK_OCCUPANCY * 0.5f));
//    cv::Point2f avoidance_gk_area1_br(center_goal.x, center_goal.y);

//    cv::Point2f avoidance_gk_area2_tl(center_goal.x - PENALTY_MARK_DISTANCE - BORDER_STRIP_WIDTH, center_goal.y);
//    cv::Point2f avoidance_gk_area2_br(center_goal.x, center_goal.y + (GK_OCCUPANCY * 0.5f));

//    if((_ball_pos.x >= avoidance_gk_area1_tl.x && _ball_pos.x <= avoidance_gk_area1_br.x &&
//        _ball_pos.y >= avoidance_gk_area1_tl.y && _ball_pos.y <= avoidance_gk_area1_br.y)){
//        cv::Point2f target_goal(center_goal.x, center_goal.y + GOAL_WIDTH * 0.5f);
//        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
//        if(target_dir < .0f)target_dir = 360.0f + target_dir;
//        return target_dir;
//    }

//    if((_ball_pos.x > avoidance_gk_area2_tl.x && _ball_pos.x < avoidance_gk_area2_br.x &&
//        _ball_pos.y > avoidance_gk_area2_tl.y && _ball_pos.y < avoidance_gk_area2_br.y)){
//        cv::Point2f target_goal(center_goal.x, center_goal.y - GOAL_WIDTH * 0.5f);
//        float target_dir = atan2(target_goal.y - _ball_pos.y, target_goal.x - _ball_pos.x) * Math::RAD2DEG;
//        if(target_dir < .0f)target_dir = 360.0f + target_dir;
//        return target_dir;
//    }

//==============
    float goal_width = fabs(goal_post1.y - goal_post2.y);
    float diff_x1 = _ball_pos.x - goal_post1.x;
    float diff_y1 = _ball_pos.y - goal_post1.y;
    float diff_x2 = _ball_pos.x - goal_post2.x;
    float diff_y2 = _ball_pos.y - goal_post2.y;
    float dist_to_post1 = sqrt(diff_x1*diff_x1 + diff_y1*diff_y1);
    float dist_to_post2 = sqrt(diff_x2*diff_x2 + diff_y2*diff_y2);
    float angle_interval = (dist_to_post1*dist_to_post1 + dist_to_post2*dist_to_post2 - goal_width*goal_width)/
            (2.0f * dist_to_post1*dist_to_post2);
    angle_interval = acos(angle_interval);
//    float center_dir = (attack_dir_?CV_PI - atan2(center_goal.y - _ball_pos.y,_ball_pos.x - center_goal.x):
//      atan2(center_goal.y - _ball_pos.y,center_goal.x - _ball_pos.x))*Math::RAD2DEG;
    float center_dir = atan2(center_goal.y - _ball_pos.y, center_goal.x - _ball_pos.x) * Math::RAD2DEG;
    if(center_dir < .0f)center_dir = 360.0f + center_dir;

//    std::cout << "Center Dir : " << center_dir << std::endl;
    //not yet added random dir, but the interval is already
    return center_dir;
}

void BallDetector::genRadialPattern() {
    for(int i = 1; i <= 10; i++){
        float angle_step = 360.0f/(8.0f*(float)i) * Math::DEG2RAD;
//        std::cout << "========================================" << std::endl;
        int total_nb = 8*i;
        for(int j=0;j<total_nb;j++){
            std::pair<int,int > nb_pattern;
            int radius=i;
            if(j <= total_nb/8)
                radius /= cos(angle_step*(float)j);
            else if (j <= (3*total_nb)/8)
                radius /= sin(angle_step*(float)j);
            else if (j <= (5*total_nb)/8)
                radius /= -cos(angle_step*(float)j);
            else if (j < (7*total_nb)/8)
                radius /= -sin(angle_step*(float)j);
            else
                radius /= cos(angle_step*(float)j);

            float est_x = (float)radius * cos(angle_step*(float)j);
            float est_y = (float)(-radius) * sin(angle_step*(float)j);
            // numerical error
            nb_pattern.first = est_x < -1e-4f ? std::floor(est_x):
                                                (est_x > 1e-4f ? std::ceil(est_x):std::abs(est_x));
            nb_pattern.second = est_y < -1e-4f ? std::floor(est_y):
                                                 (est_y > 1e-4f ? std::ceil(est_y):std::abs(est_y));
//            std::cout << est_x << " ; " << est_y << std::endl;
//            std::cout << nb_pattern.first << " ; " << nb_pattern.second << std::endl;
            radial_pattern_.push_back(nb_pattern);
        }
    }
}

void BallDetector::initializeFieldFeaturesData(){
    landmark_pos_.resize(4);
    line_segment_pos_.resize(11);
    //L - landmark
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH - GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH - GOAL_AREA_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[0].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    //T - landmark
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH)*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + FIELD_WIDTH)*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH - GOAL_AREA_WIDTH)>>1))*0.01f);
    landmark_pos_[1].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + FIELD_LENGTH, BORDER_STRIP_WIDTH + ((FIELD_WIDTH + GOAL_AREA_WIDTH)>>1))*0.01f);
    //X - landmark
    landmark_pos_[2].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + ((FIELD_WIDTH - CENTER_CIRCLE_DIAMETER)>>1))*0.01f);
    landmark_pos_[2].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + ((FIELD_WIDTH + CENTER_CIRCLE_DIAMETER)>>1))*0.01f);
    //Center circle
    landmark_pos_[3].emplace_back(cv::Point2f(BORDER_STRIP_WIDTH + (FIELD_LENGTH>>1), BORDER_STRIP_WIDTH + (FIELD_WIDTH>>1))*0.01f);

    //Vertical Line Segment
    line_segment_pos_[0] = {landmark_pos_[0][0].x,landmark_pos_[0][0].y,landmark_pos_[0][1].x,landmark_pos_[0][1].y};
    line_segment_pos_[1] = {landmark_pos_[0][2].x,landmark_pos_[0][2].y,landmark_pos_[0][3].x,landmark_pos_[0][3].y};
    line_segment_pos_[2] = {landmark_pos_[1][2].x,landmark_pos_[1][2].y,landmark_pos_[1][3].x,landmark_pos_[1][3].y};
    line_segment_pos_[3] = {landmark_pos_[0][4].x,landmark_pos_[0][4].y,landmark_pos_[0][5].x,landmark_pos_[0][5].y};
    line_segment_pos_[4] = {landmark_pos_[0][6].x,landmark_pos_[0][6].y,landmark_pos_[0][7].x,landmark_pos_[0][7].y};
    //Horizontal Line Segment
    line_segment_pos_[5] = {landmark_pos_[0][0].x,landmark_pos_[0][0].y,landmark_pos_[0][6].x,landmark_pos_[0][6].y};
    line_segment_pos_[6] = {landmark_pos_[1][0].x,landmark_pos_[1][0].y,landmark_pos_[0][2].x,landmark_pos_[0][2].y};
    line_segment_pos_[7] = {landmark_pos_[0][4].x,landmark_pos_[0][4].y,landmark_pos_[1][4].x,landmark_pos_[0][4].y};
    line_segment_pos_[8] = {landmark_pos_[1][1].x,landmark_pos_[1][1].y,landmark_pos_[0][3].x,landmark_pos_[0][3].y};
    line_segment_pos_[9] = {landmark_pos_[0][5].x,landmark_pos_[0][5].y,landmark_pos_[1][5].x,landmark_pos_[1][5].y};
    line_segment_pos_[10] = {landmark_pos_[0][1].x,landmark_pos_[0][1].y,landmark_pos_[0][7].x,landmark_pos_[0][7].y};
}
