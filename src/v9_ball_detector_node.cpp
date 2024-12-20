#include <v9_ball_detector/v9_ball_detector.h>
#include <immintrin.h>

int main(int argc,char **argv){

    ros::init(argc,argv,"v9_ball_detector_node");

    BallDetector ball_detector;

    ros::Rate loop_rate(30);

    while(ros::ok()){

        ball_detector.process();
        // ball_detector.processBaru();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
