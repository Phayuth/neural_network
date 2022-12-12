#include <iostream>
#include <torch/torch.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>

int main(int argc, char **argv)
{
	ros::init(argc, argv, "P");
	
	ros::NodeHandle n;

	ros::Publisher pub = n.advertise<std_msgs::Float64>("pub_number", 1000);

	ros::Rate loop_rate(100);

	ROS_INFO("Starting Publisher");

	at::Tensor a = at::ones({2, 2}, at::kInt);

	while (ros::ok())
	{
		torch::Tensor x = torch::randn({3,3} , torch::kCUDA);

		float y = x[1][1].item<float>(); 

		at::Tensor b = at::randn({2, 2});
	
		auto c = a + b.to(at::kInt);

		// std::cout << x << std::endl;

		std::cout << c << std::endl;
		
		std_msgs::Float64 msg;

		msg.data = y;

		pub.publish(msg);

		ros::spinOnce();

		loop_rate.sleep();
	}
	return 0;
}