#include <network.h>

#include <iostream>
#include <torch/torch.h>

int main(int argc, char const *argv[])
{
	Net network(50,10);
	std::cout << network << std::endl;

	torch::Tensor x, output;

	x = torch::randn({2,50}, torch::kCUDA);

	std::cout << "Input x is : " <<std::endl;
	std::cout << x << std::endl;

	output = network->forward(x);

	std::cout << "Output is :" <<std::endl;

	std::cout << output <<std::endl;
	return 0;
}