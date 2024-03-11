#include <iostream>
#include <torch/torch.h>

int main(int argc, char const *argv[])
{
	std::cout << "HI ALL " << std::endl;
	torch::Tensor x = torch::randn({3,3} , torch::kCUDA);
	std::cout << x << std::endl;
	return 0;
}