#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>

int main(int argc, char const *argv[])
{
	torch::jit::script::Module net = torch::jit::load("../models/net.pt");

	torch::Tensor x = torch::randn({1,100});

	std::cout << x << std::endl;

	std::vector<torch::jit::IValue> input;

	input.push_back(x);

	for (int i = 0; i < 10; ++i)
	{
		auto out = net.forward(input);

		std::cout << out << std::endl;

		// std::cout << typeid(out).name() << std::endl;
	}
	

	return 0;
}