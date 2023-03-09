#include <iostream>
#include <math.h>
#include <windows.h>

double SigmoidFunc(double _neuronX);

int main(void) {

	std::cout << "\n\tNPC Decision ANN\n" << std::endl 
		<< "-----------------------------------" << std::endl << "\n";

	//Set number of iterations
	int maxIteration = 1000000;

	//Initial weights between layers of model
	double w_1to3 = 0.5;
	double w_2to3 = 0.4;
	double w_1to4 = 0.9;
	double w_2to4 = 1.0;
	double w_3to5 = -1.2;
	double w_4to5 = 1.1;

	//Initial Bias
	double b_3 = 0.8;
	double b_4 = -0.1;
	double b_5 = 0.3;

	//Set Learning Rate
	double learnRate = 0.638;

	//Training data for reference
	int x1[4] = { 1, 0, 1, 0 };
	int x2[4] = { 1, 1, 0, 0 };
	int y[4] = { 0, 1, 1, 0 };

	//Current Iteration Number
	int currIteration = 1;

	//Show Starting Weights and Bias
	std::cout << "Iteration No. = " << currIteration - 1 << std::endl
		<< "\nCurrent Weights + Bias:" << std::endl << std::endl
		<< "\tWeight 1 to 3: " << w_1to3 << std::endl
		<< "\tWeight 2 to 3: " << w_2to3 << std::endl
		<< "\tWeight 1 to 4: " << w_1to4 << std::endl
		<< "\tWeight 2 to 4: " << w_2to4 << std::endl
		<< "\tWeight 3 to 5: " << w_3to5 << std::endl
		<< "\tWeight 4 to 5: " << w_4to5 << std::endl << std::endl
		<< "\tBias 3: " << b_3 << std::endl
		<< "\tBias 4: " << b_4 << std::endl
		<< "\tBias 5: " << b_5 << std::endl << "\n"
		<< "-----------------------------------" << std::endl << "\n";

	//While under the max amount of iterations
	while (currIteration <= maxIteration) {

		//Termination Condition
		double epochSumError = 0;

		for (int i = 0; i < 4; i++) {

			//FEEDING FORWARD

			//Calculate Neuron 3 Output and Activation - Sigmoid Function
			double neuron3X = x1[i] * w_1to3 + x2[i] * w_2to3 + b_3;
			double neuron3Y = SigmoidFunc(neuron3X);

			//Calculate Neuron 4 Output and Activation - Sigmoid Function
			double neuron4X = x1[i] * w_1to4 + x2[i] * w_2to4 + b_4;
			double neuron4Y = SigmoidFunc(neuron4X);

			//Calculate Neuron 5 Output and Activation - Sigmoid Function
			double neuron5X = neuron3Y * w_3to5 + neuron4Y * w_4to5 + b_5;
			double neuron5Y = SigmoidFunc(neuron5X);

			//BACK PROPAGATION

			//Calculate Error
			double error_5 = y[i] - neuron5Y;

			//Neuron 5 Adjustment
			double delta_5 = neuron5Y * (1 - neuron5Y) * error_5;
			double w_3to5curr = w_3to5; //Save Weight 3 to 5
			double w_4to5curr = w_4to5; //Save Weight 4 to 5
			w_3to5 = w_3to5 + learnRate * neuron3Y * delta_5;
			w_4to5 = w_4to5 + learnRate * neuron4Y * delta_5;
			b_5 = b_5 + learnRate * 1 * delta_5;

			//Neuron 3 Adjustment
			double delta_3 = neuron3Y * (1 - neuron3Y) * delta_5 * w_3to5curr;
			w_1to3 = w_1to3 + learnRate * x1[i] * delta_3;
			w_2to3 = w_2to3 + learnRate * x2[i] * delta_3;
			b_3 = b_3 + learnRate * 1 * delta_3;

			//Neuron 4 Adjustment
			double delta_4 = neuron4Y * (1 - neuron4Y) * delta_5 * w_4to5curr;
			w_1to4 = w_1to4 + learnRate * x1[i] * delta_4;
			w_2to4 = w_2to4 + learnRate * x2[i] * delta_4;
			b_4 = b_4 + learnRate * 1 * delta_4;

			//Accumulating Squared Errors and Recalculating error_5
			double tNeuron3X = x1[i] * w_1to3 + x2[i] * w_2to3 + b_3;
			double tNeuron3Y = SigmoidFunc(tNeuron3X);
			double tNeuron4X = x1[i] * w_1to4 + x2[i] * w_2to4 + b_4;
			double tNeuron4Y = SigmoidFunc(tNeuron4X);
			double tNeuron5X = tNeuron3Y * w_3to5 + tNeuron4Y * w_4to5 + b_5;
			double tNeuron5Y = SigmoidFunc(tNeuron5X);
			double tError_5 = y[i] - tNeuron5Y;

			epochSumError = epochSumError + pow(tError_5, 2);

		}

		//Show Weights and Bias Every 1000 Iterations
		if (currIteration % 1000 == 0) {

			Sleep(100);

			std::cout << "Iteration No. = " << currIteration << std::endl
				<< "\nCurrent Weights + Bias:" << std::endl << std::endl
				<< "\tWeight 1 to 3: " << w_1to3 << std::endl
				<< "\tWeight 2 to 3: " << w_2to3 << std::endl
				<< "\tWeight 1 to 4: " << w_1to4 << std::endl
				<< "\tWeight 2 to 4: " << w_2to4 << std::endl
				<< "\tWeight 3 to 5: " << w_3to5 << std::endl
				<< "\tWeight 4 to 5: " << w_4to5 << std::endl << std::endl
				<< "\tBias 3: " << b_3 << std::endl
				<< "\tBias 4: " << b_4 << std::endl
				<< "\tBias 5: " << b_5 << std::endl << "\n"
				<< "-----------------------------------" << std::endl << "\n";

		}

		//Show Starting Weights and Bias
		if (epochSumError < 0.001) {

			Sleep(1);

			std::cout << "Final Iteration No. = " << currIteration << std::endl
				<< "\nFinal Weights + Bias:" << std::endl << std::endl
				<< "\tWeight 1 to 3: " << w_1to3 << std::endl
				<< "\tWeight 2 to 3: " << w_2to3 << std::endl
				<< "\tWeight 1 to 4: " << w_1to4 << std::endl
				<< "\tWeight 2 to 4: " << w_2to4 << std::endl
				<< "\tWeight 3 to 5: " << w_3to5 << std::endl
				<< "\tWeight 4 to 5: " << w_4to5 << std::endl << std::endl
				<< "\tBias 3: " << b_3 << std::endl
				<< "\tBias 4: " << b_4 << std::endl
				<< "\tBias 5: " << b_5 << std::endl << "\n"
				<< "-----------------------------------" << std::endl;
			break;

		}

		currIteration = currIteration + 1;

	}

	std::cout << "\n";
	system("PAUSE");

}

double SigmoidFunc(double _neuronX) {

	return 1 / (1 + pow(exp(1), -_neuronX));

}
