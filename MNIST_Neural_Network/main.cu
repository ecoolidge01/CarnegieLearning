#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <random>
#include <math.h>


void swapEndian(int &i) {
	i = (0xff&(i >> 24)) | (0xff00&(i >> 8)) | (0xff0000&(i << 8)) | (0xff000000&(i << 24));
}

float *** getImages(std::string filename, int & numImages, int & numRows, int & numCols) {
	std::ifstream file;
	int magicNum;
	file.open(filename.c_str(), std::ios::binary);
	if(!(file)){
		std::cout << "Error: could not open images file\n";
		exit(2);
	}
	file.read(reinterpret_cast<char*>(&magicNum), sizeof(magicNum));
	file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	file.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	file.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

	swapEndian(numImages);
	swapEndian(numRows);
	swapEndian(numCols);

	float ***retArr = new float**[numImages];
	for (int i = 0; i < numImages; i++){
		retArr[i] = new float*[numRows];
		for (int r = 0; r < numRows; r++){
			retArr[i][r] = new float[numCols];
			for (int c = 0; c < numCols; c++){
				unsigned char temp;
				file.read(reinterpret_cast<char*>(&temp), 1);
				retArr[i][r][c] = ((int)temp / 127.5) - 1.0;
			}
		}
	}
	return retArr;
}

int * getLabels(const std::string filename, int & numLabels){
	std::ifstream file;
	int magicNum;
	file.open(filename.c_str(), std::ios::binary);
	if(!(file)){
		std::cout << "Error: could not open labels file\n";
		exit(2);
	}
	file.read(reinterpret_cast<char*>(&magicNum), 4);
	file.read(reinterpret_cast<char*>(&numLabels), 4);

	swapEndian(numLabels);

	int *retArr = new int[numLabels];
	for (int i = 0; i < numLabels; i++) {
		char temp;
		file.read(reinterpret_cast<char*>(&temp), 1);
		retArr[i] = temp;
	}
	return retArr;
}

float *** getWeightDeriv(int reluNodes, int numRows, int numCols){
	float *** retArr = new float**[reluNodes];
	for (int i = 0; i < reluNodes; i++) {
		retArr[i] = new float*[numRows];
		for (int j = 0; j < numRows; j++) {
			retArr[i][j] = new float[numCols];
			for (int k = 0; k < numCols; k++) {
				retArr[i][j][k] = 0;
			}
		}
	}
	return retArr;
}

float *** getReluWeights(int numNodes, int numRows, int numCols){
	std::default_random_engine eng;
	eng.seed(time(NULL));
	std::normal_distribution<float> init;

	float *** retArr = new float**[numNodes];
	for (int i = 0; i < numNodes; i++) {
		retArr[i] = new float*[numRows];
		for (int j = 0; j < numRows; j++) {
			retArr[i][j] = new float[numCols];
			for (int k = 0; k < numCols; k++) {
				retArr[i][j][k] = init(eng) / sqrt(numNodes);
			}
		}
	}
	return retArr;
}

float ** getReluWeightDeriv(int numClasses, int reluNodes){
	float ** retArr = new float*[numClasses];
	for (int i = 0; i < numClasses; i++) {
		retArr[i] = new float[reluNodes];
		for (int j = 0; j < reluNodes; j++) {
			retArr[i][j] = 0;
		}
	}
	return retArr;
}

float * reluDropoutForward(float ** image, float *** weights, float * bias, int reluNodes, int numRows, int numCols){
	srand(time(NULL));
	float * retArr = new float[reluNodes];

	for (int i = 0; i < reluNodes; i++){
		float sum = 0.0;
		for (int r = 0; r < numRows; r++){
			for (int c = 0; c < numCols; c++){
				sum += weights[i][r][c] * image[r][c];
			}
		}
		sum += bias[i];
		int temp = rand() % 1000;
		if (temp < 4 || sum <= 0.0){
			retArr[i] = 0.0;
		}
		else{
			retArr[i] = sum;
		}
	}
	return retArr;
}

void reluBackward(float * output, float * upstream, float ** originalInput, float * biasDeriv, float *** originalWeights, float *** weightDeriv, float batchSize, int numRows, int numCols, int reluNodes){
	for (int i = 0; i < reluNodes; i++){
		for (int j = 0; j < numRows; j++){
			for (int k = 0; k < numCols; k++){
				weightDeriv[i][j][k] += (upstream[i] * originalInput[j][k]) / batchSize;
			}
		}
		biasDeriv[i] = upstream[i] / batchSize;
	}
}

__global__ void reluUpdateGPU(float * weights, float * weightDeriv, float * bias, float * biasDeriv, float learningRate, int reluNodes, int numRows, int numCols){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index <= reluNodes*numRows*numCols){
		weights[index] -= learningRate * weightDeriv[index];
		weightDeriv[index] = 0;
		if (index % (numRows * numCols) == 0) {
			bias[index / (numRows * numCols)] -= learningRate * biasDeriv[index / (numRows * numCols)];
			biasDeriv[index / (numRows * numCols)] = 0;
		}
	}
}


float ** getLinearWeights(int numClasses, int reluNodes){
	std::default_random_engine eng;
	eng.seed(456);
	std::normal_distribution<float> init;
	float **ret = new float*[numClasses];

	for (int i = 0; i < numClasses; i++) {
		ret[i] = new float[reluNodes];
		for (int j = 0; j < reluNodes; j++){
			ret[i][j] = init(eng) / sqrt(numClasses);
		}
	}
	return ret;
}

float * linearForward(float * input, float ** weights, float * bias, int reluNodes, int numClasses){
	float *retArr = new float[numClasses];
	for (int i = 0; i < numClasses; i++){
		float sum = 0.0;
		for (int j = 0; j < reluNodes; j++){
			sum += weights[i][j] * input[j];
		}
		sum += bias[i];
		retArr[i] = sum;
	}
	return retArr;
}

void linearBackward(float * output, float * upstream, float * originalInput, float * biasDeriv, float ** originalWeights, float ** weightDeriv, float batchSize, int reluNodes, int numClasses){
	for (int i = 0; i < numClasses; i++){
		for (int j = 0; j < reluNodes; j++){
			output[j] = 0;
		}
		for (int j = 0; j < reluNodes; j++){
			output[j] += upstream[i] * originalWeights[i][j];
			weightDeriv[i][j] += (upstream[i] * originalInput[j]) / batchSize;
		}
		biasDeriv[i] = upstream[i] / batchSize;
	}
}

__global__ void linearUpdate(float * weight, float * weightDeriv, float * bias, float * biasDeriv, float learningRate, int numClasses, int reluNodes){
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < numClasses*reluNodes){
		weight[index] -= learningRate * weightDeriv[index];
		weightDeriv[index] = 0;
		if (index % (reluNodes) == 0)
		{
			bias[index / reluNodes] -= learningRate * biasDeriv[index / reluNodes];
			biasDeriv[index / reluNodes] = 0;
		}
	}
}

float * softMax(float * input, int numClasses){
	float * retArr = new float[numClasses];
	float sum = 0;
	float max = input[0];
	for (int i = 0; i < numClasses; i++){
		if(input[i] > max){
			max = input[i];
		}
	}
	for (int i = 0; i < numClasses; i++){
		retArr[i] = exp(input[i] - max);
		sum += retArr[i];
	}
	for (int i = 0; i < numClasses; i++){
		retArr[i] = retArr[i] / sum;
	}
	return retArr;
}

void softMaxBackward(float * output, float * upstream, float * originalInput, int numClasses){
	for (int i = 0; i < numClasses; i++) {
		output[i] = 0.0;
		for (int j = 0; j < numClasses; j++){
			if (i == j) {
				output[i] += upstream[j] * (originalInput[j] * (1 - originalInput[i]));
			}
			else 
			{
				output[i] += upstream[j] * (-originalInput[i] * originalInput[j]);
			}
		}
	}
}

float cross_entropy(float * soft_maxes, int correct){
	return -1 * log(soft_maxes[correct]);
}

void flattenRelu(float * flatWeights, float * flatWeightDeriv, float *** reluWeights, float *** imageDeriv, int reluNodes, int numRows, int numCols){
	for (int i = 0; i < reluNodes; i++){
		for (int j = 0; j < numRows; j++){
			for (int k = 0; k < numCols; k++){
				flatWeights[i*numRows*numCols + j*numRows + k] = reluWeights[i][j][k];
				flatWeightDeriv[i*numRows*numCols + j*numRows + k] = imageDeriv[i][j][k];
			}
		}
	}
}

void unflattenRelu(float *** reluWeights, float *** image_weight_deriv, float * h_fcl1_weights, float * h_fcl1_weight_deriv, int reluNodes, int numRows, int numCols){
	for (int i = 0; i < reluNodes; i++){
		for (int j = 0; j < numRows; j++){
			for (int k = 0; k < numCols; k++){
				reluWeights[i][j][k] = h_fcl1_weights[i*numRows*numCols + j*numRows + k];
				image_weight_deriv[i][j][k] = h_fcl1_weight_deriv[i*numRows*numCols + j*numCols + k];
			}
		}
	}
}

void flattenLinear(float * flatLinearWeights, float * flatReluWeightDeriv, float ** linearWeights, float ** reluWeightDeriv, int numClasses, int reluNodes){
	for (int i = 0; i < numClasses; i++){
		for (int j = 0; j < reluNodes; j++){
			flatLinearWeights[i*reluNodes + j] = linearWeights[i][j];
			flatReluWeightDeriv[i*reluNodes + j] = reluWeightDeriv[i][j];
		}
	}
}

void unflattenLinear(float ** linearWeights, float ** reluWeightDeriv, float * flatLinearWeights, float * flatReluWeightDeriv, int numClasses, int reluNodes){
	for (int i = 0; i < numClasses; i++){
		for (int j = 0; j < reluNodes; j++){
			linearWeights[i][j] = flatLinearWeights[i*reluNodes + j];
			reluWeightDeriv[i][j] = flatReluWeightDeriv[i*reluNodes + j];
		}
	}
}

int main(int argc, char ** argv){
	//getting training files
	int numImages;
	int numRows;
	int numCols;
	int numLabels;
	int numClasses = 10;
	int reluNodes = 1024;

	const int batchSize = 100;
	int totalBatches = 150;
	float learningRate = .1;
	int validationFrequency = 10;

	float *** trainImages = getImages("./MNIST_Data/train-images.idx3-ubyte", numImages, numRows, numCols);
	int * trainLabels = getLabels("./MNIST_Data/train-labels.idx1-ubyte", numLabels);

	// std::cout << numImages << ", " << numRows << ", " << numCols << "\n";
	// std::cout << numLabels << "\n";

	std::cout << "Files Opened, Starting Network Training\n\n\n";

	//pre-declaring vars to use in testing
	//relu vars
	float * reluResult = new float[reluNodes]();
	float *** reluWeights = getReluWeights(reluNodes, numRows, numCols);
	float * reluBias = new float[reluNodes]();
	float ** reluWeightDeriv = getReluWeightDeriv(numClasses, reluNodes);
	float * reluDeriv = new float[reluNodes]();
	float * reluBiasDeriv = new float[reluNodes]();
	float *** imageDeriv = getWeightDeriv(reluNodes, numRows, numCols);

	float * flatWeights = new float[numRows*numCols*reluNodes];
	float * flatWeightDeriv = new float[numRows*numCols*reluNodes];

	float * flatWeightsGPU;
	float * flatWeightsDerivGPU;
	float * reluBiasGPU;
	float * reluBiasDerivGPU;

	float * flatLinearWeightsGPU;
	float * flatReluWeightDerivGPU;
	float * linearBiasGPU;
	float * linearBiasDerivGPU;

	cudaMalloc(&flatWeightsGPU, numRows*numCols*reluNodes*sizeof(float));
	cudaMalloc(&flatWeightsDerivGPU, numRows*numCols*reluNodes*sizeof(float));
	cudaMalloc(&reluBiasGPU, reluNodes*sizeof(float));
	cudaMalloc(&reluBiasDerivGPU, reluNodes*sizeof(float));
	
	cudaMalloc(&flatLinearWeightsGPU, numClasses*reluNodes*sizeof(float));
	cudaMalloc(&flatReluWeightDerivGPU, numClasses*reluNodes*sizeof(float));
	cudaMalloc(&linearBiasGPU, numClasses*sizeof(float));
	cudaMalloc(&linearBiasDerivGPU, numClasses*sizeof(float));

	//linear vars
	float * linearResult = new float[numClasses]();
	float ** linearWeights = getLinearWeights(numClasses, reluNodes);
	float * linearBias = new float[numClasses]();
	float * linearBiasDeriv = new float[numClasses]();
	float * linearDeriv = new float[reluNodes]();
	float * flatLinearWeights = new float[numClasses*reluNodes];
	float * flatReluWeightDeriv = new float[numClasses*reluNodes];

	//softMax cross entropy vars
	float * softMaxResult = new float[numClasses]();
	float * crossEntropyResult = new float[numClasses]();
	float * softMaxDeriv = new float[numClasses]();

	// training -------------------------------------------------------------------
	// for epoch
	// for batchCount
	for(int batchCount = 0; batchCount < totalBatches; batchCount++){
		int correct = 0;
		int incorrect = 0;
		for(int batchImage = 0; batchImage < batchSize; batchImage++){

			//forward pass -------------------------------------------------------------
			//relu/dropout forward
			reluResult = reluDropoutForward(trainImages[batchCount + batchImage], reluWeights, reluBias, reluNodes, numRows, numCols);

			//linear forward
			linearResult = linearForward(reluResult, linearWeights, linearBias, reluNodes, numClasses);

			//softmax and cross entropy
			softMaxResult = softMax(linearResult, numClasses);
			int expected = trainLabels[batchCount + batchImage];

			int calculated = std::distance(softMaxResult, std::max_element(softMaxResult, softMaxResult + numClasses));
			if (calculated == expected){
				correct += 1;
			}
			else{
				incorrect +=1;
			}

			for (int i = 0; i < numClasses; i++){
				crossEntropyResult[i] = 0.0;
			}
			crossEntropyResult[expected] = -1 / softMaxResult[expected];

			//backpass ---------------------------------------------------------------------
			//soft max backpass
			softMaxBackward(softMaxDeriv, crossEntropyResult, softMaxResult, numClasses);

			//linear layer backpass
			linearBackward(linearDeriv, softMaxDeriv, reluResult, linearBiasDeriv, linearWeights, reluWeightDeriv, batchSize, reluNodes, numClasses);

			//relu backpass
			reluBackward(reluDeriv, linearDeriv, trainImages[batchCount + batchImage], reluBiasDeriv, reluWeights, imageDeriv, batchSize, numRows, numCols, reluNodes);	

			//updating weights
			flattenRelu(flatWeights, flatWeightDeriv, reluWeights, imageDeriv, reluNodes, numRows, numCols);	

			cudaMemcpy(flatWeightsGPU, flatWeights, numRows*numCols*reluNodes*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(flatWeightsDerivGPU, flatWeightDeriv, numRows*numCols*reluNodes*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(reluBiasGPU, reluBias, reluNodes*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(reluBiasDerivGPU, reluBiasDeriv, reluNodes*sizeof(float), cudaMemcpyHostToDevice);	

			reluUpdateGPU<<<(numRows*numCols), reluNodes>>>(flatWeightsGPU, flatWeightsDerivGPU, reluBiasGPU, reluBiasDerivGPU, learningRate, reluNodes, numRows, numCols);

			cudaMemcpy(flatWeights, flatWeightsGPU, numRows*numCols*reluNodes*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(flatWeightDeriv, flatWeightsDerivGPU, numRows*numCols*reluNodes*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(reluBias, reluBiasGPU, reluNodes*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(reluBiasDeriv, reluBiasDerivGPU, reluNodes*sizeof(float), cudaMemcpyDeviceToHost);

			unflattenRelu(reluWeights, imageDeriv, flatWeights, flatWeightDeriv, reluNodes, numRows, numCols);

			flattenLinear(flatLinearWeights, flatReluWeightDeriv, linearWeights, reluWeightDeriv, numClasses, reluNodes);

			cudaMemcpy(flatLinearWeightsGPU, flatLinearWeights, numClasses*reluNodes*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(flatReluWeightDerivGPU, flatReluWeightDeriv, numClasses*reluNodes*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(linearBiasGPU, linearBias, numClasses*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(linearBiasDerivGPU, linearBiasDeriv, numClasses*sizeof(float), cudaMemcpyHostToDevice);

			linearUpdate<<<numClasses, reluNodes>>>(flatLinearWeightsGPU, flatReluWeightDerivGPU, linearBiasGPU, linearBiasDerivGPU, learningRate, numClasses, reluNodes);

			cudaMemcpy(flatLinearWeights, flatLinearWeightsGPU, numClasses*reluNodes*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(flatReluWeightDeriv, flatReluWeightDerivGPU, numClasses*reluNodes*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(linearBias, linearBiasGPU, numClasses*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(linearBiasDerivGPU, linearBiasDerivGPU, numClasses*sizeof(float), cudaMemcpyDeviceToHost);

			unflattenLinear(linearWeights, reluWeightDeriv, flatLinearWeights, flatReluWeightDeriv, numClasses, reluNodes);

		}
		//validation
		if(batchCount%validationFrequency == 0){
			std::cout <<  " batch: " << batchCount << " accuracy: " << (float)((float)correct/(float)(correct+incorrect)) << "\n";
		}
	}

	std::cout << "\n\nFinished Training Now Testing\n\n";
	//testing
	int correct = 0;
	int incorrect = 0;
	
	float *** testImages = getImages("./MNIST_Data/train-images.idx3-ubyte", numImages, numRows, numCols);
	int * testLabels = getLabels("./MNIST_Data/train-labels.idx1-ubyte", numLabels);

	for(int batchImage = 0; batchImage < batchSize; batchImage++){

		//relu/dropout forward
		reluResult = reluDropoutForward(trainImages[batchImage], reluWeights, reluBias, reluNodes, numRows, numCols);

		//linear forward
		linearResult = linearForward(reluResult, linearWeights, linearBias, reluNodes, numClasses);

		//softmax and cross entropy
		softMaxResult = softMax(linearResult, numClasses);
		int expected = trainLabels[batchImage];

		int calculated = std::distance(softMaxResult, std::max_element(softMaxResult, softMaxResult + numClasses));
		if (calculated == expected){
			correct += 1;
		}
		else{
			incorrect +=1;
		}
	}
	std::cout <<  "Test Accuracy: " << (float)((float)correct/(float)(correct+incorrect)) << "\n";




	std::cout << "\nFinished Code!\n";
}
