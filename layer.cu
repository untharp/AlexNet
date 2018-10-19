#include "layer.h"
#include <cstdio>
//#include "mnist.h"

// Constructor
Layer::Layer(long int M, long int N, long int O)
{
	this->M = M;
	this->N = N;
	this->O = O;
	float h_bias[N];
	int tmp_m = 0;
	if (M <= 3*3*192){
		tmp_m = M;

	}
	else{
		tmp_m = M/64;

	}
	float h1_weight[N][tmp_m];
//	float h2_weight[N][M/2];
	float h_output[O];
	float h_preact[O];
	cudaEvent_t start, stop;
  	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
#if 1
	output = NULL;
	preact = NULL;
	act_result = NULL;//for normalization
	bias   = NULL;
	weight = NULL;
	L_output = NULL; //output result need to be copied from GPU to CPU to print out

	for (int i = 0; i < N; ++i) {
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/*h_bias[i] = 0.0f;*/

		for (int j = 0; j < tmp_m; ++j) {
			h1_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_weight[i][j] = 0.05f;*/
//			h2_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);

		}	
	}

	for (int i = 0; i < O; i++){
		h_output[i] = 0.f;
		h_preact[i] = 0.f;
	}
	cudaMalloc(&output, sizeof(float) * O);
/*	for (int i = 0; i < O; i++){
		*(output + i*sizeof(float)) = 0.f;
	}*/
	cudaMalloc(&preact, sizeof(float) * O);
/*	for (int i = 0; i < O; i++){
		*(preact + i*sizeof(float)) = 0.f;
	}*/

	cudaMalloc(&act_result, sizeof(float) * O);//for normalization

	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&L_output, sizeof(float) * O);

	cudaMalloc(&weight, sizeof(float) * M * N);
	cudaMalloc(&d_output, sizeof(float) * O);
	cudaMalloc(&d_preact, sizeof(float) * O);
	cudaMalloc(&d_act_result, sizeof(float) * O);//for normalization
	cudaMalloc(&d_weight, sizeof(float) * M * N);
#endif	
	cudaEventRecord(start,0);
#if 1
	cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
	if (M <= 3*3*192){
		cudaMemcpy(weight, h1_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	}
	else{
		for(int i = 0; i < 64; i++){
			cudaMemcpy(weight + (i * sizeof(float) * (M/64)), h1_weight, sizeof(float) * (M/64) * N, cudaMemcpyHostToDevice);
		}

	}
	//cudaMemcpy(weight + sizeof(float) * (M/2) * N, h1_weight, (M/2) * N, cudaMemcpyHostToDevice);
	
	cudaMemcpy(output, h_output, sizeof(float) * O, cudaMemcpyHostToDevice);
	cudaMemcpy(preact, h_preact, sizeof(float) * O, cudaMemcpyHostToDevice);
#endif	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
#if 1
	float milliseconds;
  	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stdout ,"millisecond : %f\n", milliseconds);
#endif
	cudaEventDestroy(start);
  	cudaEventDestroy(stop);
}

// Destructor
Layer::~Layer()
{
	cudaFree(output);
	cudaFree(preact);
	cudaFree(act_result);//for normalization

	cudaFree(bias);

	cudaFree(weight);

	cudaFree(d_output);
	cudaFree(d_preact);
	cudaFree(d_act_result);//for normalization
	cudaFree(d_weight);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
	cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
	cudaMemset(output, 0x00, sizeof(float) * O);
	cudaMemset(preact, 0x00, sizeof(float) * O);
	cudaMemset(act_result, 0x00, sizeof(float) * O);//for normalization
	cudaMemset(L_output, 0x00, sizeof(float) * O);//for layer result print

}

void Layer::bp_clear()
{
	cudaMemset(d_output, 0x00, sizeof(float) * O);
	cudaMemset(d_preact, 0x00, sizeof(float) * O);
	cudaMemset(d_act_result, 0x00, sizeof(float) * O);//for normalization
	cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
}

void Layer::Output_Layer(float *data){
		cudaMemcpy(L_output, data, sizeof(float) * O, cudaMemcpyDeviceToHost);

}


__device__ float step_function(float v) //Sigmoid function::Activation Function
{
	//return 1 / (1 + exp(-v));
	return max(0.f, v);
}

__device__ float normalization(float *input, float u, int idx, const int O, const int N){
	
	int i = ((idx/(O/N))    % N);
	int j, k;
	float tmp_sum = 0.f;
	if ((i - (5/2)) > 0){
		j = i - (5/2);
	}
	else{
		j = 0;
	}
	if ((i + (5/2)) > (N - 1)){
		k = (N - 1);
	}
	else{
		k = (i + (5/2));
	}
	for (int tmp_i = j; tmp_i < k; tmp_i++){
		tmp_sum += pow(input[(idx - (i - 1) * (O/N)) + ((j - 1)*(O/N)) ], 2);
	}	
	return u/pow((2 + (0.0001 * tmp_sum)), 0.75);
}//normalization

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void normalization_function(float *input, float *output, const int O, const int N){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	for (int idx = O * pos / size; idx < O * (pos+1) / size; ++idx) {
		output[idx] = normalization(input, input[idx], idx, O, N);
	}


}//normalization

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;// find specific index/thread in GPU
	const int size = blockDim.x * gridDim.x; // the size of all index/thread in GPU

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

//conv1 227*227*3 to 55*55*96
__global__ void fp_preact_c1(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
//printf("%d, %d, %d is running\n", blockIdx.x, threadIdx.x, pos);
	const int N = 11*11*96*55*55*3;
	//const int N = 55*55*96;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;

		const int i6 = ((idx /= 1       ) % 3);
		const int i1 = ((idx /= 3	) % 11);
		const int i2 = ((idx /= 11	) % 11);
		const int i3 = ((idx /= 11	) % 96);
		const int i4 = ((idx /= 96	) % 55);
		const int i5 = ((idx /= 55	) % 55);
		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2][i6] * input[i4*4 + i1][i5*4 + i2][i6]);
		//times 4 means stride is 4
	}
}

__global__ void fp_bias_c1(float preact[96][55][55], float bias[96])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 96*55*55;
	//const int N = 55*55;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 96);
		const int i2 = ((idx /= 96	) % 55);
		const int i3 = ((idx /= 55	) % 55);

		preact[i1][i2][i3] += bias[i1];
	}
}

//pooling 1 55*55*96 to 31*31*96
__global__ void fp_preact_p1(float input[96][55][55], float preact[96][31][31])
{
	pool_data pos;
	pos.x = threadIdx.x;
	pos.y = threadIdx.y;
	//const int size = blockDim.x * gridDim.x;
	
	//const int N = 8*27*27*96;
	float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; j++){
			if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]){
				tmp_preact =  input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
			}
		}
	}
	preact[blockIdx.x][pos.x + 2][pos.y + 2] = tmp_preact;
	/*for (int n = N * pos.x / size; n < N * (pos.x+1) / size; ++n) {
			int idx = n;
			const int i1 = ((idx /= 1	) % 3);
			const int i2 = ((idx /= 3	) % 3);
			const int i3 = ((idx /= 3	) % 96);
			const int i4 = ((idx /= 96	) % 27);
			const int i5 = ((idx /= 27	) % 27);

		atomicAdd(&preact[i3][i4][i5], input[i3][i4 * 2 + i1][i5 * 2 + i2]);
	}*/
}


//conv2 31*31*96 to 27*27*128
__global__ void fp_preact_c2(float input[96][31][31], float preact[256][27][27], float weight[256][96][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5*5*96*256*27*27;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 5);
		const int i2 = ((idx /= 5	) % 5);				
		const int i3 = ((idx /= 5	) % 96);
		const int i6 = ((idx /= 96	) % 256);
		const int i4 = ((idx /= 256	) % 27);
		const int i5 = ((idx /= 27	) % 27);

		atomicAdd(&preact[i6][i4][i5], weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2]);
	}
}

__global__ void fp_bias_c2(float preact[256][27][27], float bias[256])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 256*27*27;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 256);
		const int i2 = ((idx /= 256	) % 27);
		const int i3 = ((idx /= 27	) % 27);

		preact[i1][i2][i3] += bias[i1];
	}
}

//pooling 2 27*27*128 to 15*15*128
__global__ void fp_preact_p2(float input[256][27][27], float preact[256][15][15])
{
	pool_data pos;
	pos.x = threadIdx.x;
	pos.y = threadIdx.y;
//	N = 13 * 13 * 256;	
	float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; j++){
			if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]){
				tmp_preact =  input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
			}
		}
	}
	preact[blockIdx.x][pos.x + 2][pos.y + 2] = tmp_preact;

	/*const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 2*2*16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 2);
		const int i2 = ((idx /= 2	) % 2);
		const int i3 = ((idx /= 2	) % 16);
		const int i4 = ((idx /= 16	) % 4);
		const int i5 = ((idx /= 4	) % 4);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 2 + i1][i5 * 2 + i2]);
	}*/
}

/*__global__ void fp_bias_s2(float preact[16][4][4], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 4);
		const int i3 = ((idx /= 4	) % 4);

		preact[i1][i2][i3] += bias[0];
	}
}*/

//conv3 256*15*15 to 13*13*384
__global__ void fp_preact_c3(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 5*5*96*256*27*27;
	const int N = 3*3*256*384*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 3);
		const int i2 = ((idx /= 3	) % 3);				
		const int i3 = ((idx /= 3	) % 256);
		const int i6 = ((idx /= 256	) % 384);
		const int i4 = ((idx /= 384	) % 13);
		const int i5 = ((idx /= 13	) % 13);

		atomicAdd(&preact[i6][i4][i5], weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2]);
	}

	/*for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);				
		const int i3 = ((idx /= 4	) % 16);
		const int i6 = ((idx /= 16	) % 120);
		atomicAdd(&preact[i6], weight[i6][i3][i1][i2] * input[i3][i1][i2]);
	}*/
}


__global__ void fp_bias_c3(float preact[384][13][13], float bias[384])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 384*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 384);
		const int i2 = ((idx /= 384	) % 13);
		const int i3 = ((idx /= 13	) % 13);

		preact[i1][i2][i3] += bias[i1];
	}

	/*for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 120);

		preact[i1] += bias[i1];
	}*/
}

__global__ void fp_preact_c4(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 3*3*256*384*13*13;
	const int N = 3*3*384*384*11*11;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 3);
		const int i2 = ((idx /= 3	) % 3);				
		const int i3 = ((idx /= 3	) % 384);
		const int i6 = ((idx /= 384	) % 384);
		const int i4 = ((idx /= 384	) % 11);
		const int i5 = ((idx /= 11	) % 11);

		atomicAdd(&preact[i6][i4 + 1][i5 + 1], weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2]);
	}

}

__global__ void fp_bias_c4(float preact[384][13][13], float bias[384])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 384*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 384);
		const int i2 = ((idx /= 384	) % 13);
		const int i3 = ((idx /= 13	) % 13);

		preact[i1][i2][i3] += bias[i1];
	}

}

__global__ void fp_preact_c5(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	//const int N = 3*3*256*384*13*13;
	const int N = 3*3*256*384*11*11;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 3);
		const int i2 = ((idx /= 3	) % 3);				
		const int i3 = ((idx /= 3	) % 384);
		const int i6 = ((idx /= 384	) % 256);
		const int i4 = ((idx /= 256	) % 11);
		const int i5 = ((idx /= 11	) % 11);

		atomicAdd(&preact[i6][i4 + 1][i5 + 1], weight[i6][i3][i1][i2] * input[i3][i4 + i1][i5 + i2]);//after convolution, dim is 11*11, so plus 1 to fill the center matric with calculaed num, boarder is 0.
	}

}

__global__ void fp_bias_c5(float preact[256][13][13], float bias[256])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 256*13*13;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 256);
		const int i2 = ((idx /= 256	) % 13);
		const int i3 = ((idx /= 13	) % 13);

		preact[i1][i2][i3] += bias[i1];
	}

}

__global__ void fp_preact_p3(float input[256][13][13], float preact[256][6][6])
{
	pool_data pos;
	pos.x = threadIdx.x;
	pos.y = threadIdx.y;
//	N = 6 * 6 * 256;	
	//stride is 2, so times 2
	float tmp_preact = input[blockIdx.x][2 * pos.x][2 * pos.y];
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; j++){
			if (tmp_preact < input[blockIdx.x][2 * pos.x + i][2 * pos.y + j]){
				tmp_preact =  input[blockIdx.x][2 * pos.x + i][2 * pos.y + j];
			}
		}
	}
	preact[blockIdx.x][pos.x][pos.y] = tmp_preact;

}


//full connect 1 6*6*256 to 4096*1*1
__global__ void fp_preact_f1(float input[256][6][6], float preact[4096], float weight[4096][256][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4096*256*6*6;
	
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);				
		const int i3 = ((idx /= 6	) % 256);
		const int i6 = ((idx /= 256	) % 4096);
		//const int i4 = ((idx /= 256	) % 6);
		//const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i6], weight[i6][i3][i1][i2] * input[i3][i1][i2]);
	}
}
__global__ void fp_bias_f1(float preact[4096], float bias[4096])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4096;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

//full connect 2 4096 to 4096
__global__ void fp_preact_f2(float input[4096], float preact[4096], float weight[4096][4096])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4096*4096*1*1;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4096);
		const int i2 = ((idx /= 4096	) % 4096);

		atomicAdd(&preact[i1], weight[i1][i2] * input[i2]);
	}
}

__global__ void fp_bias_f2(float preact[4096], float bias[4096])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4096;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

//full connect 3 4096 to 1000
__global__ void fp_preact_f3(float input[4096], float preact[1000], float weight[1000][4096])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1000*4096*1*1;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1000);
		const int i2 = ((idx /= 4096	) % 4096);

		atomicAdd(&preact[i1], weight[i1][i2] * input[i2]);
	}
}

__global__ void fp_bias_f3(float preact[1000], float bias[1000])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1000;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

//back prop start
// output to f3
__global__ void bp_weight_f3(float d_weight[1000][4096], float d_preact[1000], float p_output[4096])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1000*4096;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1000);
		const int i2 = ((idx /= 1000	) % 4096);

		d_weight[i1][i2] = d_preact[i1] * p_output[i2];
	}
}

__global__ void bp_bias_f3(float bias[1000], float d_preact[1000])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1000;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

// output to f1
__global__ void bp_output_f1(float d_output[84], float n_weight[10][84], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*84;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 84);

		atomicAdd(&d_output[i2], n_weight[i1][i2] * nd_preact[i1]);
	}
}

__global__ void bp_preact_f1(float d_preact[84], float d_output[84], float preact[84])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 84;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 84);

		const float o = step_function(preact[i1]);

		d_preact[i1] = d_output[i1] * o * (1 - o);
	}
}



__global__ void bp_weight_f1(float d_weight[84][120], float d_preact[84], float p_output[120])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 84*120;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 84);
		const int i2 = ((idx /= 84	) % 120);

		d_weight[i1][i2] = d_preact[i1] * p_output[i2];
	}
}

__global__ void bp_bias_f1(float bias[84], float d_preact[84])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 84;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

// output to c3
__global__ void bp_output_c3(float d_output[120], float n_weight[84][120], float nd_preact[84])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 84*120;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 84);
		const int i2 = ((idx /= 84	) % 120);

		atomicAdd(&d_output[i2], n_weight[i1][i2] * nd_preact[i1]);
	}
}

__global__ void bp_preact_c3(float d_preact[120], float d_output[120], float preact[120])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 120;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 120);

		const float o = step_function(preact[i1]);

		d_preact[i1] = d_output[i1] * o * (1 - o);
	}
}

__global__ void bp_weight_c3(float d_weight[120][16][4][4], float d_preact[120], float p_output[16][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 120*16*4*4;
	const float d = 16.0f*4.0f*4.0f;
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 120);
		const int i2 = ((idx /= 120	) % 16);
		const int i3 = ((idx /= 16	) % 4);
		const int i4 = ((idx /= 4	) % 4);

		atomicAdd(&d_weight[i1][i2][i3][i4], d_preact[i1] * p_output[i2][i3][i4]/d);
	}
}

__global__ void bp_bias_c3(float bias[120], float d_preact[120])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 120;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 120);
		atomicAdd(&bias[i1], dt * d_preact[i1]);
	}
}

// output to s2
__global__ void bp_output_s2(float d_output[16][4][4], float n_weight[120][16][4][4], float nd_preact[120])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*16*120;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 16);
		const int i4 = ((idx /= 16	) % 120);
		atomicAdd(&d_output[i3][i1][i2], n_weight[i4][i3][i1][i2] * nd_preact[i4]);
	}
}

__global__ void bp_preact_s2(float d_preact[16][4][4], float d_output[16][4][4], float preact[16][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 4);
		const int i3 = ((idx /= 4	) % 4);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s2(float d_weight[1][2][2], float d_preact[16][4][4], float p_output[16][8][8])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*2*2*16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 16);
		const int i5 = ((idx /= 16	) % 4);
		const int i6 = ((idx /= 4	) % 4);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 2 + i2][i6 * 2 + i3]);
	}
}

__global__ void bp_bias_s2(float bias[1], float d_preact[16][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3]/N);
	}
}

// output to c2
__global__ void bp_output_c2(float d_output[16][8][8], float n_weight[1][2][2], float nd_preact[16][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*2*2*16*4*4;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 16);
		const int i5 = ((idx /= 16	) % 4);
		const int i6 = ((idx /= 4	) % 4);

		atomicAdd(&d_output[i4][i5 * 2 + i2][i6 * 2 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c2(float d_preact[16][8][8], float d_output[16][8][8], float preact[16][8][8])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16 * 8 * 8;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 8);
		const int i3 = ((idx /= 8	) % 8);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c2(float d_weight[16][6][5][5], float d_preact[16][8][8], float p_output[6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*6*5*5*8*8;
	const float d = pow(8.0f, 2.0f);
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 6);
		const int i3 = ((idx /= 6	) % 5);
		const int i4 = ((idx /= 5	) % 5);
		const int i5 = ((idx /= 5	) % 8);
		const int i6 = ((idx /= 8	) % 8);

		atomicAdd(&d_weight[i1][i2][i3][i4], d_preact[i1][i5][i6] * p_output[i2][i5+i3][i6+i4]/d);
	}
}

__global__ void bp_bias_c2(float bias[16], float d_preact[16][8][8])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*8*8;
	const float d = pow(8.0f, 2.0f);
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1 ) % 16);
  		const int i2 = ((idx /= 16 ) % 8);
  		const int i3 = ((idx /= 8 ) % 8);

  		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);	
	}
}

//output s1
__global__ void bp_output_s1(float d_output[6][12][12], float n_weight[16][6][5][5], float nd_preact[16][8][8])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 16*6*5*5*8*8;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 16);
		const int i2 = ((idx /= 16	) % 6);
		const int i3 = ((idx /= 6	) % 5);
		const int i4 = ((idx /= 5	) % 5);
		const int i5 = ((idx /= 5	) % 8);
		const int i6 = ((idx /= 8	) % 8);

		atomicAdd(&d_output[i2][i3+i5][i4+i6], n_weight[i1][i2][i3][i4] * nd_preact[i1][i5][i6]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][12][12], float d_output[6][12][12], float preact[6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*12*12;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 12);
		const int i3 = ((idx /= 12	) % 12);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][2][2], float d_preact[6][12][12], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*2*2*6*12*12;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 12);
		const int i6 = ((idx /= 12	) % 12);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 2 + i2][i6 * 2 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*12*12;
	//const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 12);
		const int i3 = ((idx /= 12	) % 12);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / N);
	}
}

//output c1
__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][2][2], float nd_preact[6][12][12])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*2*2*6*12*12;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 2);
		const int i3 = ((idx /= 2	) % 2);
		const int i4 = ((idx /= 2	) % 6);
		const int i5 = ((idx /= 6	) % 12);
		const int i6 = ((idx /= 12	) % 12);

		atomicAdd(&d_output[i4][i5 * 2 + i2][i6 * 2 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}
