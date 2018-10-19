#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

typedef struct pool_data{
	unsigned int x;
	unsigned int y;
}pool_data;//for pooling layer thread x and y
class Layer {
	public:
	long int M, N, O;

	float *output;
	float *preact;
	float *act_result;//for normalization

	float *bias;
	float *weight;

	float *L_output;

	float *d_output;
	float *d_preact;
	float *d_act_result;//for normalization
	float *d_weight;

	Layer(long int M, long int N, long int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
	void Output_Layer(float *data);
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__device__ float normalization(float *input, float u, int idx, const int O, const int N);
__global__ void normalization_function(float *input, float *output, const int O, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[227][227][3], float preact[96][55][55], float weight[96][11][11][3]);
__global__ void fp_bias_c1(float preact[96][55][55], float bias[96]);
__global__ void fp_preact_p1(float input[96][55][55], float preact[96][31][31]);
//__global__ void fp_bias_p1(float preact[6][12][12], float bias[1]);
__global__ void fp_preact_c2(float input[96][31][31], float preact[128][27][27], float weight[128][96][5][5]);
__global__ void fp_bias_c2(float preact[128][27][27], float bias[128]);
__global__ void fp_preact_p2(float input[256][27][27], float preact[256][15][15]);
//__global__ void fp_preact_p2(float input[256]27][27], float preact[256][15][15]);

//__global__ void fp_preact_s2(float input[16][8][8], float preact[16][4][4], float weight[1][2][2]);
//__global__ void fp_bias_p2(float preact[16][4][4], float bias[1]);
//__global__ void fp_bias_s2(float preact[16][4][4], float bias[1]);
__global__ void fp_preact_c3(float input[256][15][15], float preact[384][13][13], float weight[384][256][3][3]);
__global__ void fp_bias_c3(float preact[384][13][13], float bias[384]);

__global__ void fp_preact_c4(float input[384][13][13], float preact[384][13][13], float weight[384][384][3][3]);
__global__ void fp_bias_c4(float preact[384][13][13], float bias[384]);

__global__ void fp_preact_c5(float input[384][13][13], float preact[256][13][13], float weight[256][384][3][3]);
__global__ void fp_bias_c5(float preact[256][13][13], float bias[256]);
__global__ void fp_preact_p3(float input[256][13][13], float preact[256][6][6]);
//__global__ void fp_preact_p3(float input[256]13][13], float preact[256][6][6]);


__global__ void fp_preact_f1(float input[256][6][6], float preact[4096], float weight[4096][256][6][6]);
__global__ void fp_bias_f1(float preact[4096], float bias[4096]);
__global__ void fp_preact_f2(float input[4096], float preact[4096], float weight[4096][4096]);
__global__ void fp_bias_f2(float preact[4096], float bias[4096]);
__global__ void fp_preact_f3(float input[4096], float preact[1000], float weight[1000][4096]);
__global__ void fp_bias_f3(float preact[1000], float bias[1000]);

// Back propagation kernels
__global__ void bp_weight_f2(float d_weight[10][84], float d_preact[10], float p_output[84]);
__global__ void bp_bias_f2(float bias[10], float d_preact[10]);

__global__ void bp_output_f1(float d_output[84], float n_weight[10][84], float nd_preact[10]);
__global__ void bp_preact_f1(float d_preact[84], float d_output[84], float preact[84]);
__global__ void bp_weight_f1(float d_weight[84][120], float d_preact[84], float p_output[120]);
__global__ void bp_bias_f1(float bias[84], float d_preact[84]);

__global__ void bp_output_c3(float d_output[120], float n_weight[84][120], float nd_preact[84]);
__global__ void bp_preact_c3(float d_preact[120], float d_output[120], float preact[120]);
__global__ void bp_weight_c3(float d_weight[120][16][4][4], float d_preact[120], float p_output[16][4][4]);
__global__ void bp_bias_c3(float bias[120], float d_preact[120]);

__global__ void bp_output_s2(float d_output[16][4][4], float n_weight[120][16][4][4], float nd_preact[120]);
__global__ void bp_preact_s2(float d_preact[16][4][4], float d_output[16][4][4], float preact[16][4][4]);
__global__ void bp_weight_s2(float d_weight[1][2][2], float d_preact[16][4][4], float p_output[16][8][8]);
__global__ void bp_bias_s2(float bias[1], float d_preact[16][4][4]);

__global__ void bp_output_c2(float d_output[16][8][8], float n_weight[1][2][2], float nd_preact[16][4][4]);
__global__ void bp_preact_c2(float d_preact[16][8][8], float d_output[16][8][8], float preact[16][8][8]);
__global__ void bp_weight_c2(float d_weight[16][6][5][5], float d_preact[16][8][8], float p_output[6][12][12]);
__global__ void bp_bias_c2(float bias[16], float d_preact[16][8][8]);

__global__ void bp_output_s1(float d_output[6][12][12], float n_weight[16][6][5][5], float nd_preact[16][8][8]);
__global__ void bp_preact_s1(float d_preact[6][12][12], float d_output[6][12][12], float preact[6][12][12]);
__global__ void bp_weight_s1(float d_weight[1][2][2], float d_preact[6][12][12], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][12][12]);

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][2][2], float nd_preact[6][12][12]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);




/*
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);
*/
