#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"
#include "pixels.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

#define fp_L1 "/root/beilei/AlexNet/result/fp_Layer1.txt"
#define fp_L2 "/root/beilei/AlexNet/result/fp_Layer2.txt"
#define fp_L3 "/root/beilei/AlexNet/result/fp_Layer3.txt"
#define fp_L4 "/root/beilei/AlexNet/result/fp_Layer4.txt"
#define fp_L5 "/root/beilei/AlexNet/result/fp_Layer5.txt"



static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN

static Layer L_input = Layer(0, 0, 227*227*3);
static Layer L_c1 = Layer(11*11*3, 2*48, 2*55*55*48);
static Layer L_p1 = Layer(3*3, 2*1, 2*31*31*48);
static Layer L_c2 = Layer(5*5*48, 2*128, 2*128*27*27);
static Layer L_p2 = Layer(3*3, 2*1, 2*15*15*128);
static Layer L_c3 = Layer(3*3*256, 384, 2*13*13*192);
static Layer L_c4 = Layer(3*3*192, 2*192, 2*13*13*192);
static Layer L_c5 = Layer(3*3*192, 2*128, 2*13*13*128);
static Layer L_p3 = Layer(3*3, 2*1, 2*6*6*128);
static Layer L_f1 = Layer(6*6*256, 2*2048, 4096*1);
static Layer L_f2 = Layer(1*4096, 2*2048, 4096*1);
static Layer L_f3 = Layer(1*4096, 1000, 1000);

static void learn(double data[227][227][3]);
static unsigned int classify(double data[227][227][3]);
static void test();
static double forward_pass(double data[227][227][3]);
static double back_pass();

/*static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}*/
int main(int argc, const  char **argv)
{
//	printf("TEST IN MAIN\n");
//	fflush(stdout);
	srand(time(NULL));
//	printf("TEST IN MAIN\n");
//	fflush(stdout);
	/*CUresult err = cuInit(0);
	if (err != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
		return 1;
	}
	*/
	//for test
	double test_data[227][227][3] = {0.0};
	for (int i = 0; i < 227; i++){
		for (int j = 0; j < 227; j++){
			for (int k = 0; k < 3; k++){
				test_data[j][k][i] = double(PIXELS[j][i][k]); 
				//printf("JIANG %-10f", test_data[j][k][i]);
			}
			//printf("\n");
		}
		//printf("\n");
	}
	//fflush(stdout);
	//for test

	//loaddata();
//printf("TEST IN LEARNING");
//fflush(stdout);
	learn(test_data);
	//test();

	return 0;
}
// Forward propagation of a single row in dataset
static double forward_pass(double data[227][227][3])
{
	float input[227][227][3];

	for (int i = 0; i < 227; ++i) {
		for (int j = 0; j < 227; ++j) {
			for (int k = 0; k < 3; k++){
				input[i][j][k] = data[i][j][k];
			}
		}
	}
	//fprintf(stdout ,"Init Forward\n");
	L_input.clear();
	L_c1.clear();
	L_p1.clear();
	L_c2.clear();
	L_p2.clear();
	L_c3.clear();
	L_c4.clear();
	L_c5.clear();
	L_p3.clear();
	L_f1.clear();
	L_f2.clear();
	L_f3.clear();
	//fprintf(stdout ,"Init Done\n");
	clock_t start, end;
	start = clock();

	L_input.setOutput((float *)input);
	fprintf(stdout ,"Conv1 Forwarding\n");
	fp_preact_c1<<<112, 1280>>>((float (*)[227][3])L_input.output, (float (*)[55][55])L_c1.preact, (float (*)[11][11][3])L_c1.weight);

	fp_bias_c1<<<112, 1280>>>((float (*)[55][55])L_c1.preact, L_c1.bias);
	apply_step_function<<<112, 1280>>>(L_c1.preact, L_c1.act_result, L_c1.O);
	normalization_function<<<112, 640>>>(L_c1.act_result, L_c1.output, L_c1.O, L_c1.N);

	fprintf(stdout ,"Pool1 Forwarding\n");
	dim3 ft_map(27, 27);
	fp_preact_p1<<<96, ft_map>>>((float (*)[55][55])L_c1.output, (float (*)[31][31])L_p1.output);

L_p1.Output_Layer(L_p1.output);
FILE *C1 = fopen(fp_L1, "w");
for(int i = 0; i < 96; i++){
	for(int j = 0; j < 31; j++){
		for (int k = 0; k < 31; k++){
			fprintf(C1, "%f", *(L_p1.L_output + (i*31*31 + j*31 + k)*sizeof(float)));
			//fprintf(C1, "%f", L_p1.output[i][j][k]);
		}
		fprintf(C1, "\n");
	}
	fprintf(C1, "\n");
}
fclose(C1);
C1 = NULL;
	
	fprintf(stdout ,"Conv2 Forwarding\n");
	fp_preact_c2<<<112, 1280>>>((float (*)[31][31])L_p1.output, (float (*)[27][27])L_c2.preact, (float (*)[96][5][5])L_c2.weight);
	fp_bias_c2<<<112, 1280>>>((float (*)[27][27])L_c2.preact, L_c2.bias);
	apply_step_function<<<112, 1280>>>(L_c2.preact, L_c2.act_result, L_c2.O);
	normalization_function<<<112, 1280>>>(L_c2.act_result, L_c2.output, L_c2.O, L_c1.N);


	fprintf(stdout ,"Pool2 Forwarding\n");
	dim3 ft_map1(13, 13);
	fp_preact_p2<<<256, ft_map1>>>((float (*)[27][27])L_c2.output, (float (*)[15][15])L_p2.output);

L_p2.Output_Layer(L_p2.output);
FILE *C2 = fopen(fp_L2, "w");
for(int i = 0; i < 256; i++){
	for(int j = 0; j < 15; j++){
		for (int k = 0; k < 15; k++){
			fprintf(C2, "%f", *(L_p2.L_output + (i*15*15 + j*15 + k)*sizeof(float)));
		}
		fprintf(C2, "\n");
	}
	fprintf(C2, "\n");
}
fclose(C2);
C2 = NULL;

	
	fprintf(stdout ,"Conv3 Forwarding\n");
	fp_preact_c3<<<128, 128>>>((float (*)[15][15])L_p2.output, (float (*) [13][13])L_c3.preact, (float (*)[256][3][3])L_c3.weight);
	fp_bias_c3<<<128, 128>>>((float (*)[13][13])L_c3.preact, L_c3.bias);
	apply_step_function<<<128, 128>>>(L_c3.preact, L_c3.output, L_c3.O);

L_c3.Output_Layer(L_c3.output);
FILE *C3 = fopen(fp_L3, "w");
for(int i = 0; i < 384; i++){
	for(int j = 0; j < 13; j++){
		for (int k = 0; k < 13; k++){
			fprintf(C3, "%f", *(L_c3.L_output + (i*13*13 + j*13 + k)*sizeof(float)));
		}
		fprintf(C3, "\n");
	}
	fprintf(C3, "\n");
}
fclose(C3);
C3 = NULL;


	fprintf(stdout ,"Conv4 Forwarding\n");
	fp_preact_c4<<<128, 128>>>((float (*)[13][13])L_c3.output, (float (*) [13][13])L_c4.preact, (float (*)[384][3][3])L_c4.weight);
	fp_bias_c4<<<128, 128>>>((float (*)[13][13])L_c4.preact, L_c4.bias);
	apply_step_function<<<128, 128>>>(L_c4.preact, L_c4.output, L_c4.O);

L_c4.Output_Layer(L_c4.output);
FILE *C4 = fopen(fp_L4, "w");
for(int i = 0; i < 384; i++){
	for(int j = 0; j < 13; j++){
		for (int k = 0; k < 13; k++){
			fprintf(C4, "%f", *(L_c4.L_output + (i*13*13 + j*13 + k)*sizeof(float)));
		}
		fprintf(C4, "\n");
	}
	fprintf(C4, "\n");
}
fclose(C4);
C4 = NULL;


	fprintf(stdout ,"Conv5 Forwarding\n");
	fp_preact_c5<<<128, 128>>>((float (*)[13][13])L_c4.output, (float (*)[13][13])L_c5.preact, (float (*)[384][3][3])L_c5.weight);
	fp_bias_c5<<<128, 128>>>((float (*)[13][13])L_c5.preact, L_c5.bias);
	apply_step_function<<<128, 128>>>(L_c5.preact, L_c5.output, L_c5.O);


	fprintf(stdout ,"Pool3 Forwarding\n");
	dim3 ft_map2(6, 6);
	fp_preact_p3<<<256, ft_map2>>>((float (*)[13][13])L_c5.output, (float (*)[6][6])L_p3.output);

L_p3.Output_Layer(L_p3.output);
FILE *C5 = fopen(fp_L5, "w");
for(int i = 0; i < 256; i++){
	for(int j = 0; j < 6; j++){
		for (int k = 0; k < 6; k++){
			fprintf(C5, "%f", *(L_p3.L_output + (i*6*6 + j*6 + k)*sizeof(float)));
		}
		fprintf(C5, "\n");
	}
	fprintf(C5, "\n");
}
fclose(C5);
C5 = NULL;


#if 1	
	fprintf(stdout ,"Full1 Forwarding\n");
	fp_preact_f1<<<128, 128>>>((float (*)[6][6])L_p3.output, L_f1.preact, (float (*)[256][6][6])L_f1.weight);
	fp_bias_f1<<<128, 128>>>(L_f1.preact, L_f1.bias);
	apply_step_function<<<128, 128>>>(L_f1.preact, L_f1.output, L_f1.O);
	
	fprintf(stdout ,"Full2 Forwarding\n");
	fp_preact_f2<<<128, 128>>>(L_f1.output, L_f2.preact, (float (*)[4096])L_f2.weight);
	fp_bias_f2<<<128, 128>>>(L_f2.preact, L_f2.bias);
	apply_step_function<<<128, 128>>>(L_f2.preact, L_f2.output, L_f2.O);

	fprintf(stdout ,"Full3 Forwarding\n");
	fp_preact_f2<<<128, 128>>>(L_f2.output, L_f3.preact, (float (*)[4096])L_f3.weight);
	fp_bias_f2<<<128, 128>>>(L_f3.preact, L_f3.bias);
	apply_step_function<<<128, 128>>>(L_f3.preact, L_f3.output, L_f3.O);


	fprintf(stdout ,"forward pass done!!\n");
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
#endif
}
#if 0
// Back propagation to update weights
static double back_pass()
{
	clock_t start, end;

	start = clock();
	fprintf(stdout ,"Full2 Backwarding\n");
	bp_weight_f2<<<128, 128>>>((float (*)[84])L_f2.d_weight, L_f2.d_preact, L_f1.output);
	bp_bias_f2<<<128, 128>>>(L_f2.bias, L_f2.d_preact);
	
	fprintf(stdout ,"Full1 Backwarding\n");
	bp_output_f1<<<128, 128>>>(L_f1.d_output, (float (*)[84])L_f2.weight, L_f2.d_preact);
	bp_preact_f1<<<128, 128>>>(L_f1.d_preact, L_f1.d_output, L_f1.preact);
	bp_weight_f1<<<128, 128>>>((float (*)[120])L_f1.d_weight, L_f1.d_preact, L_c3.output);
	bp_bias_f1<<<128, 128>>>(L_f1.bias, L_f1.d_preact);

	fprintf(stdout ,"Conv3 Backwarding\n");
	bp_output_c3<<<128, 128>>>(L_c3.d_output, (float (*)[120])L_f1.weight, L_f1.d_preact);
	bp_preact_c3<<<128, 128>>>(L_c3.d_preact, L_c3.d_output, L_c3.preact);
	bp_weight_c3<<<128, 128>>>((float (*)[16][4][4])L_c3.d_weight, L_c3.d_preact, (float (*)[4][4])L_p2.output);
	bp_bias_c3<<<128, 128>>>(L_c3.bias, L_c3.d_preact);

	fprintf(stdout ,"Pool2 Backwarding\n");
	bp_output_s2<<<128, 128>>>((float (*)[4][4])L_p2.d_output, (float (*)[16][4][4])L_c3.weight, L_c3.d_preact);
	bp_preact_s2<<<128, 128>>>((float (*)[4][4])L_p2.d_preact, (float (*)[4][4])L_p2.d_output, (float (*)[4][4])L_p2.preact);
	bp_weight_s2<<<128, 128>>>((float (*)[2][2])L_p2.d_weight, (float (*)[4][4])L_p2.d_preact, (float (*)[8][8])L_c2.output);
	bp_bias_s2<<<128, 128>>>(L_p2.bias, (float (*)[4][4])L_p2.d_preact);
	
	fprintf(stdout ,"Conv2 Backwarding\n");
	bp_output_c2<<<128, 128>>>((float (*)[8][8])L_c2.d_output, (float (*)[2][2])L_p2.weight, (float (*)[4][4])L_p2.d_preact);
	bp_preact_c2<<<128, 128>>>((float (*)[8][8])L_c2.d_preact, (float (*)[8][8])L_c2.d_output, (float (*)[8][8])L_c2.preact);
	bp_weight_c2<<<128, 128>>>((float (*)[6][5][5])L_c2.d_weight, (float (*)[8][8])L_c2.d_preact, (float (*)[12][12])L_p1.output);
	bp_bias_c2<<<128, 128>>>(L_c2.bias, (float (*)[8][8])L_c2.d_preact);

	fprintf(stdout ,"Pool1 Backwarding\n");
	bp_output_s1<<<128, 128>>>((float (*)[12][12])L_p1.d_output, (float (*)[6][5][5])L_c2.weight, (float (*)[8][8])L_c2.d_preact);
	bp_preact_s1<<<128, 128>>>((float (*)[12][12])L_p1.d_preact, (float (*)[12][12])L_p1.d_output, (float (*)[12][12])L_p1.preact);
	bp_weight_s1<<<128, 128>>>((float (*)[2][2])L_p1.d_weight, (float (*)[12][12])L_p1.d_preact, (float (*)[24][24])L_c1.output);
	bp_bias_s1<<<128, 128>>>(L_p1.bias, (float (*)[12][12])L_p1.d_preact);

	fprintf(stdout ,"Conv1 Backwarding\n");
	bp_output_c1<<<128, 128>>>((float (*)[24][24])L_c1.d_output, (float (*)[2][2])L_p1.weight, (float (*)[12][12])L_p1.d_preact);
	bp_preact_c1<<<128, 128>>>((float (*)[24][24])L_c1.d_preact, (float (*)[24][24])L_c1.d_output, (float (*)[24][24])L_c1.preact);
	bp_weight_c1<<<128, 128>>>((float (*)[5][5])L_c1.d_weight, (float (*)[24][24])L_c1.d_preact, (float (*)[28])L_input.output);
	bp_bias_c1<<<128, 128>>>(L_c1.bias, (float (*)[24][24])L_c1.d_preact);

	fprintf(stdout ,"Update Weight\n");
	apply_grad<<<128, 128>>>(L_f2.weight, L_f2.d_weight, L_f2.M * L_f2.N);
	apply_grad<<<128, 128>>>(L_f1.weight, L_f1.d_weight, L_f1.M * L_f1.N);
	apply_grad<<<128, 128>>>(L_c3.weight, L_c3.d_weight, L_c3.M * L_c3.N);
	apply_grad<<<128, 128>>>(L_p2.weight, L_p2.d_weight, L_p2.M * L_p2.N);
	apply_grad<<<128, 128>>>(L_c2.weight, L_c2.d_weight, L_c2.M * L_c2.N);
	apply_grad<<<128, 128>>>(L_p1.weight, L_p1.d_weight, L_p1.M * L_p1.N);
	apply_grad<<<128, 128>>>(L_c1.weight, L_c1.d_weight, L_c1.M * L_c1.N);


	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Unfold the input layer
static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int b = 0;
			for (int x = i; x < i + 2; ++x)
				for (int y = j; y < j+2; ++y)
					unfolded[a][b++] = input[x][y];
			a++;
		}
}
#endif
static void learn(double data[227][227][3])
{
	//static cublasHandle_t blas;
	//cublasCreate(&blas);
printf("TEST IN LEARN");
fflush(stdout);
	float err;
	int iter = 1;
	
	double time_taken = 0.0;

	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < 1; ++i) {
			float tmp_err;
			//fprintf(stdout ,"Before Forward_Pass\n %d\n", i);
			//time_taken += forward_pass(train_set[i].data);

			time_taken += forward_pass(data);

			//fprintf(stdout ,"After Forward_Pass\n");
			L_f2.bp_clear();
			L_f1.bp_clear();
			L_p2.bp_clear();
			L_p1.bp_clear();
			L_c3.bp_clear();
			L_c2.bp_clear();
			L_c1.bp_clear();

			// Euclid distance of train_set[i]
#if 0
			makeError<<<10, 1>>>(l_f2.d_preact, l_f2.output, train_set[i].label, 10);
			//cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
			tmp_err = 0.3;
			err += tmp_err;
			//fprintf(stdout ,"Before Backward_Pass\n");
			time_taken += back_pass();
			//fprintf(stdout ,"After Backward_Pass\n");
#endif
		}
#if 0
		err /= train_cnt;
		fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);
		/*
		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}
		*/
#endif

	}
#if 0	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
#endif
}

#if 0
// Returns label of given data (0-9)
static unsigned int classify(double data[227][227][3])
{
	float res[10];

	forward_pass(data);

	unsigned int max = 0;

	cudaMemcpy(res, L_f2.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

// Perform forward propagation of test data
static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}
#endif
