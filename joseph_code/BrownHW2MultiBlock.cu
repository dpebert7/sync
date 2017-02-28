/*
Joseph Brown
Homework2

Problem1
Allows arbitrarily large vectors to be processed by using multiple blocks.
*/

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

int N = 100;
	//Sets N to the default value of 100.

float *A_CPU, *B_CPU, *C_CPU;

float *A_GPU, *B_GPU, *C_GPU;

dim3 dimBlock;	//

void AllocateMemory(){
	cudaMalloc((void**)&A_GPU, N*sizeof(float));
	cudaMalloc((void**)&B_GPU, N*sizeof(float));
	cudaMalloc((void**)&C_GPU, N*sizeof(float));
	
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
} // */	//Saves the appropriate memory chunks for later use.
	//References the globally defined variables.

void Initialize(){
	for(int i = 0; i < N; i++){
		A_CPU[i] = (float)i;
		B_CPU[i] = (float)i;
	}	//Sets these arrays to the values 1..N.		
} // */

void CleanUp(float *A_CPU, float *B_CPU, float *C_CPU, 
	     float *A_GPU, float *B_GPU, float *C_GPU){
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);

	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
} // */	//Frees the memory for the three relevant global variables.

/*void VectorAddition(float *A, float *B, float *C, int n){
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	add<<<N,1>>>(A_GPU, B_GPU, C_GPU)
	cudaMemcpy(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
} // */	//Takes the component-wise sum of the first n 
	// values of two vectors, A and B, and stores them in the
	// corresponding values of a third vector, C.

__global__ void Addition(float *A, float *B, float *C, int n){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < n){
		C[bid*1024 + tid] = A[bid*1024 + tid] + B[bid*1024 +tid]; 
	}
}


int main(int argc, char *argv[]){
	//I would like this program to accept command line
	// arguments.  Simply run "./VectorAdditionCPU.cu #"
	// to run the same program with a different parameter.

	timeval start, end;
		//Declares two objects of type timeval.

	if(argc == 2){
		char *ptr;
		N = strtol(argv[1], &ptr, 10);
	}
	else if(argc > 2){
		printf("One or zero arguments expected.");
		return(1);
	}

	AllocateMemory();
	Initialize();
	gettimeofday(&start,NULL);
	int numblocks = (N-1)/1024 + 1;
			//Required number of blocks of size 1024,
			// plus one for the leftovers.  -NST

	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	Addition<<<numblocks,1024>>>(A_GPU, B_GPU, C_GPU, N);
	cudaMemcpy(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	//VectorAddition(A_CPU, B_CPU, C_CPU, N);
	gettimeofday(&end, NULL);
	float time = (end.tv_sec*1000000 + end.tv_usec*1) - 
		     (start.tv_sec*1000000 + start.tv_usec*1);
		     //tv_sec is in seconds, while tv_usec
		     // is in microseconds, so they need to
		     // be scaled appropriately.
		     //Then, it's a matter of subtracting
		     // the value of the start from the
		     // value of the end.
	printf("CPU Time in milliseconds= %.10f\n", (time/1000.0));
	printf("Blocks used= %d\n", numblocks);
	for(int i = 0; i < 5; i++){
		printf("A[%d] = %.5f   B[%d] = %.5f   C[%d] = %.5f\n",
			  i,    A_CPU[i], i,    B_CPU[i], i,    C_CPU[i]);
	} // */
	/*for(int i = 5; i < N-1; i++){
		printf("A[%d] = %.5f   B[%d] = %.5f   C[%d] = %.5f\n",
			  i,   A_CPU[i], i,    B_CPU[i], i,    C_CPU[i]);
	}// */
	printf("...\n");// */
	printf("A[%d] = %.5f   B[%d] = %.5f   C[%d] = %.5f\n",
		  N-1,  A_CPU[N-1],N-1, B_CPU[N-1],N-1, C_CPU[N-1]);
	
	CleanUp(A_CPU, B_CPU, C_CPU,
		A_GPU, B_GPU, C_GPU);

	return(0);
}

//Output for:
//ccudaclass2016@lannister:/media/storage/CUDAClasses/CUDACLASS2017/JosephBrown/Homework2$ ./temp1 46565
// where temp1 is the compiled version of this script run on lannister with arg 46565.
/*
CPU Time in milliseconds= 0.4740000000
Blocks used= 46
A[0] = 0.00000   B[0] = 0.00000   C[0] = 0.00000
A[1] = 1.00000   B[1] = 1.00000   C[1] = 2.00000
A[2] = 2.00000   B[2] = 2.00000   C[2] = 4.00000
A[3] = 3.00000   B[3] = 3.00000   C[3] = 6.00000
A[4] = 4.00000   B[4] = 4.00000   C[4] = 8.00000
...
A[46564] = 46564.00000   B[46564] = 46564.00000   C[46564] = 93128.00000
*/

