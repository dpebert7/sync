//nvcc fractalAnim.cu -o temp -lglut -lGL -lm -run

#include <GL/glut.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <signal.h>

using namespace std;

float *A_CPU, *B_CPU, *C_CPU, *pixels_CPU;

float *A_GPU, *B_GPU, *C_GPU, *pixels_GPU;

dim3 dimBlock;

/*float A = -0.624;
float B = 0.4351; // */

float A = 0;
float B = 0.75; // */
float t = 0;
float tmod = 100.0;
float titer = 1.0;
float moveiter = 1.0;
int N = 100;

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

void AllocateMemory(){
	cudaMalloc((void**)&pixels_GPU, window_width*window_height*3*sizeof(float));
	cudaMalloc((void**)&B_GPU, N*sizeof(float));
	cudaMalloc((void**)&C_GPU, N*sizeof(float));
	
	
	pixels_CPU = (float *)malloc(window_width*window_height*3*sizeof(float));
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

float color (float x, float y)	//hopefully centered on (0,0)? 
{
	float mag,maxMag,t1;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + A;
		y = (2.0 * t1 * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(1.0);
	}
	else
	{
		return(0.0);
	}// */
}

__global__ void cudaColor(float *pixels_GPU, float X, float iY){
	
	float x = (((float)threadIdx.x)/(blockDim.x))*4-2;
	float y = (((float)blockIdx.x)/(gridDim.x))*4-2;
	float mag,maxMag, t1;
	int maxCount = 200;
	int count = 0;
	maxMag = 10;
	mag = 0.0;

	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + X;
		y = (2.0 * t1 * y) + iY;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 
			0.5*log((double)count)/log((double)maxCount); 
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] =
			1.0*log((double)count)/log((double)maxCount); 
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 
			0.4;
	}
	else
	{
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3] = 0.0;
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 1] = 0.0;
		pixels_GPU[(blockIdx.x*blockDim.x + threadIdx.x)*3 + 2] = 0.0;
	}// */
	

}

void update(int value){
	
	t = t + titer;
	/*A = -pow(sin(t/500.0),2);
	B = sin(2*t/500.0)/2;// */
	/*A = -pow(sin((2*t/5)/500.0),2);
	B = sin((2*t/3)/500.0)/2;// */
	/*A = -pow(sin((2*t/5)/100.0),2);
	B = sin(sqrt(2)*(t)/100.0)/2;// */
	A = -pow(sin((2*t/5)/100.0),2);
	B = sin(((sqrt(5)+1)/2)*(t)/100.0)/2;// */

	glutPostRedisplay();
	glutTimerFunc(16,update, 0);
}

/*static void signalHandler(int signum) {
	int command;
	bool exitMenu = 0;
	//cout << "I handled it :)" << endl;
	
	while (exitMenu == 0) {
	    	cout << "Enter 0 to exit the program." << endl;
		cout << "Enter 1 to continue." << endl;
		cin >> command;
		
		if(command == 0) {
			exitMenu = 1;
		} else if (command == 1){
			exitMenu = 1;
			cout << "resuming..." << endl;
		} else {
			cout << "Invalid Command!" << endl;
		}
		cout << endl;
	}
}// */

void display(void) 
{ 

	
	cudaColor<<<1024, 1024>>>(pixels_GPU, A, B);
	cudaMemcpy(pixels_CPU, pixels_GPU, window_width*window_height*3*sizeof(float), 
		   cudaMemcpyDeviceToHost);
	
	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels_CPU); 
	glFlush(); 
	
}

__global__ void Addition(float *A, float *B, float *C, int n){
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid < n){
		C[bid*1024 + tid] = A[bid*1024 + tid] + B[bid*1024 +tid]; 
	}
}

void CleanUp(float *A_CPU, float *B_CPU, float *C_CPU, 
	     float *A_GPU, float *B_GPU, float *C_GPU){
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);

	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
} // */	//Frees the memory for the three relevant global variables.


int main(int argc, char *argv[])
{ 
	if(argc == 2){
		char *ptr;
		N = strtol(argv[1], &ptr, 10);
	}
	else if(argc > 2){
		printf("One or zero arguments expected.");
		return(1);
	}
	
	AllocateMemory();
	//Initialize();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractals man, fractals.");
   	glutDisplayFunc(display);
	glutSpecialFunc(processSpecialKeys);
	glutTimerFunc(16, update, 0);
	glutMainLoop();
}

