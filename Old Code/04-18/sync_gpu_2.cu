/*
David and Mikaela
SYNC
Move to GPU
*/

//Compiling: nvcc sync_gpu_2.cu -o sync_1 -lglut -lm -lGLU -lGL && ./sync_1

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPSILON			0.000001
#define EYEZ			50.0 // Effectively sets x- and y-coordinates from -EYEZ to +EYEZ
#define BOUNDARY		EYEZ-1.0 // Walls
#define BDIST			10.0 // Distance from boundary at which curving should start.
#define PI				3.1415926535
#define DRAW 			1	// For draw_picture
#define XWindowSize 	1000 	// 700 initially 
#define YWindowSize 	1000 	// 700 initially

#define DT        		.1 	// Time step
#define STOP_TIME		100.0 	// How long to go
//#define STOP_TIME		0.0003

#define SIGHT 			10.0 // How far the fish can 'see'
#define WA			2.0 // Attraction Weight Ratio
#define WD 			10.0 // Directional Weight Ratio
#define CA 			2.0 // Attraction Coefficient
#define CR			1.0 // Repulsion Coefficient
#define CPR			100000000.0 // Repulsion Coefficient (predator)
#define CTA			100.0 // Attraction Coefficient (Target)

#define N 			16 // Number of fish
#define M			1 // Number of predators
#define P			1 //Number of targets

#define FISHRAD 		0.4 // Radius of fish spheres
#define FISHR			1.0 // Red of fish color
#define FISHG			1.0 // Green of fish color
#define FISHB			0.5 // Blue of fish color

#define PREDRAD			0.9 // Radius of predator
#define PREDR			1.0 // Red of predator color
#define PREDG			0.5 // Green of predator color
#define PREDB			1.0 // Blue of predator color

#define TARGRAD			0.5 // Radius of target (food)
#define TARGR			0.0 // Red of target color
#define TARGG			0.0 // Green of target color
#define TARGB			1.0 // Blue of target color 
 
#define THREADS			N // Number of threads per block


//Globals
float4 p[N];
float3 v[N], f[N];
float4 p_pred[M];
float3 v_pred[M], f_pred[M];
float4 p_targ[P];
float3 v_targ[P], f_targ[P];
float4 *p_GPU;
float3 *v_GPU, *f_GPU;
float4 *pred_GPU, *targ_GPU;
dim3 block, grid;

double SPEED = 5.0; //500000.0
double TIMERUNNING = 0.0;
int PAUSE = 0;
int FOODPOINTER = 0;
double STARTRAD = 10.0;




void initializeBodies()
{
	TIMERUNNING = 0.0;
	int i;
	
	// Initialize Fish
	for(i=0; i<N; i++)
	{
		// Starting at random positions
		p[i].x = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD-45.0;
		p[i].y = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		p[i].z = 0.0; //((double)rand()/(double)RAND_MAX)*STARTRAD;
		
		// Setting mass to 1
		p[i].w = 1.0;
		
		// Setting velocity to 0
		v[i].x = 0.01;
		v[i].y = 0.0;
		v[i].z = 0.0;
		
		// Setting force to 0
		f[i].x = 0.0;
		f[i].y = 0.0;
		f[i].z = 0.0;
	}
	
	// Initialize Predator
	for(i=0; i<M; i++)
	{
		p_pred[i].x = 40.0*sin(i*2*PI/M);
		p_pred[i].y = 40.0*cos(i*2*PI/M);
		p_pred[i].z = 0.0; //75.0*cos(i*2*PI);
		p_pred[i].w = 1.0;
		v_pred[i].x = 1.0*cos(p_pred[i].x);
		v_pred[i].y = 1.0*sin(p_pred[i].y);
		v_pred[i].z = 0.0;
	}
	
	// Initialize Targets
	for(i=0; i<P; i++)
	{
		p_targ[i].x = 40.0;
		p_targ[i].y = 0.0;
		p_targ[i].z = 0.0;
		p_targ[i].w = 1.0;
	}
	
	block.x = THREADS;
	block.y = 1;
	block.z = 1;
	
	grid.x = (N-1)/block.x + 1;
	grid.y = 1;
	grid.z = 1;
	
	cudaMalloc( (void**)&p_GPU, N*sizeof(float4) );
	cudaMalloc( (void**)&v_GPU, N*sizeof(float3) );
	cudaMalloc( (void**)&f_GPU, N*sizeof(float3) );
	cudaMalloc( (void**)&pred_GPU, M*sizeof(float4) );
	cudaMalloc( (void**)&targ_GPU, P*sizeof(float4) );
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	int i;
	
	// Drawing the fish
	for(i=0; i<N; i++){
		glColor3d(FISHR, FISHG, FISHB); 	//Object color
		glPushMatrix();
		glTranslatef(p[i].x, p[i].y, p[i].z);
		glutSolidSphere(FISHRAD, 10, 10);    //First argument affects size.
		glPopMatrix();
	}
	
	// Drawing the predators
	for(i=0; i<M; i++)
	{
		glColor3d(PREDR, PREDG, PREDB);
		glPushMatrix();
		glTranslatef(p_pred[i].x, p_pred[i].y, p_pred[i].z);
		glutSolidSphere(PREDRAD, 10, 10);
		glPopMatrix();
	}
	
	// Drawing the targets
	for(i=0; i<P; i++)
	{
		glColor3d(TARGR, TARGG, TARGB);
		glPushMatrix();
		glTranslatef(p_targ[i].x, p_targ[i].y, p_targ[i].z);
		glutSolidSphere(TARGRAD, 10, 10);
		glPopMatrix();
	}
	glutSwapBuffers();
}

__device__ float3 getFishForces(float4 p0, float4 p1, float3 v1)
{
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
	float r = sqrt(r2) + EPSILON;
	float r4 = r2*r2 + EPSILON;
	
	// Check that we're within the sight radius
	if(r < SIGHT)
	{
		f.x = WA*(CA*(dx/r2) - CR*(dx/r4)) + WD*(v1.x/r);
		f.y = WA*(CA*(dy/r2) - CR*(dy/r4)) + WD*(v1.y/r);
		f.z = WA*(CA*(dz/r2) - CR*(dz/r4)) + WD*(v1.z/r);
	}
	else // If not, this particular fish does not affect the forces on p0
	{
		f.x = 0.0;
		f.y = 0.0;
		f.z = 0.0;
	}
	
	return(f);	
}

__device__ float3 getPredForces(float4 p0, float4 ppred)
{
	float3 f;
	float dx = ppred.x - p0.x;
	float dy = ppred.y - p0.y;
	float dz = ppred.z - p0.z;
	float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
	float r = sqrt(r2) + EPSILON;
	float r4 = r2*r2 + EPSILON;
	
	// Check that we're within ten times the sight radius (looking out farther for predators than for other fish)
	if(r < SIGHT*10.0)
	{
		f.x = -(CPR*dx/r4);
		f.y = -(CPR*dy/r4);
		f.z = -(CPR*dz/r4);
	}
	else
	{
		f.x = 0.0;
		f.y = 0.0;
		f.z = 0.0;
	}
	
	return(f);
}

__device__ float3 getTargForces(float4 p0, float4 ptarg)
{
	float3 f;
	float dx = ptarg.x - p0.x;
	float dy = ptarg.y - p0.y;
	float dz = ptarg.z - p0.z;
	
	float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
	float r = sqrt(r2) + EPSILON;
	
	if(r < SIGHT)
	{
		f.x = CTA*dx/r2;
		f.y = CTA*dy/r2;
		f.z = CTA*dz/r2;
	}
	else
	{
		f.x = 0.0;
		f.y = 0.0;
		f.z = 0.0;
	}
	
	return(f);
}

__global__ void getForces(float4 *pos, float3 *vel, float3 *force, float4 *predp, int m, int p, float4 *predtarg, float speed)
{
	int i, j, ii;
	float3 forceMag, forceSum; //, forcePred, forceTarg;
	float4 posMe;
	float walldist;
	
	__shared__ float4 shPos[THREADS];
	__shared__ float3 shVel[THREADS];
	
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	
	forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;
	
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	posMe.w = pos[id].w;
	
	for(j=0; j<gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockIdx.x*j];
		shVel[threadIdx.x] = vel[threadIdx.x + blockIdx.x*j];
		__syncthreads();
		
		#pragma unroll 32
		for(i=0; i<blockDim.x; i++)
		{
			ii = i + blockDim.x*j;
			if(ii != id)
			{
				forceMag = getFishForces(posMe, shPos[i], shVel[i]);
				forceSum.x += forceMag.x;
				forceSum.y += forceMag.y;
				forceSum.z += forceMag.z;
			}
			__syncthreads();
		}
		__syncthreads();
	}
	
	/*for(i=0; i<m; i++)
	{
		forcePred = getPredForces(posMe, predp[i]);
		forceSum.x += forcePred.x;
		forceSum.y += forcePred.y;
		forceSum.z += forcePred.z;
	}
	__syncthreads();
	*/
	
	/*for(i=0; i<p; i++)
	{
		forceTarg = getTargForces(posMe, predtarg[i]);
		forceSum.x += forceTarg.x;
		forceSum.y += forceTarg.y;
		forceSum.z += forceTarg.z;
	}
	__syncthreads();
	*/
	

	
	// Walls
	if(posMe.x>40.0)// && particle[i].v[0]>0.0){
	{
		walldist = 50.0-posMe.x;
		//printf("Distance: %.2f\n", walldist);
		//printf("Force before: %.5f\n", forceSum.x);
		forceSum.x -= 100000.0*((10.0-walldist)*speed/walldist);
		printf("Force after: %.5f\n", forceSum.x);	
		// For directional change
		//if(particle[i].p[1] < (BOUNDARY-BDIST) && particle[i].p[1] > (BDIST-BOUNDARY))
		//{
		//	particle[i].f[1] += SPEED*(sqrt(particle[i].v[0]*particle[i].v[0])/(particle[i].v[1]+EPSILON));
		//}
	}
	__syncthreads();
	
	
	
	
	force[id].x = forceSum.x;
	force[id].y = forceSum.y;
	force[id].z = forceSum.z;
	__syncthreads();
	
	//printf("ID: %i Force in x direction: %.2f\n", id, force[id].x);
	//printf("ID: %i Force in y direction: %.2f\n", id, force[id].y);
	
	// Normalize forces before updating velocity and 	
	float forceNormalizer = sqrt(force[id].x*force[id].x + force[id].y*force[id].y + force[id].z*force[id].z);
	
	force[id].x /= forceNormalizer;
	force[id].y /= forceNormalizer;
	force[id].z /= forceNormalizer;


	printf("ID: %i Normalized force in x direction:    %.2f\n", id, force[id].x);
	printf("ID: %i Normalized force in y direction:    %.2f\n", id, force[id].y);
	printf("ID: %i Normalized velocity in x direction: %.2f\n", id,   vel[id].x);
	printf("ID: %i Normalized velocity in y direction: %.2f\n", id,   vel[id].y);
	
}

__global__ void swimFishSwim(float4 *pos, float3 *vel, float3 *force, float SPEED)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	
	float velocityNormalizer = sqrt(vel[id].x*vel[id].x + vel[id].y*vel[id].y + vel[id].z*vel[id].z);
	
	vel[id].x /= velocityNormalizer;
	vel[id].y /= velocityNormalizer;
	vel[id].z /= velocityNormalizer;
	
	vel[id].x = (9*vel[id].x + force[id].x*DT)/10;
	vel[id].y = (9*vel[id].y + force[id].y*DT)/10;
	vel[id].z = (9*vel[id].z + force[id].z*DT)/10;

	printf("ID: %i Velocity in x direction: %.2f\n", id, vel[id].x);
	printf("ID: %i Velocity in y direction: %.2f\n", id, vel[id].y);
	
	pos[id].x += vel[id].x*DT;
	pos[id].y += vel[id].y*DT;
	pos[id].z += vel[id].z*DT;
}

void goPredGo(float4 *predp, float3 *predv, float3 *predf, int m, double dt)
{
	int i;
	for(i=0; i<m; i++)
	{
		predf[i].x = 0.0 - predp[i].x;
		predf[i].y = 0.0 - predp[i].y;
		predf[i].z = 0.0 - predp[i].z;
		
		float forceNormalizer = sqrt(predf[i].x*predf[i].x + predf[i].y*predf[i].y + predf[i].z*predf[i].z);
		predf[i].x /= forceNormalizer;
		predf[i].y /= forceNormalizer;
		predf[i].z /= forceNormalizer;	
		
		predv[i].x += predf[i].x;
		predv[i].y += predf[i].y;
		predv[i].z += predf[i].z;
		
		float velocityNormalizer = sqrt(predv[i].x*predv[i].x + predv[i].y*predv[i].y + predv[i].z*predv[i].z);
		predv[i].x /= velocityNormalizer;
		predv[i].y /= velocityNormalizer;
		predv[i].z /= velocityNormalizer;	
	
		predp[i].x += predv[i].x*dt;
		predp[i].y += predv[i].y*dt;
		predp[i].z += predv[i].z*dt;
	}
}

void goFoodGo(float4 *targp, float3 *targv, float3 *targf, int p)
{
	int i;
	for(i=0; i<p; i++)
	{
		targp[i].x = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		targp[i].y = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		targp[i].z = (((double)rand()/(double)RAND_MAX)-0.5)*0.0;
	}
}

void nBody()
{
	double dt;
	int tdraw = 1;
	float time = 0.0;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	dt = DT;
	
	//while(time < STOP_TIME)
	//{
		goPredGo(p_pred, v_pred, f_pred, M, dt);
		//goFoodGo(p_targ, v_targ, f_targ, P);
		
		cudaMemcpy(p_GPU, p, N*sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(v_GPU, v, N*sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(pred_GPU, p_pred, M*sizeof(float4), cudaMemcpyHostToDevice);
		cudaMemcpy(targ_GPU, p_targ, P*sizeof(float4), cudaMemcpyHostToDevice);
	
		getForces<<<grid, block>>>(p_GPU, v_GPU, f_GPU, pred_GPU, M, P, targ_GPU, SPEED);
		swimFishSwim<<<grid, block>>>(p_GPU, v_GPU, f_GPU, SPEED);
		
		if(tdraw == DRAW)
		{
			cudaMemcpy(p, p_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
			//cudaMemcpy(p, p_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
			//cudaMemcpy(p, p_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
			drawPicture();
			tdraw = 0;
		}
		tdraw++;
		
		time += dt;
	//}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	//printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime);
	
	cudaMemcpy(p, p_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
	
}

void arrowFunc(int key, int x, int y) 
{
	switch (key) 
	{	//100 is left arrow, 102 is right arrow.    
		case 101 : // Up arrow
			SPEED *= 1.1;
			printf("Speed: %.4f\n", SPEED);		
			; break;
		case 103 : // Down arrow
			SPEED /= 1.1;
			printf("Speed: %.4f\n", SPEED);		
			; break;		
	}
}


void keyboardFunc( unsigned char key, int x, int y )
{
	int i,j;
	switch(key) 
	{
		case 'q': 
			exit(1);

		case ' ': 
			j=0;
			for(i=0;i<N;i++)
			{	
				if(abs((int)p[i].x) + abs((int)p[i].y) + abs((int)p[i].z) < EYEZ*3.0)
				{
					j++;
				}
			}	
			printf("There are %i particles remaining\n", j);	

			PAUSE++;
			if(PAUSE == 2)
			{
				PAUSE = 0;
			}
	}
}


void mouseFunc( int button, int state, int x, int y )
{
	double coord[3];
	if( button == GLUT_LEFT_BUTTON ) 
	{
		if( state == GLUT_DOWN && PAUSE == 0) // when left mouse button goes down.
		{
			//printf("FOODPOINTER is %i \n", FOODPOINTER);
			coord[0] = (x*EYEZ*2.0/XWindowSize)-EYEZ;
			coord[1] = -(y*EYEZ*2.0/YWindowSize)+EYEZ;
			coord[2] = 0.0;
			printf("The food is at (%.4f, %.4f, %.4f)\n",
				coord[0], coord[1], coord[2]);
			p_targ[0].x = coord[0];
			p_targ[0].y = coord[1];
			p_targ[0].z = coord[2];
			
			//Change pointer to next food particle
			FOODPOINTER++;
			if(FOODPOINTER == P)
			{
				FOODPOINTER = 0;
			}		
		}
	}
}

	
void update(int value)
{
	if(TIMERUNNING < STOP_TIME){
		if(PAUSE == 0)
		{
			nBody();
		}	
	}
	glutSpecialFunc( arrowFunc );
	glutKeyboardFunc( keyboardFunc );
	glutMouseFunc( mouseFunc );
	glutPostRedisplay();

	glutTimerFunc(1, update, 0);
	
	TIMERUNNING += DT;
	
}


void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	drawPicture();
	glutSwapBuffers();
	glFlush();

}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 150.0);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("GPU1");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {50.0};
	//glClearColor(0.8, 0.8, 1.0, 0.0); Light blue background
	glClearColor(0.0, 0.0, 0.2, 0.0); // Dark blue background
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	initializeBodies();
	gluLookAt(0.0, 0.0, EYEZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glutDisplayFunc(Display);
	glutTimerFunc(16, update, 0);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}

