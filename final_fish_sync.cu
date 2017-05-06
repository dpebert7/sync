/*
David and Mikaela
SYNC
6 May 2017

This is the final code for our math modeling class project.
N is the number of fish. Tested up to N=2048.
Make sure the THREADSPERBLOCK divides evenly into N, else problems will occur.
*/

//nvcc final_fish_snyc.cu -o final_fish_snyc -lglut -lm -lGLU -lGL && ./final_fish_snyc

// Header Files to include:
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constants for Math:
#define EPSILON        0.000001
#define PI            3.1415926535

// Constants for the Window/Drawing
#define EYEZ            50.0    // Sets x and y coordinates between -EYEZ and EYEZ
#define DRAW            10         // Draw every 10 timesteps
#define XWINDOWSIZE		800     // How many pixels in x direction
#define YWINDOWSIZE		800     // How many pixels in y direction
#define DIM				3         // Are we in 2D or 3D
#define DT				0.005     // Time step
#define STOP_TIME		20.0     // How long to go
//#define STOP_TIME        0.005


// Constants for Force Calculations 
#define SIGHT			10.0 // How far the fish can 'see'
#define WA				2.0 // Attraction Weight Ratio
#define WD				50.0 // Directional Weight Ratio
#define CA				15.0 // Attraction Coefficient
#define CR				60.0 // Repulsion Coefficient
#define CTA				5000.0 // Attraction Coefficient (Target)
//#define CPR            100000000.0 // Repulsion Coefficient (predator)


// Constants for Food
#define P				100 //Max Number of food particles
#define FOODRAD			1.0
float    foodrad[P];        // Food radius may change over time.
#define FOODR			0.529412 // Red of target color
#define FOODG			0.807843 // Green of target color
#define FOODB			0.921569 // Blue of target color 

// Constants for GPU
#define THREADSPERBLOCK    12 // Number of threads per block

// Constants for Fishies
#define N                12 // Number of fish
#define FISHRAD            0.4 // Radius of fish spheres
#define FISHR            1.0 // 1.0 Red of fish color
#define FISHG            0.3 // 0.8 Green of fish color
#define FISHB            0.3 // 0.5 Blue of fish color
#define FISHTAIL        3.0 // How wide tail should be 
#define MINSPEED        0.1 // Slowest possible speed
#define MAXSPEED        25.0 // Fastest allowable speed


// Constants for Wall Calculations
#define BOUNDARY        EYEZ-1.0 // Walls
#define BDIST            10.0 // Distance from boundary at which curving should start.
#define WALLSTRENGTH    MAXSPEED*100.0 // How strong our walls should be

// Global Variables
float4 p[N]; // positions for fish,
float3 v[N]; // velocities for fish, 
float3 f[N]; // forces for fish, 
float4 p_food[P]; // positions for food
float3 v_food[P], f_food[P]; // velocity and force on food
int fishCounter[N]; // Counter to hold how many fish are within sight radius of each fish

//Globals for GPU
float4 *p_GPU;
float3 *v_GPU, *f_GPU;
float4 *p_food_GPU;
float3 *v_food_GPU, *f_food_GPU;
int *fishCounter_GPU;
dim3 block, grid;

double SPEED        = 10.0; // Multiplier for speed - LOOK INTO THIS. IT SUCKS AND FUCK IT
double TIMERUNNING = 0.0; // Stores how long the nBody code has been running
int PAUSE            = 0; // hold whether we are pausing simulatin
int FOODPOINTER    = 0; // where does the first target particle start at
double STARTRAD    = 48.0; //how big of a box do we want to start the fish in 

void initializeBodies()
{
    // A function to initialize position, velocity, and force of fish,
    int i;
    
    // Initializing Fish
    for(i=0; i<N; i++)
    {
        // Start the fish at random positions
        p[i].x = (((float)rand()/(float)RAND_MAX)-0.5)*STARTRAD;
        p[i].y = (((float)rand()/(float)RAND_MAX)-0.5)*STARTRAD;
        if(DIM == 3)
        {
            p[i].z = (((float)rand()/(float)RAND_MAX)*20.0)-30.0; // z positions between -30 and -10
        }
        else
        {
            p[i].z = -20.0;
        }
        // Set mass to 1 - can change later if needed, but this is fine for now
        p[i].w = 1.0;
        
        // Set starting velocity to 0
        v[i].x = 0.0;
        v[i].y = 25.0;
        v[i].z = 0.0;
        
        // Set starting force to 0
        f[i].x = 0.0;
        f[i].y = 0.0;
        f[i].z = 0.0;
    }
    
    // Initialize Targets
    for(i=0; i<P; i++)
    {
        p_food[i].x = 10000.0;
        p_food[i].y = 0.0;
        p_food[i].z = 0.0;
        p[i].w = 1.0; // Set mass to 1
        
        // Starting radius    
        foodrad[i] = FOODRAD;
        
        // Set starting velocity to 0
        v_food[i].x = 0.0;
        v_food[i].y = 0.0;
        v_food[i].z = -2.0;   /// How fast the food will sink after it's dropped
    }

    block.x = THREADSPERBLOCK;
    block.y = 1;
    block.z = 1;
    
    grid.x = (N-1)/block.x + 1;
    grid.y = 1;
    grid.z = 1;

    cudaMalloc( (void**)&p_GPU,           N*sizeof(float4) );
    cudaMalloc( (void**)&v_GPU,           N*sizeof(float3) );
    cudaMalloc( (void**)&f_GPU,           N*sizeof(float3) );
    cudaMalloc( (void**)&p_food_GPU,      P*sizeof(float4) );
    cudaMalloc( (void**)&v_food_GPU,      P*sizeof(float3) );
    cudaMalloc( (void**)&f_food_GPU,      P*sizeof(float3) );
    cudaMalloc( (void**)&fishCounter_GPU, N*sizeof(int) );
    
    // Copy memory over to GPU for the first and only time
    cudaMemcpy(p_GPU,      p,      N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(v_GPU,      v,      N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(f_GPU,      f,      N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(p_food_GPU, p_food, P*sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(v_food_GPU, v_food, P*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(f_food_GPU, f_food, P*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(fishCounter_GPU, fishCounter, N*sizeof(int), cudaMemcpyHostToDevice);
}


void drawLines()
{
    // Draw back lines
    glLineWidth(6.0); 
    glColor3f(1.0, 1.0, 1.0);
    //glColor3f(0.9, 0.9, 0.9);
    glBegin(GL_LINES);
    glVertex3f(-50.0, -50.0, -40.0);
    glVertex3f( 50.0, -50.0, -40.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f( 50.0, -50.0, -40.0);
    glVertex3f( 50.0,  50.0, -40.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f( 50.0,  50.0, -40.0);
    glVertex3f(-50.0,  50.0, -40.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-50.0,  50.0, -40.0);
    glVertex3f(-50.0, -50.0, -40.0);
    glEnd();
    // End lines

    // Draw side lines
    glLineWidth(10.0); 
    //glColor3f(0.9, 0.9, 0.9);
    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_LINES);
    glVertex3f( 50.0,  50.0, -40.0);
    glVertex3f( 50.0,  50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f( 50.0, -50.0, -40.0);
    glVertex3f( 50.0, -50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-50.0,  50.0, -40.0);
    glVertex3f(-50.0,  50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-50.0, -50.0, -40.0);
    glVertex3f(-50.0, -50.0, - 0.0);
    glEnd();
    
    // Draw front lines
    glLineWidth(5.0); 
    glColor3f(1.0, 0.0, 0.0);
    glBegin(GL_LINES);
    glVertex3f(-50.0, -50.0, - 0.0);
    glVertex3f( 50.0, -50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f( 50.0, -50.0, - 0.0);
    glVertex3f( 50.0,  50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f( 50.0,  50.0, - 0.0);
    glVertex3f(-50.0,  50.0, - 0.0);
    glEnd();
    glBegin(GL_LINES);
    glVertex3f(-50.0,  50.0, - 0.0);
    glVertex3f(-50.0, -50.0, - 0.0);
    glEnd();
    // End lines
// End lines
}

void drawPicture()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    
    drawLines();

    int i;
    
    // Drawing the fish
    for(i=0; i<N; i++)
    {
        float velMag = sqrt(v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
        
        glColor3d(FISHR, FISHG, FISHB);     //Object color
        
        glLineWidth(FISHTAIL);            // How wide should the tail line be
        glBegin(GL_LINES);
        glVertex3f(p[i].x, p[i].y, p[i].z);
        glVertex3f( p[i].x - (v[i].x/velMag), p[i].y - (v[i].y/velMag), p[i].z - (v[i].z/velMag) );
        glEnd();
        
        glPushMatrix();
        glTranslatef(p[i].x, p[i].y, p[i].z);
        glutSolidSphere(FISHRAD, 10, 10);    //First argument affects size.
        glPopMatrix();
    }

    // Drawing the food
    for(i=0; i<P; i++)
    {
        //float velMag = sqrt(v_food[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
        glColor3d(FOODR, FOODG, FOODB);     //Object color
        glPushMatrix();
        glTranslatef(p_food[i].x, p_food[i].y, p_food[i].z);
        glutSolidSphere(foodrad[i], 10, 10);    //First argument affects size.
        glPopMatrix();
    }
    
    glutSwapBuffers();
    //makePicture("record1",1000,1000,1);//Call this after glutSwapBuffers.
}


void countFish(float4 *p)
{
    for(int i = 0; i < N; i++)
    {
        fishCounter[i] = 0;
        for(int j=0; j < N; j++)
        {
            float3 d;
            d.x = p[j].x-p[i].x;
            d.y = p[j].y-p[i].y;
            d.z = p[j].z-p[i].z;
    
            float r2 = d.x*d.x + d.y*d.y + d.z*d.z;
            float r = sqrt(r2) + EPSILON;
    
            if(r < SIGHT)
            {
                fishCounter[i]++;
            }
        }
        //printf("There are %d fish in sight radius of fish %d\n", fishCounter[i], i);
    }
}

float3 getFishForces(float4 p0, float4 p1, float3 v1, int fishCounter)
{    
    // A function to calcultae the forces between 
    float3 f, d;
    d.x = p1.x - p0.x;
    d.y = p1.y -p0.y;
    d.z = p1.z -p0.z;
    
    float r2 = d.x*d.x + d.y*d.y + d.z*d.z;
    float r = sqrt(r2) + EPSILON;
    float r4 = r2*r2 + EPSILON;
    
    //printf("There is a distance of %.2f between 2 fish\n", r);
    float vMag = sqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
    
    if(r < SIGHT)
    {
        f.x = WA*(CA*(d.x/r2) - CR*(d.x/r4)) + WD*(v1.x/(r*vMag))/(float)fishCounter;
        f.y = WA*(CA*(d.y/r2) - CR*(d.y/r4)) + WD*(v1.y/(r*vMag))/(float)fishCounter;
        f.z = WA*(CA*(d.z/r2) - CR*(d.z/r4)) + WD*(v1.z/(r*vMag))/(float)fishCounter;
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
    
    if(r < SIGHT*3.0)
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

__device__ float3 getWallForces(float4 p0, int dim)
{
    float3 f;
    
    f.x = 0.0;
    f.y = 0.0;
    f.z = 0.0;
    
    float wallStart = BOUNDARY - BDIST; //BOUNDARY for x and y direction is 49, BDIST is 10
                                        //wallStart is 39 in x and y direction
    
    // Right Wall
    if(p0.x > wallStart){f.x -= WALLSTRENGTH/(BOUNDARY - p0.x);}
        
    // Left Wall
    if(p0.x < -wallStart){f.x += WALLSTRENGTH/(BOUNDARY + p0.x);}
        
    // Top Wall
    if(p0.y > wallStart){f.y -= WALLSTRENGTH/(BOUNDARY - p0.y);}
        
    // Bottom Wall
    if(p0.y < -wallStart){f.y += WALLSTRENGTH/(BOUNDARY + p0.y);}
    
    if(dim == 3)
    {
        // Front Wall
        if(p0.z > -11.0){f.z -= WALLSTRENGTH/(-1.0 - p0.z);}
        
        // Back Wall
        if(p0.z < -29.0){f.z += WALLSTRENGTH/(40.0 + p0.z);}
    }
    
    return(f);
}



__device__ float3 getFishForcesDevice(float4 p0, float4 p1, float3 v1, int fishCounter)
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
        f.x = WA*(CA*(dx/r2) - CR*(dx/r4)) + WD*(v1.x/r)/(float)fishCounter; 
                                            // In some versions, WD is divided by vMag.
        f.y = WA*(CA*(dy/r2) - CR*(dy/r4)) + WD*(v1.y/r)/(float)fishCounter;
        f.z = WA*(CA*(dz/r2) - CR*(dz/r4)) + WD*(v1.z/r)/(float)fishCounter;
    }
    else // If not, this particular fish does not affect the forces on p0
    {
        f.x = 0.0;
        f.y = 0.0;
        f.z = 0.0;
    }
    
    return(f);    
}



__device__ int getRadiusNumberDevice(float4 p0, float4 p1)
{
    int fishCounter = 0;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
    float r = sqrt(r2) + EPSILON;

    if(r < SIGHT)
    {
        fishCounter ++;
    }

    return(fishCounter);
}


__global__ void getAllForcesKernel(float4 *p, float3 *v, float3 *f, int *fishCounter, float4 *p_food)
{
    
    int i, j, ii;
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    float3 forceSum0, forceMag, forceTarg, forceWall;
    float4 posMe;
    int cntMe;
    //int numberFishies;
    
    __shared__ float4 shPos[THREADSPERBLOCK];
    __shared__ float3 shVel[THREADSPERBLOCK];
    
    forceSum0.x = 0.0;
    forceSum0.y = 0.0;
    forceSum0.z = 0.0;
    
    posMe.x = p[id].x;
    posMe.y = p[id].y;
    posMe.z = p[id].z;
    posMe.w = p[id].w;
    cntMe = fishCounter[id];
    
    
    for(j=0; j<gridDim.x; j++)
    {
        shPos[threadIdx.x] = p[threadIdx.x + blockDim.x*j];
        shVel[threadIdx.x] = v[threadIdx.x + blockDim.x*j];
                                            //^^^Wyatt's code has blockDim.x, we have blockIdx.x
        __syncthreads();
        
        
        #pragma unroll 32
        for(i=0; i<blockDim.x; i++)
        {
            ii = i + blockDim.x*j;
            if(ii != id)
            {    
                forceMag = getFishForcesDevice(posMe, shPos[i], shVel[i], cntMe); 
                forceSum0.x += forceMag.x;
                forceSum0.y += forceMag.y;
                forceSum0.z += forceMag.z;
            }
            __syncthreads();
        }
        __syncthreads();
    }
    
    for(i=0; i<P; i++)
    {
        forceTarg = getTargForces(posMe, p_food[i]);
        forceSum0.x += forceTarg.x;
        forceSum0.y += forceTarg.y;
        forceSum0.z += forceTarg.z;
        __syncthreads();
    }
    
    forceWall = getWallForces(posMe,DIM);
    forceSum0.x += forceWall.x;
    forceSum0.y += forceWall.y;
    forceSum0.z += forceWall.z;
    __syncthreads();
        
    f[id].x = forceSum0.x;
    f[id].y = forceSum0.y;
    f[id].z = forceSum0.z;
    __syncthreads();
}


void sinkFood(float4 *p, float3 *v, float3 *f, float4 *p_fish)
{
    int i;
    for(i=0; i<P; i++)
    {
        // Update velocity and force. Currently not necessary.
        /*
        v[i].x += f[i].x*DT;
        v[i].y += f[i].y*DT;
        v[i].z += f[i].z*DT;
        
        float vMag = sqrt(v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z);
        
        if(vMag > MAXSPEED)
        {
            v[i].x *= (MAXSPEED/vMag);
            v[i].y *= (MAXSPEED/vMag);
            v[i].z *= (MAXSPEED/vMag);
        
        */
        
        p[i].x += v[i].x*DT;
        p[i].y += v[i].y*DT;
        p[i].z += v[i].z*DT;
        
        // Iterate through fish to see which are eating
        for(int j = 0; j < N; j++)
        {
            float3 dist;
            dist.x = p_fish[j].x - p[i].x;
            dist.y = p_fish[j].y - p[i].y;
            dist.z = p_fish[j].z - p[i].z;
        
            float distance = sqrt(dist.x*dist.x + dist.y*dist.y + dist.z*dist.z) + EPSILON;

            if(distance < 1.5)
            {
                printf("Fish %i is eating food %i\n", j, i);
                printf("Fish %d has a position of (%.2f, %.2f, %.2f)\n", j, p_fish[j].x, p[j].y, p[j].z);
                printf("Food %d has a position of (%.2f, %.2f, %.2f)\n", i, p[i].x, p_food[i].y, p_food[i].z);
                
                foodrad[i] /= 1.1;
                
                if(foodrad[i]<0.1)
                {
                    p_food[i].x = 1000.0;
                }
            }
        }
        
        //printf("Food %d has a velocity of (%.2f, %.2f, %.2f)\n", i, v[i].x, v[i].y, v[i].z);
        //printf("Food %d has a position of (%.2f, %.2f, %.2f)\n", i, p[i].x, p[i].y, p[i].z);
    }
}

__global__ void swimFishKernel(float4 *p, float3 *v, float3 *f)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;

    v[id].x += f[id].x*DT;
    v[id].y += f[id].y*DT;
    v[id].z += f[id].z*DT;
    
    float vMag = sqrt(v[id].x*v[id].x + v[id].y*v[id].y + v[id].z*v[id].z);
    if(vMag > MAXSPEED)
    {
        v[id].x *= (MAXSPEED/vMag);
        v[id].y *= (MAXSPEED/vMag);
        v[id].z *= (MAXSPEED/vMag);
    }
    //printf("Fish %d has a velocity of (%.2f, %.2f, %.2f)\n", id, v[id].x, v[id].y, v[id].z);
    
    p[id].x += v[id].x*DT;
    p[id].y += v[id].y*DT;
    p[id].z += v[id].z*DT;
}



void nBody()
{
    cudaError_t err;
    
    countFish(p); // This is now done on the CPU instead of the GPU
    sinkFood(p_food, v_food, f_food, p);
    
    cudaMemcpy(fishCounter_GPU, fishCounter, N*sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(p_food_GPU,         p_food,      P*sizeof(float4), cudaMemcpyHostToDevice);
    getAllForcesKernel<<<grid, block>>>(p_GPU, v_GPU, f_GPU, fishCounter_GPU, p_food_GPU);
    swimFishKernel<<<    grid, block>>>(p_GPU, v_GPU, f_GPU);
    cudaMemcpy(p, p_GPU, N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, v_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(f, f_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
    
    
    for(int i=0; i<10; i++)
    {
        //printf("Fish %d has a force of (%.2f, %.2f, %.2f)\n", i, f[i].x, f[i].y, f[i].z);
        //printf("Fish %d has a velocity of (%.2f, %.2f, %.2f)\n", i, v[i].x, v[i].y, v[i].z);
        //printf("Fish %d has a position of (%.2f, %.2f, %.2f)\n", i, p[i].x, p[i].y, p[i].z);
    }
    
    err = cudaGetLastError();
    if (err != 0) 
    {
        printf("\n CUDA error = %s\n", cudaGetErrorString(err));
        //return(1);
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
            printf("There are %i fish remaining\n", j);    

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
            coord[0] = (x*EYEZ*2.0/XWINDOWSIZE)-EYEZ;
            coord[1] = -(y*EYEZ*2.0/YWINDOWSIZE)+EYEZ;
            coord[2] = -1.0;
            printf("Food %i is at (%.4f, %.4f, %.4f)\n", FOODPOINTER,
                coord[0], coord[1], coord[2]);
            p_food[FOODPOINTER].x = coord[0];
            p_food[FOODPOINTER].y = coord[1];
            p_food[FOODPOINTER].z = coord[2];
            foodrad[FOODPOINTER]  = FOODRAD;
            
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
    //glutSwapBuffers();
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
    glutInitWindowSize(XWINDOWSIZE,YWINDOWSIZE);
    glutInitWindowPosition(0,0);
    glutCreateWindow("");
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
