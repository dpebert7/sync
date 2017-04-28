/*
David and Mikaela
An annoyed attempt at trying to fix things and make them work 
WTF BYE

12 April 2017
*/

//nvcc sync_gpu_2.cu -o sync_1 -lglut -lm -lGLU -lG

// Header Files to include:
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Constants for Math:
#define EPSILON			0.000001
#define PI			3.1415926535

// Constants for the Window/Drawing
#define EYEZ			50.0 // Sets x and y coordinates between -EYEZ and EYEZ
#define DRAW			1 // Draw every 10 timesteps
#define XWINDOWSIZE		1000 // How many pixels in x direction
#define YWINDOWSIZE		1000 // How many pixels in y direction
#define DIM			2 // Are we in 2D or 3D
#define DT        		0.001 	// Time step
#define STOP_TIME		10.0 	// How long to go
//#define STOP_TIME		0.05


// Constants for Force Calculations 
#define SIGHT 			10.0 // How far the fish can 'see'
#define WA			2.0 // Attraction Weight Ratio
#define WD 			5.0 // Directional Weight Ratio
#define CA 			2.0 // Attraction Coefficient
#define CR			1.0 // Repulsion Coefficient
#define CPR			100000000.0 // Repulsion Coefficient (predator)
#define CTA			100.0 // Attraction Coefficient (Target)
#define BOUNDARY		EYEZ-1.0 // Walls
#define BDIST			10.0 // Distance from boundary at which curving should start.

// Constants for Fishies
#define N			16 // Number of fish
#define FISHRAD			0.4 // Radius of fish spheres
#define FISHR			1.0 // Red of fish color
#define FISHG			1.0 // Green of fish color
#define FISHB			0.5 // Blue of fish color
#define FISHTAIL		3.0 // How wide tail should be 
#define MINSPEED		0.1 // Slowest possible speed
#define MAXPEED			500 // Fastest possible speed

// Constants for Predators
#define M			1 // Number of Predators
#define PREDRAD			0.9 // Radius of predator
#define PREDR			1.0 // Red of predator color
#define PREDG			0.5 // Green of predator color
#define PREDB			1.0 // Blue of predator color
#define PREDTAIL		3.0 // How wide tail should be

// Constants for Targets
#define P			1 // Number of Targets
#define TARGRAD			0.5 // Radius of target (food)
#define TARGR			0.0 // Red of target color
#define TARGG			0.0 // Green of target color
#define TARGB			1.0 // Blue of target color 
#define TARGTAIL		3.0 // How wide tail should be

// Global Variables
float4 p[N], p_pred[M], p_targ[P]; // positions for fish, predators, and targets
float3 v[N], v_pred[M], v_targ[P]; // velocities for fish, predators, and targets
float3 f[N], f_pred[M], f_targ[P]; // forces for fish, predators, and targets

double SPEED = 10.0; // Multiplier for speed - LOOK INTO THIS. IT SUCKS AND FUCK IT
double TIMERUNNING = 0.0; // Stores how long the nBody code has been running
int PAUSE = 0; // hold whether we are pausing simulatin
int FOODPOINTER = 0; // where does the first target particle start at
double STARTRAD = 20.0; //how big of a box do we want to start the fish in 

void initializeBodies()
{
	// A function to initialize position, velocity, and force of fish, predators, and targets.
	int i;
	
	// Initializing Fish
	for(i=0; i<N; i++)
	{
		// Start the fish at random positions
		p[i].x = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		p[i].y = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		if(DIM == 3)
		{
			p[i].z = ((double)rand()/(double)RAND_MAX)*STARTRAD;
		}
		else
		{
			p[i].z = 0.0;
		}
		// Set mass to 1 - can change later if needed, but this is fine for now
		p[i].w = 1.0;
		
		// Set starting velocity to 0
		v[i].x = 0.0;
		v[i].y = 0.0;
		v[i].z = 0.0;
		
		// Set starting force to 0
		f[i].x = 0.0;
		f[i].y = 0.0;
		f[i].z = 0.0;
	}
	
	// Initializing Predator
	if(M > 0) // Only initialize the predator if we have at least one predator
	{
		for(i=0; i<M; i++)
		{
			// Set initial positions to be in a circle of radius 40
			p_pred[i].x = 40.0*sin(i*2*PI/M);
			p_pred[i].y = 40.0*cos(i*2*PI/M);
			
			// Set the mass to be 1- can change later if needed
			p_pred[i].w = 1.0;
			
			// Not sure why we set these original velocities, but don't want to mess with too much right now
			v_pred[i].x = cos(p_pred[i].x); 			// WHY ARE WE DOING THIS HERE I DONT KNOW WHAT 
			v_pred[i].y = sin(p_pred[i].y);
			
			if(DIM ==3)
			{
				p_pred[i].z = 75.0*cos(i*2*PI);
				v_pred[i].z = 0.0;
			}
			else
			{
				p_pred[i].z = 0.0;
				v_pred[i].z = 0.0;
			}
			
			// Don't need forces to start, so set them to 0
			f_pred[i].x = 0.0;
			f_pred[i].y = 0.0;
			f_pred[i].z = 0.0;
		}
	}
	
	// Initializng Target
	if(P > 0) // Only initialize target particle if we have at least one target
	{
		// Set the target to start at the center
		p_targ[i].x = 0.0;
		p_targ[i].x = 0.0;
		p_targ[i].x = 0.0;
		
		// Set the initial mass to be 1
		p_targ[i].w = 1.0;
		
		v_targ[i].x = 0.0;
		v_targ[i].y = 0.0;
		v_targ[i].z = 0.0;
		
		f_targ[i].x = 0.0;
		f_targ[i].y = 0.0;
		f_targ[i].z = 0.0;	
	}
}

void drawLines()
{
	// Draw back lines
	glLineWidth(5.0); 
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	glVertex3f(-50.0, -50.0, -50.0);
	glVertex3f( 50.0, -50.0, -50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f( 50.0, -50.0, -50.0);
	glVertex3f( 50.0,  50.0, -50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f( 50.0,  50.0, -50.0);
	glVertex3f(-50.0,  50.0, -50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-50.0,  50.0, -50.0);
	glVertex3f(-50.0, -50.0, -50.0);
	glEnd();
	// End lines

	// Draw side lines
	glLineWidth(10.0); 
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	glVertex3f( 50.0,  50.0, -50.0);
	glVertex3f( 50.0,  50.0,  50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f( 50.0, -50.0, -50.0);
	glVertex3f( 50.0, -50.0,  50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-50.0,  50.0, -50.0);
	glVertex3f(-50.0,  50.0,  50.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-50.0, -50.0, -50.0);
	glVertex3f(-50.0, -50.0,  50.0);
	glEnd();
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
		
		glColor3d(FISHR, FISHG, FISHB); 	//Object color
		
		glLineWidth(FISHTAIL);			// How wide should the tail line be
		glBegin(GL_LINES);
		glVertex3f(p[i].x, p[i].y, p[i].z);
		glVertex3f( p[i].x - (v[i].x/velMag), p[i].y - (v[i].y/velMag), p[i].z - (v[i].z/velMag) );
		glEnd();
		
		glPushMatrix();
		glTranslatef(p[i].x, p[i].y, p[i].z);
		glutSolidSphere(FISHRAD, 10, 10);    //First argument affects size.
		glPopMatrix();
	}
	
	// Drawing the predators
	if(M > 0)
	{
		for(i=0; i<M; i++)
		{
			float velMag = sqrt(v_pred[i].x*v_pred[i].x + v_pred[i].y*v_pred[i].y + v_pred[i].z*v_pred[i].z);
			
			glColor3d(PREDR, PREDG, PREDB);
			
			glLineWidth(PREDTAIL);			// How wide should the tail line be
			glBegin(GL_LINES);
			glVertex3f(p_pred[i].x, p_pred[i].y, p_pred[i].z);
			glVertex3f( p_pred[i].x - (v_pred[i].x/velMag), p_pred[i].y - (v_pred[i].y/velMag), p_pred[i].z - (v_pred[i].z/velMag) );
			glEnd();
		
			glPushMatrix();
			glTranslatef(p_pred[i].x, p_pred[i].y, p_pred[i].z);
			glutSolidSphere(PREDRAD, 10, 10);
			glPopMatrix();
		}
	}
	
	// Drawing the targets
	if(P > 0)
	{
		for(i=0; i<P; i++)
		{
			float velMag = sqrt(v_targ[i].x*v_targ[i].x + v_targ[i].y*v_targ[i].y + v_targ[i].z*v_targ[i].z);
			
			glColor3d(TARGR, TARGG, TARGB);
			
			glLineWidth(TARGTAIL);			// How wide should the tail line be
			glBegin(GL_LINES);
			glVertex3f(p_targ[i].x, p_targ[i].y, p_targ[i].z);
			glVertex3f( p_targ[i].x - (v_targ[i].x/velMag), p_targ[i].y - (v_targ[i].y/velMag), p_targ[i].z - (v_targ[i].z/velMag) );
			glEnd();
			
			glPushMatrix();
			glTranslatef(p_targ[i].x, p_targ[i].y, p_targ[i].z);
			glutSolidSphere(TARGRAD, 10, 10);
			glPopMatrix();
		}
	}
	glutSwapBuffers();
}

float3 getFishForces(float4 p0, float4 p1, float3 v1)
{
	// A function called that calculates the forces between any 2 fish given their positions and the velocity of the fish that isn't " ME"
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2) + EPSILON;
	float r4 = r2*r2 + EPSILON;
	
	if(r < SIGHT)
	{
		f.x = WA*(CA*(dx/r2) - CR*(dx/r4)) + WD*(v1.x/r);
		f.y = WA*(CA*(dy/r2) - CR*(dy/r4)) + WD*(v1.y/r);
		f.z = WA*(CA*(dz/r2) - CR*(dz/r4)) + WD*(v1.z/r);
	}
	else
	{	
		f.x = 0.0;
		f.y = 0.0;
		f.z = 0.0;
	}
	
	return(f);
}

float3 getPredForces(float4 p0, float4 p1)
{
	// A function called that calculates the forces between a fish and a predator
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2) + EPSILON;
	float r4 = r2*r2 + EPSILON;
	
	f.x = -(CPR*dx)/r4;
	f.y = -(CPR*dy)/r4;
	f.z = -(CPR*dz)/r4;
	
	return(f);
}

float3 getTargForces(float4 p0, float4 p1)
{
	// A function called that calculates the forces between a fish and a target
	float3 f;
	float dx = p1.x - p0.x;
	float dy = p1.y - p0.y;
	float dz = p1.z - p0.z;
	
	float r2 = dx*dx + dy*dy + dz*dz;
	float r = sqrt(r2) + EPSILON;
	float r4 = r2*r2 + EPSILON;
	
	f.x = (CTA*dx)/r2;
	f.y = (CTA*dy)/r2;
	f.z = (CTA*dz)/r2;
	
	return(f);
}

float3 getWallForces(float4 p0, int dim)
{
	float pdist;
	float3 f;
	
	f.x = 0.0;
	f.y = 0.0;
	f.z = 0.0;
	
	// Right Wall
	if(p0.x > (BOUNDARY - BDIST))
	{
		pdist = BOUNDARY - p0.x;
		if(pdist < BOUNDARY*0.25)
		{
			f.x -= (BDIST - pdist)/pdist;
		}
	}
	
	// Left Wall
	if(p0.x < (BDIST - BOUNDARY))
	{
		pdist = BOUNDARY + p0.x;
		if(pdist < BOUNDARY*0.25)
		{
			f.x += (BDIST - pdist)/pdist;
		}
	}
	
	// Top Wall
	if(p0.y > (BOUNDARY - BDIST))
	{
		pdist = BOUNDARY - p0.y;
		if(pdist < BOUNDARY*0.25)
		{
			f.y -= (BDIST - pdist)/pdist;
		}
	}
	
	// Bottom Wall
	if(p0.y < (BDIST - BOUNDARY))
	{
		pdist = BOUNDARY + p0.y;
		if(pdist < BOUNDARY*0.25)
		{
			f.y += (BDIST - pdist)/pdist;
		}
	}
	
	if(dim == 3)
	{
		// Front Wall 
		if(p0.z > (BOUNDARY - BDIST))
		{
			pdist = BOUNDARY - p0.z;
			if(pdist < BOUNDARY*0.25)
			{
			f.z -= (BDIST - pdist)/pdist;
			}
		}	
		
		// Back Wall
		if(p0.z < (BOUNDARY - BDIST))
		{
			pdist = BOUNDARY + p0.z;
			if(pdist < BOUNDARY*0.25)
			{
			f.z += (BDIST - pdist)/pdist;
			}
		}
	}
	
	return(f);
}

void getForces(float4 *p, float3 *v, float3 *f, float4 *p_pred, float4 *p_targ)
{
	int i, j;
	float3 forceSum0, predForce, targForce, wallForce, fishForce;
	
	forceSum0.x = 0.0;
	forceSum0.y = 0.0;
	forceSum0.z = 0.0;
	
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			if(i != j)
			{
				fishForce = getFishForces(p[i], p[j], v[j]);
				forceSum0.x += fishForce.x;
				forceSum0.y += fishForce.y;
				forceSum0.z += fishForce.z;
			}
		}
		
		/*if(M > 0)
		{
			for(j=0; j<M; j++)
			{
				predForce = getPredForces(p[i], p_pred[j]);
				forceSum0.x += predForce.x;
				forceSum0.y += predForce.y;
				forceSum0.z += predForce.z;
			}
		}
		
		if(P > 0)
		{
			for(j=0; j<p; j++)
			{
				targForce = getTargForces(p[i], p_targ[j]);
				forceSum0.x += targForce.x;
				forceSum0.y += targForce.y;
				forceSum0.z += targForce.z;
			}
		}
		*/
		wallForce = getWallForces(p[i], 2);
		forceSum0.x += wallForce.x;
		forceSum0.y += wallForce.y;
		forceSum0.z += wallForce.z;
		
		
		f[i].x = forceSum0.x;
		f[i].y = forceSum0.y;
		f[i].z = forceSum0.z;
		
		/*float forceNormalizer = sqrt(f[i].x*f[i].x + f[i].y*f[i].y + f[i].z*f[i].z);
		
		f[i].x /= forceNormalizer;
		f[i].y /= forceNormalizer;
		f[i].z /= forceNormalizer;
		*/
		
		// Print Statements for Diagnostics
		printf("Fish %d has a force of (%.2f, %.2f, %.2f) acting upon it\n", i, f[i].x, f[i].y, f[i].z);
	}
	
	
}

void swimFishSwim(float4 *p, float3 *v, float3 *f, float SPEED)
{
	int i;
	for(i=0; i<N; i++)
	{
		v[i].x += f[i].x*DT;
		v[i].y += f[i].y*DT;
		v[i].z += f[i].z*DT;
		
		p[i].x += v[i].x*DT;
		p[i].y += v[i].y*DT;
		p[i].z += v[i].z*DT;
	}
}

void goPredGo(float4 *predp, float3 *predv, float3 *predf, int m)
{
	int i;
	for(i=0; i<m; i++)
	{
		predf[i].x = 0.0 - predp[i].x;
		predf[i].y = 0.0 - predp[i].y;
		predf[i].z = 0.0 - predp[i].z;
	
		predv[i].x += predf[i].x;
		predv[i].y += predf[i].y;
		predv[i].z += predf[i].z;
	
		predp[i].x += predv[i].x*DT;
		predp[i].y += predv[i].y*DT;
		predp[i].z += predv[i].z*DT;
	}
}

void nBody()
{
	int i;
	int tdraw = 0;

	if (M > 0)
	{
		goPredGo(p_pred, v_pred, f_pred, M);
	}
	
	getForces(p, v, f, p_pred, p_targ);
	swimFishSwim(p, v, f, SPEED);
	
	if(tdraw == DRAW)
	{
		drawPicture();
		tdraw = 0;
	}
	tdraw++;
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
			coord[0] = (x*EYEZ*2.0/XWINDOWSIZE)-EYEZ;
			coord[1] = -(y*EYEZ*2.0/YWINDOWSIZE)+EYEZ;
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
	glutInitWindowSize(XWINDOWSIZE,YWINDOWSIZE);
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




