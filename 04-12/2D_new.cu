/*
David and Mikaela
SYNC
Move to GPU
*/

//Compiling: nvcc 2D_new.cu -o 2d -lglut -lm -lGLU -lGL

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define EPSILON			0.000001
#define EYEZ			50.0 // Effectively sets x- and y-coordinates from -EYEZ to +EYEZ
#define BOUNDARY		EYEZ-1.0 // Walls
#define BDIST			10.0 // Distance from boundary at which curving should start.
#define PI			3.1415926535
#define DRAW 			100	// For draw_picture
#define XWindowSize 		1000 	// 700 initially 
#define YWindowSize 		1000 	// 700 initially

#define DT        		0.001 	// Time step
#define STOP_TIME		10.0 	// How long to go
//#define STOP_TIME		0.010

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
 
#define THREADS			10 // Number of threads per block




//Globals
float4 p[N];
float3 v[N], f[N];
float4 p_pred[M];
float3 v_pred[M], f_pred[M];
float4 p_targ[P];
float3 v_targ[P], f_targ[P];
/*
float4 *p_GPU;
float3 *v_GPU, *f_GPU;
float4 *pred_GPU, *targ_GPU;
dim3 block, grid;
*/
double SPEED = 1.0;
double TIMERUNNING = 0.0;
int PAUSE = 0;
int FOODPOINTER = 0;
double STARTRAD = 20.0;

void initializeBodies()
{
	TIMERUNNING = 0.0;
	int i;
	
	// Initialize Fish
	for(i=0; i<N; i++)
	{
		// Starting at random positions
		p[i].x = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		p[i].y = (((double)rand()/(double)RAND_MAX)-0.5)*STARTRAD;
		p[i].z = 0.0; //((double)rand()/(double)RAND_MAX)*STARTRAD;
		
		// Setting mass to 1
		p[i].w = 1.0;
		
		// Setting velocity to 0
		v[i].x = 0.0;
		v[i].y = 0.0;
		v[i].z = 0.0;
		
		// Setting force to 0
		f[i].x = 0.0;
		f[i].y = 0.0;
		f[i].z = 0.0;
	}
	
	// Initialize Predator
	if(M > 0)
	{
		for(i=0; i<M; i++)
		{
			p_pred[i].x = 40.0*sin(i*2*PI/M);
			p_pred[i].y = 40.0*cos(i*2*PI/M);
			p_pred[i].z = 0.0; //75.0*cos(i*2*PI);
			p_pred[i].w = 1.0;
			v_pred[i].x = cos(p_pred[i].x);
			v_pred[i].y = sin(p_pred[i].y);
			v_pred[i].z = 0.0;
		}
	}
	
	// Initialize Targets
	if(P > 0)
	{
		for(i=0; i<P; i++)
		{
			p_targ[i].x = 0.0;
			p_targ[i].x = 0.0;
			p_targ[i].x = 0.0;
			p_targ[i].w = 1.0;
		}
	}
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
	if(M > 0)
	{
		for(i=0; i<M; i++)
		{
			glColor3d(PREDR, PREDG, PREDB);
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
			glColor3d(TARGR, TARGG, TARGB);
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

float3 getPredForces(float4 p0, float4 ppred)
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

float3 getTargForces(float4 p0, float4 ptarg)
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

float3 getWallForces(float4 p0, float3 v0, int dim)
{
	float pdist;
	float3 force;
	
	// Right Wall
	if( p0.x > (BOUNDARY-BDIST) )
	{
		pdist = BOUNDARY - p0.x;
		if(pdist < BOUNDARY*0.25)
		{
			force.x -= (BDIST-pdist)/pdist;
		}
	}
	
	// Left Wall
	else if( p0.x < (BDIST - BOUNDARY) )
	{
		pdist = BOUNDARY + p0.x;
		if(pdist < BOUNDARY*0.25)
		{
			force.x += (BDIST - pdist)/pdist;
		} 
	}
	
	// Top Wall 
	else if( p0.y > (BOUNDARY - BDIST) )
	{
		pdist = BOUNDARY - p0.y;
		if( pdist < BOUNDARY*0.25 )
		{
			force.y -= (BDIST - pdist)/pdist;
		}
	}
	
	// Bottom Wall 
	else if( p0.y < (BDIST - BOUNDARY) )
	{
		pdist = BOUNDARY + p0.y;
		if( pdist < BOUNDARY*0.25 )
		{
			force.y += (BDIST - pdist)/pdist;
		}
	}
	
	
	/*if(dim == 3)
	{
		// Front Wall 
	
		// Back Wall
	}
	*/
	
	return(force);
}

void getForces(float4 *pos, float3 *vel, float3 *force, float4 *predp, float4 *predtarg)
{
	int i, j;
	float3 forceMag, forcePred, forceTarg, forceSum, forceWall;
	
	forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;
	
	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			if(i != j)
			{
				forceMag = getFishForces(pos[i], pos[j], vel[j]);
				forceSum.x += forceMag.x;
				forceSum.y += forceMag.y;
				forceSum.z += forceMag.z;
			}
		}
		if(M > 0)
		{
			for(j=0; j<M; j++)
			{
				forcePred = getPredForces(pos[i], predp[j]);
				forceSum.x += forcePred.x;
				forceSum.y += forcePred.y;
				forceSum.z += forcePred.z;
			}
		}
		if(P > 0)
		{
			for(j=0; j<P; j++)
			{
				forceTarg = getTargForces(pos[i], predtarg[j]);
				forceSum.x += forceTarg.x;
				forceSum.y += forceTarg.y;
				forceSum.z += forceTarg.z;
			}
		}
		
		forceWall = getWallForces(pos[i], vel[i], 2);
		forceSum.x += forceWall.x;
		forceSum.y += forceWall.y;
		forceSum.z += forceWall.z;
		
	force[i].x = forceSum.x;
	force[i].y = forceSum.y;
	force[i].z = forceSum.z;
	
	float forceNormalizer = sqrt(force[i].x*force[i].x + force[i].y*force[i].y + force[i].z*force[i].z);
	if(forceNormalizer > 25.0)
	{
		force[i].x /= forceNormalizer;
		force[i].y /= forceNormalizer;
		force[i].z /= forceNormalizer;
		
		force[i].x *= 25.0;
		force[i].y *= 25.0;
		force[i].z *= 25.0;
	}
	
	float fx, fy, fz;
	fx = force[i].x;
	fy = force[i].y;
	fz = force[i].z;
	
	printf("Fish %d has a force of (%.2f, %.2f, %.2f) acting upon it\n", i, fx, fy, fz);
	}

}

void swimFishSwim(float4 *pos, float3 *vel, float3 *force, float SPEED)
{
	int i;
	for(i=0; i<N; i++)
	{	
	
		// A sad attempt at something to try and fix things, not really sure why we tried this
		printf("Velocity[%d]:   (%.2f, %.2f, %.2f)\n", i, vel[i].x, vel[i].y, vel[i].z);
		
		float velocityNormalizer = sqrt(vel[i].x*vel[i].x + vel[i].y*vel[i].y + vel[i].z*vel[i].z);
		if(velocityNormalizer > 25.0)
		{
			printf("Velocity[%d]:   (%.2f, %.2f, %.2f)\n", i, vel[i].x, vel[i].y, vel[i].z);
		
			vel[i].x /= velocityNormalizer;
			vel[i].y /= velocityNormalizer;
			vel[i].z /= velocityNormalizer;
		}
		
		//printf("Velocity[%d]:   (%.2f, %.2f, %.2f)\n", i, vel[i].x, vel[i].y, vel[i].z);
		
		
		vel[i].x = SPEED*(vel[i].x + force[i].x*DT)/2;
		vel[i].y = SPEED*(vel[i].y + force[i].y*DT)/2;
		vel[i].z = SPEED*(vel[i].z + force[i].z*DT)/2;
		
		pos[i].x += vel[i].x*DT;
		pos[i].y += vel[i].y*DT;
		pos[i].z += vel[i].z*DT;
		
		float px, py, pz, vx, vy, vz;
		px = pos[i].x;
		py = pos[i].y;
		pz = pos[i].z;
		
		vx = vel[i].x;
		vy = vel[i].y;
		vz = vel[i].z;
		printf("Fish %d is at (%.2f, %.2f, %.2f) with a velocity of (%.2f, %.2f, %.2f)\n", i, px, py, pz, vx, vy, vz);	
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
	int i;
	double dt = DT;
	int tdraw = 0;
	float time = 0.0;
	
	if(M > 0)
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
	time += dt;
	
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



