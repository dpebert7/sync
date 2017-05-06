/*
David & Mikaela
Sync - Test Human swarm

Final-ish Script for Friday 2/24/16
Uses Hooke's Law for Forces - "springs" between particles

Output:
N humans walk toward center of circle.
*/

// gcc swarm2.4.c -o temp -lglut -lm -lGLU -lGL
// ./temp
// To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define XWindowSize 700 //700 initially 
#define YWindowSize 700 //700 initially

//#define STOP_TIME 	0.0003 	// Go few steps before stopping.  
#define STOP_TIME 	100.0	// Go lots of steps
#define STOP_DISTANCE 	0.5	// Distance at which particles should stop.
#define DT        	0.0001 	// Step size, global variable
#define DRAW 		10 	// Not sure?
#define EPSILON	     	0.001 	// Small number
#define PI		3.1415926
#define SEARCH_DISTANCE	10	//Only care about neighbors that are within this radius
				// This value seems really important.

//			
#define DAMP 200		
#define K1 100
#define N 500  // number of bodies


// Globals  //2-vectors for storing position & mass in x, y, and z directions.
double 	p[N][3],		// positions x,y,z
	v[N][3],		// sum velocity x,y,z
	vc[N][3],		// velocities x,y,z toward center
	vn[N][3],		// velocities x,y,z away from closest neighbor
	r,			// distance between particles
	CENTER[3],		// Center of Bodies
	SENSITIVITY[N],		// How sensitive a body is to its neighbors.	
	f[N][3];		// force vector	
int MIN_POS[N][N];// Minimum of distances. Column 1 is distance; Column 2 is index 0 to N-1

void set_initial_conditions() // Initial conditions
{	
	int i, j;
		
/*
	// Option 1: Particles are positioned in a circle
	for(i = 0; i<N; i++){
		p[i][0] = sin(i*2*PI/N)*2.5;
		p[i][1] = cos(i*2*PI/N)*2.5;
		p[i][2] = 0.0;
	}
//*/
	
//*
	// Option 2: Random
	srand(time(NULL));
	for(i = 0; i<N; i++)
	{
		p[i][0] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		p[i][1] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		p[i][2] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
	}
//*/

	// Setting the Center to 0
	CENTER[0] = 0.0;
	CENTER[1] = 0.0;
	CENTER[2] = 0.0;
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	int i;
	for(i=0;i<1;i++){
		glColor3d(0.65,0.16,0.16); 	//brown color
		SENSITIVITY[i]=1.0;
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.025,10,10);  //First argument affects size.
		glPopMatrix();
	}
	for(i=1;i<2;i++){
		glColor3d(1.0,0.5,1.0); 	//pink color
		SENSITIVITY[i]=1.0;
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.025,10,10);  //First argument affects size.
		glPopMatrix();
	}
	for(i=2;i<N;i++){
		glColor3d(1.0,1.0,0.5); 	//yellow color
		SENSITIVITY[i]=1.0;
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.025,10,10);  //First argument affects size.
		glPopMatrix();
	}
	glutSwapBuffers();
}

int n_body()
{
	double d[3], dt, time = 0.0;
	double magc, magn, fmag, dist;
	int tdraw = 0;
	int tprint = 0;
	int i, j, k;
	
	dt = DT;
	
	//Find distance between bodies and get forces
	while (time < STOP_TIME)
	{
		for(i=0; i<N; i++)
		{
			// reset the force 
			f[i][0] = 0.0;
			f[i][1] = 0.0;
			f[i][2] = 0.0;
		}
		
		for(i=0; i<N; i++)
		{
			for(j=i+1; j<N; j++)
			{
				d[0] = p[j][0] - p[i][0];
				d[1] = p[j][1] - p[i][1];
				d[2] = p[j][2] - p[i][2];
				
				r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
				
				
				
				if(r < SEARCH_DISTANCE && i != j)
				{	
					f[i][0] -= (STOP_DISTANCE - r)*d[0]*K1/r;
					f[i][1] -= (STOP_DISTANCE - r)*d[1]*K1/r;
					f[i][2] -= (STOP_DISTANCE - r)*d[2]*K1/r;
					
					f[j][0] += (STOP_DISTANCE - r)*d[0]*K1/r;
					f[j][1] += (STOP_DISTANCE - r)*d[1]*K1/r;
					f[j][2] += (STOP_DISTANCE - r)*d[2]*K1/r;
				}
			}
		}
		
		// velocities
		for(i=0; i<N; i++)
		{
			
			for(j=0; j<3; j++)
			{
				vc[i][j] = (CENTER[j] - p[i][j]);
				vn[i][j] += (f[i][j] - DAMP*vn[i][j])*dt;
				//vc[i][j] = 0.0;
			}
			
			//magc = sqrt(vc[i][0]*vc[i][0] + vc[i][1]*vc[i][1] + vc[i][2]*vc[i][2]);
			//magn = sqrt(vn[i][0]*vn[i][0] + vn[i][1]*vn[i][1] + vn[i][2]*vn[i][2]);
			
			for(j=0; j<3; j++)
			{
				v[i][j] = vc[i][j] + vn[i][j];
				//v[i][j] *= 5.0;
				
				p[i][j] += v[i][j]*dt;
			}
			
			
			
		}
		
		if(tdraw == DRAW) 
		{
			draw_picture();
			tdraw = 0;
		}
		
		time += dt;
		//printf("%.4f\n", time);
		tdraw++;
		tprint++;
		

	}
	
}
	
void control()
{	
	int    tdraw = 0;
	double  time = 0.0;

	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	set_initial_conditions();
	
	draw_picture();
	
	n_body();
	
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	control();
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
	glutCreateWindow("Human Swarming I");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}	
	
	
	
	

