/*
David and Mikaela
SYNC FORCES 

Try new force calculation 

28 March 2017
*/

// gcc mj_forces2.c -o mj_forces2 -lglut -lm -lGLU -lGL && ./mj_forces2

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI		3.1415926535
#define DRAW 		10	// For draw_picture
#define XWindowSize 	1000 	// 700 initially 
#define YWindowSize 	1000 	// 700 initially

#define DT        	0.001 	// Time step
#define STOP_TIME	10.0 	// How long to go

#define SIGHT 		10.0 // How far the fish can 'see'
#define WA		1 // Attraction Weight Ratio
#define WD 		1 // Directional Weight Ratio
#define CA 		2 // Attraction Coefficient
#define CR		1 // Repulsion Coefficient

#define NFISH 		500 // Number of fish
#define NFOOD 		1 //Number of Targets
#define NPRED		1 //Number of Predators
#define N		NFISH + NFOOD + NPRED // Total number of particles

// Global Variables
double CENTER[3], // Center or point of attraction
	r;	  // Distance between particles
double 	TIMERUNNING = 0.0;
double 	SPEED = 20.0;

int 	FOODPOINTER = NFISH; // Our first food particle
int	PAUSE = 0;

struct body {
	double 	p[3];
	double 	vc[3];
	double 	vn[3];
	double 	v[3];
	double 	f[3];
	double 	radius;
	double 	color[3];
	double 	sensitivity;
	int    	type;
} particle[N];

void initialize_bodies()
{
	TIMERUNNING = 0.0;
	int i;

	// Initialize Fish
	for(i=0; i<NFISH; i++)
	{	
		/* Option to start in a CIRCLE
		particle[i].p[0] = sin(i*2*PI/NFISH)*2.0;
		particle[i].p[1] = cos(i*2*PI/NFISH)*2.0;
		particle[i].p[2] = 0.0;
		//*/
		
		//* Option to start in RANDOM positions
		particle[i].p[0] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		particle[i].p[1] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		particle[i].p[2] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		//*/
			
		// Body forces:
		particle[i].f[0] = 0.0;
		particle[i].f[1] = 0.0;
		particle[i].f[2] = 0.0;
		
		// Radius and Color
		particle[i].radius = 0.025; // default was 0.05
		particle[i].color[0] = 1.0; // Default is yellow
		particle[i].color[1] = 1.0;
		particle[i].color[2] = 0.5;

		// Sensitivity
		particle[i].sensitivity = 1.0;
		
		// Type
		particle[i].type = 1;
		
		printf("The starting position of particle %i is (%.4f, %.4f, %.4f)\n", 
			i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
	}

	//Initialize Food
	for(i=NFISH; i<NFISH+NFOOD; i++)
	{
		/* Option to start in a CIRCLE
		particle[i].p[0] = sin(i*2*PI/NFOOD);
		particle[i].p[1] = cos(i*2*PI/NFOOD);
		particle[i].p[2] = 0.0;
		//*/

		/* Option to start in RANDOM positions
		particle[i].p[0] = (((double)rand()/(double)RAND_MAX)-0.5)*1.0;
		particle[i].p[1] = (((double)rand()/(double)RAND_MAX)-0.5)*1.0;
		particle[i].p[2] = 0.0;//(((double)rand()/(double)RAND_MAX)-0.5)*3.0;
		//*/

		//* Option to start at 0
		particle[i].p[0] = 0.0;
		particle[i].p[1] = 0.0;
		particle[i].p[2] = 0.0;
		//*/
	
		// Target Radius and Color
		particle[i].radius = 0.05; // default was 0.05
		particle[i].color[0] = 0.0;
		particle[i].color[1] = 0.0;
		particle[i].color[2] = 1.0;
	
		printf("The starting position of particle %i (target) is (%.4f, %.4f, %.4f)\n", 
			i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
	}


	// Initialize Predators
	if(NPRED>=1)
	{
		for(i=NFISH+NFOOD; i<NFISH+NFOOD+NPRED; i++)
		{
			//*Option to start in a CIRCLE
			particle[i].p[0] = sin(i*2*PI/NPRED);
			particle[i].p[1] = cos(i*2*PI/NPRED);
			particle[i].p[2] = 0.0;
			//*/
	
			/* Option to start in RANDOM positions
			particle[i].p[0] = (((double)rand()/(double)RAND_MAX)-0.5)*1.0;
			particle[i].p[1] = (((double)rand()/(double)RAND_MAX)-0.5)*1.0;
			particle[i].p[2] = 0.0; //(((double)rand()/(double)RAND_MAX)-0.5)*3.0;
			//*/
		
			// Predator Starting Velocities
			//particle[i].vc[0] = 0.0; //ignore
			//particle[i].vc[1] = 0.0; //ignore
			//particle[i].vc[2] = 0.0; //ignore
			//particle[i].vn[0] = 0.0; //ignore
			//particle[i].vn[1] = 0.0; //ignore
			//particle[i].vn[2] = 0.0; //ignore
			 particle[i].v[0] = 10.0*cos(particle[i].p[0]); 
			 particle[i].v[1] = 10.0*cos(particle[i].p[1]);
			 particle[i].v[2] = 0.0;
		
			// Predator forces:
			//particle[i].f[0] = 0.0;
			//particle[i].f[1] = 0.0;
			//particle[i].f[2] = 0.0;
		
			// Predator Radius and Color
			particle[i].radius = 0.05; // default was 0.05
			particle[i].color[0] = 1.0;
			particle[i].color[1] = 0.5;
			particle[i].color[2] = 1.0;
	
			printf("The starting position of particle %i (predator) is (%.4f, %.4f, %.4f)\n", 
				i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
		}	
	}
	
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);

	int i;
	
	for(i=0;i<N;i++){
		glColor3d(particle[i].color[0], particle[i].color[1], particle[i].color[2]); 	//Object color
		glPushMatrix();
		glTranslatef(particle[i].p[0], particle[i].p[1], particle[i].p[2]);
		glutSolidSphere(particle[i].radius, 10, 10);  //First argument affects size.
		glPopMatrix();
	}
	glutSwapBuffers();
}

int n_body()
{
	// Variables
	double d[3], dt, r2, r4, force_mag;
	int tdraw = 0;
	int tprint = 0;
	int i, j, k;
	int danger_d;
	dt = DT;

	// Move Predators in a circle
	for(i=NFISH+NFOOD; i<NFISH+NFOOD+NPRED; i++)
	{
		particle[i].f[0] = 0.0 - particle[i].p[0];	
		particle[i].f[1] = 0.0 - particle[i].p[1];
		particle[i].f[2] = 0.0 - particle[i].p[2];
		
		particle[i].v[0] += particle[i].f[0];
		particle[i].v[1] += particle[i].f[1];
		particle[i].v[2] += particle[i].f[2];		
		
		particle[i].p[0] += particle[i].v[0]*dt; 
		particle[i].p[1] += particle[i].v[1]*dt;
		particle[i].p[2] += particle[i].v[2]*dt; 
	}

	// Reset Forces
	for(i=0; i<NFISH; i++)
	{
		particle[i].f[0] = 0.0;
		particle[i].f[1] = 0.0;
		particle[i].f[2] = 0.0;
	}

	// Calculate forces
	for(i=0; i<NFISH; i++)
	{
		for(j=i+1; j<NFISH; j++)
		{
			d[0] = particle[i].p[0] - particle[j].p[0];
			d[1] = particle[i].p[1] - particle[j].p[1];
			d[2] = particle[i].p[2] - particle[j].p[2];
			//printf("The distance between %d and %d in the x direction is %lf\n The distance between %d and %d in the y direction is %lf\n The distance between %d and %d in the z direction is %lf\n", i, j, d[0], i, j, d[1], i, j, d[2]);

			r2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
			r = sqrt(r2);
			r4 = r2*r2;
			//printf("The Euclidean distance between %d and %d is %lf\n", i, j, r);

			if(r < SIGHT && i != j)
			{
				particle[i].f[0] += WA*(CA*(-d[0]/r2) - CR*(-d[0]/r4)) + WD*(particle[j].v[0]/r);
				particle[i].f[1] += WA*(CA*(-d[1]/r2) - CR*(-d[1]/r4)) + WD*(particle[j].v[1]/r);
				particle[i].f[2] += WA*(CA*(-d[2]/r2) - CR*(-d[2]/r4)) + WD*(particle[j].v[2]/r);

				particle[j].f[0] += WA*(CA*(d[0]/r2) - CR*(d[0]/r4)) + WD*(particle[i].v[0]/r);
				particle[j].f[1] += WA*(CA*(d[1]/r2) - CR*(d[1]/r4)) + WD*(particle[i].v[1]/r);
				particle[j].f[2] += WA*(CA*(d[2]/r2) - CR*(d[2]/r4)) + WD*(particle[i].v[2]/r);

			}

			//printf("The current force vector for particle %d is: (%lf, %lf, %lf)\n", i, particle[i].f[0], particle[i].f[1], particle[i].f[0]);
		}
	}

	// Update Velocities and Move Fish
	for(i=0; i<NFISH; i++)
	{	
		
		// Normalize the forces
		force_mag = sqrt(particle[i].f[0]*particle[i].f[0] + particle[i].f[1]*particle[i].f[1] + particle[i].f[2]*particle[i].f[2]); 	
		particle[i].f[0] /= force_mag;
		particle[i].f[1] /= force_mag;
		particle[i].f[2] /= force_mag;

		//  Move the each fish towards the target
		particle[i].vc[0] = particle[i].p[0] - particle[NFISH].p[0];
		particle[i].vc[1] = particle[i].p[1] - particle[NFISH].p[1];
		particle[i].vc[2] = particle[i].p[2] - particle[NFISH].p[2];

		/*particle[i].vn[0] = (9*particle[i].vn[0] + particle[i].f[0]*dt)/10;
		particle[i].vn[1] = (9*particle[i].vn[1] + particle[i].f[1]*dt)/10;
		particle[i].vn[2] = (9*particle[i].vn[2] + particle[i].f[2]*dt)/10;
		*/
		
		particle[i].vn[0] += particle[i].f[0]*dt;
		particle[i].vn[1] += particle[i].f[1]*dt;
		particle[i].vn[2] += particle[i].f[2]*dt;	

		particle[i].v[0] = particle[i].vn[0];// + 5.0*particle[i].vc[0];
		particle[i].v[1] = particle[i].vn[1];// + 5.0*particle[i].vc[1];
		particle[i].v[2] = particle[i].vn[2];// + 5.0*particle[i].vc[2];
		
		particle[i].v[0] *= SPEED;
		particle[i].v[1] *= SPEED;
		particle[i].v[2] *= SPEED;
		
		particle[i].p[0] += particle[i].v[0]*dt;
		particle[i].p[1] += particle[i].v[1]*dt;
		particle[i].p[2] += particle[i].v[2]*dt;

		//particle[i].p[0] *= 0.5;
		//particle[i].p[0] *= 0.5;
		//particle[i].p[0] *= 0.5;
	
		// Diagnostics
		printf("The position of particle %i is (%.4f, %.4f, %.4f)\n", 
				i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
		//printf("The velocity of particle %i is (%.4f, %.4f, %.4f)\n", 
				//i, particle[i].v[0], particle[i].v[1], particle[i].v[2]);
		//printf("The forces on particle %i are (%.4f, %.4f, %.4f)\n", 
				//i, particle[i].f[0], particle[i].f[1], particle[i].f[2]);
		printf("\n");
		
	}

	TIMERUNNING += dt;
	printf("%.4f\n", TIMERUNNING);
	tdraw++;
	tprint++;
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
	switch(key) 
	{
		case 'q': 
			exit(1);

		case ' ': 
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
		if( state == GLUT_DOWN ) // when left mouse button goes down.
		{
			//printf("FOODPOINTER is %i \n", FOODPOINTER);
			coord[0] = (x*4.0/XWindowSize)-2.0;
			coord[1] = -(y*4.0/YWindowSize)+2.0;
			coord[2] = 0.0;
			printf("The food is at (%.4f, %.4f, %.4f)\n",
				coord[0], coord[1], coord[2]);
			particle[FOODPOINTER].p[0] = coord[0];
			particle[FOODPOINTER].p[1] = coord[1];
			particle[FOODPOINTER].p[2] = coord[2];
			
			// Change pointer to next food particle
			FOODPOINTER++;
			if(FOODPOINTER == NFISH+NFOOD)
			{
				FOODPOINTER = NFISH;
			}			
		}
	}
}

	
void update(int value)
{
	if(TIMERUNNING < STOP_TIME){
		if(PAUSE == 0)
		{
			n_body();
		}	
	}
    	glutSpecialFunc( arrowFunc );
	glutKeyboardFunc( keyboardFunc );
	glutMouseFunc( mouseFunc );
	glutPostRedisplay();

	glutTimerFunc(1, update, 0);
	
}


void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	draw_picture();
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
	glutCreateWindow("Swarming");
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

	initialize_bodies();
	gluLookAt(0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glutDisplayFunc(Display);
	glutTimerFunc(16, update, 0);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}

