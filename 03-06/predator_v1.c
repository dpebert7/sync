/*
David & Mikaela
Sync - Test Human swarm

Fix variable names (esp t emp_f) and introduce predator
1 March 2017
*/

// gcc predator_v1.c -o temp -lglut -lm -lGLU -lGL && ./temp

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DRAW 		10	// Not sure?
#define XWindowSize 	700 	//700 initially 
#define YWindowSize 	700 	//700 initially

//#define STOP_TIME 	0.0005 	// Go few steps before stopping.  
#define STOP_TIME 	10.0	// Go lots of steps
#define STOP_DISTANCE 	0.5	// Distance at which particles should stop.
#define DT        	0.001 	// Step size, global variable

#define SEARCH_DISTANCE	10.0	// Only care about neighbors that are within this radius
				// This value seems really important.

//NAMING TEMP FILE #defines			
#define DAMP 	300
#define K1 	10
#define N 	20  // number of bodies

// Global variable
double 	CENTER[3],	// Center or point of attraction;
	//r[N][N],	// distance between particles; 
	r;

double TIMERUNNING = 0.0;
double SPEED = 1.0;
	
struct body {
	double p[3];
	double vc[3];
	double vn[3];
	double v[3];
	double f[3];
	double radius;
	double color[3];
	double sensitivity;
	int    type;
} particle[N+1]; 




void initialize_bodies()
{
	TIMERUNNING = 0.0;
	int i;
	for(i=0; i<N; i++)
	{	
		/* Option to start in a CIRCLE
		particle[i].p[0] = sin(i*2*PI/N)*4.5;
		particle[i].p[1] = cos(i*2*PI/N)*4.5;
		particle[i].p[2] = 0.0;
		//*/
		
		//* Option to start in RANDOM positions
		particle[i].p[0] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		particle[i].p[1] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		particle[i].p[2] = (((double)rand()/(double)RAND_MAX)-0.5)*3.0;
		//*/
		
		// Initial Velocities
		particle[i].vc[0] = 0.0;
		particle[i].vc[1] = 0.0;
		particle[i].vc[2] = 0.0;
		particle[i].vn[0] = 0.0;
		particle[i].vn[1] = 0.0;
		particle[i].vn[2] = 0.0;
		 particle[i].v[0] = 0.0;
		 particle[i].v[1] = 0.0;
		 particle[i].v[2] = 0.0;
		
		// Body forces:
		particle[i].f[0] = 0.0;
		particle[i].f[1] = 0.0;
		particle[i].f[2] = 0.0;
		
		// Radius and Color
		particle[i].radius = 0.025; // default was 0.05
		particle[i].color[0] = 1.0; // Default is yellow
		particle[i].color[1] = 1.0;
		particle[i].color[2] = 0.5;
		
		// Brown color:  (0.65,0.16,0.16)
		// Pink color:   (1.0,0.5,1.0)
		// Yellow color: (1.0,1.0,0.5)
		
		// Sensitivity
		particle[i].sensitivity = 1.0;
		
		// Type
		particle[i].type = 1;
		
		//printf("The starting position of particle %i is (%.4f, %.4f, %.4f)\n", 
		//	i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
	}
	//printf("\n\n");
	
	// Create Target
	particle[N].p[0] = 0.0;
	particle[N].p[1] = 0.0;
	particle[N].p[2] = 0.0;
	
	// Target Starting Velocities
	particle[N].vc[0] = 0.0; //ignore
	particle[N].vc[1] = 0.0; //ignore
	particle[N].vc[2] = 0.0; //ignore
	particle[N].vn[0] = 0.0; //ignore
	particle[N].vn[1] = 0.0; //ignore
	particle[N].vn[2] = 0.0; //ignore
	 particle[N].v[0] = 0.0; 
	 particle[N].v[1] = 0.0;
	 particle[N].v[2] = 0.0;
	
	// Target forces:
	particle[N].f[0] = 0.0;
	particle[N].f[1] = 0.0;
	particle[N].f[2] = 0.0;
	
	// Target Radius and Color
	particle[N].radius = 0.05; // default was 0.05
	particle[N].color[0] = 0.0;
	particle[N].color[1] = 0.0;
	particle[N].color[2] = 1.0;
	
	printf("The starting position of particle %i (target) is (%.4f, %.4f, %.4f)\n", 
		N, particle[N].p[0], particle[N].p[1], particle[N].p[2]);



	// Create Predator
	particle[N+1].p[0] = 1.0;
	particle[N+1].p[1] = 0.0;
	particle[N+1].p[2] = 0.0;
	
	// Predator Starting Velocities
	particle[N+1].vc[0] = 0.0; //ignore
	particle[N+1].vc[1] = 0.0; //ignore
	particle[N+1].vc[2] = 0.0; //ignore
	particle[N+1].vn[0] = 0.0; //ignore
	particle[N+1].vn[1] = 0.0; //ignore
	particle[N+1].vn[2] = 0.0; //ignore
	 particle[N+1].v[0] = 0.0; 
	 particle[N+1].v[1] = 0.0;
	 particle[N+1].v[2] = 0.0;
	
	// Predator forces:
	particle[N+1].f[0] = 0.0;
	particle[N+1].f[1] = 0.0;
	particle[N+1].f[2] = 0.0;
	
	// Predator Radius and Color
	particle[N+1].radius = 0.05; // default was 0.05
	particle[N+1].color[0] = 1.0;
	particle[N+1].color[1] = 0.5;
	particle[N+1].color[2] = 1.0;
	
	printf("The starting position of particle %i (predator) is (%.4f, %.4f, %.4f)\n", 
		N+1, particle[N+1].p[0], particle[N+1].p[1], particle[N+1].p[2]);

}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	int i;
	
	for(i=0;i<(N+2);i++){
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
	// VARIABLES
	double d[3], dt;
	int tdraw = 0;
	int tprint = 0;
	int i, j; // k;	
	dt = DT;
	
	

	// MOVE PREDATOR
	//particle[N].f[0] = 0.0;
	//particle[N].f[1] = 0.0;
	//particle[N].f[2] = 0.0;
	
	//particle[N].v[0] = 0.0;
	//particle[N].v[1] = 10.0;
	//particle[N].v[2] = 0.0;	

	//particle[N].p[0] += particle[N].v[0]*dt;
	//particle[N].p[1] += particle[N].v[1]*dt;
	//particle[N].p[2] += particle[N].v[2]*dt;

	particle[N+1].p[0] = sin(10.0*TIMERUNNING);
	particle[N+1].p[1] = cos(10.0*TIMERUNNING);
	particle[N+1].p[2] = 0.0;

	//printf("The current position of particle %i is (%.4f, %.4f, %.4f)\n", 
	//	N, particle[N].p[0], particle[N].p[1], particle[N].p[2]);
	
	
	
	// MOVE BODIES

	/*
	printf("The current position of particle 0 is (%.4f, %.4f, %.4f)\n", 
		particle[0].p[0], particle[0].p[1], particle[0].p[2]);
	printf("The current position of particle 1 is (%.4f, %.4f, %.4f)\n", 
		particle[1].p[0], particle[1].p[1], particle[1].p[2]);
	printf("The current position of particle 2 is (%.4f, %.4f, %.4f)\n", 
		particle[2].p[0], particle[2].p[1], particle[2].p[2]);
	printf("The current position of particle 3 is (%.4f, %.4f, %.4f)\n", 
		particle[3].p[0], particle[3].p[1], particle[3].p[2]);
	*/

				
	for(i=0; i<N; i++)
	{
		// reset the forces to 0.0 
		particle[i].f[0] = 0.0;
		particle[i].f[1] = 0.0;
		particle[i].f[2] = 0.0;
	}
	
	for(i=0; i<N; i++)
	{
		for(j=i+1; j<N; j++)
		{
			d[0] = particle[j].p[0] - particle[i].p[0];
			d[1] = particle[j].p[1] - particle[i].p[1];
			d[2] = particle[j].p[2] - particle[i].p[2];

			//printf("force on i is %.4f \n", particle[i].f[0]);
			//printf("force on j is %.4f \n", particle[j].f[0]);
			//printf("position i is %.4f \n", particle[i].p[0]);
			//printf("position j is %.4f \n", particle[j].p[0]);
			//printf("d is %.4f \n", d[0]);
			
			r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
			//printf("r is %.4f \n", r);
			
			if(r < SEARCH_DISTANCE && i != j)
			{	
				particle[i].f[0] -= (STOP_DISTANCE - r)*d[0]*K1/r;
				particle[i].f[1] -= (STOP_DISTANCE - r)*d[1]*K1/r;
				particle[i].f[2] -= (STOP_DISTANCE - r)*d[2]*K1/r;
				
				particle[j].f[0] += (STOP_DISTANCE - r)*d[0]*K1/r;
				particle[j].f[1] += (STOP_DISTANCE - r)*d[1]*K1/r;
				particle[j].f[2] += (STOP_DISTANCE - r)*d[2]*K1/r;
				
				//printf("force is %.4f \n", particle[i].f[0]);
			}
		}
	}
	
	// Update velocities and move particles.
	for(i=0; i<N; i++)
	{			
		for(j=0; j<3; j++)
		{
			//vc[i][j] = (CENTER[j] - p[i][j]);
			//vn[i][j] += (f[i][j] - DAMP*vn[i][j])*dt;
			
			particle[i].vc[j] = (particle[N].p[j] - particle[i].p[j]);
			particle[i].vn[j] += (particle[i].f[j] - DAMP*particle[i].vn[j]*dt);
		}
		
		for(j=0; j<3; j++)
		{
			//v[i][j] = vc[i][j] + vn[i][j];
			//v[i][j] *= 5.0;
			//p[i][j] += v[i][j]*dt;
			
			particle[i].v[j] = 5.0*particle[i].vc[j] + particle[i].vn[j];
			particle[i].v[j] *= SPEED;
			particle[i].p[j] += particle[i].v[j]*dt;
		}
		particle[i].p[2] = 0.0;
		
		// Diagnostics
		
		//printf("The position of particle %i is (%.4f, %.4f, %.4f)\n", 
		//		i, particle[i].p[0], particle[i].p[1], particle[i].p[2]);
				
		//printf("The velocity of particle %i is (%.4f, %.4f, %.4f)\n", 
		//		i, particle[i].v[0], particle[i].v[1], particle[i].v[2]);

		//printf("The forces on particle %i are (%.4f, %.4f, %.4f)\n", 
		//		i, particle[i].f[0], particle[i].f[1], particle[i].f[2]);
		
		//printf("\n");
		
	}
	

	TIMERUNNING += dt;
	//printf("%.4f\n", time);
	tdraw++;
	tprint++;


}


/*void control()
{	
	int    tdraw = 0;
	double  time = 0.0;


}*/


void arrowFunc(int key, int x, int y) 
{
	switch (key) 
	{    
		//case 100 :
		//	printf("Left arrow");
		//	; break;
		//case 102 :
		//	printf("Right arrow");  	
		//	; break;
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
		case 'q': exit(1);
		default:
			break;
	}
}


void mouseFunc( int button, int state, int x, int y )
{
	double coord[3];
	if( button == GLUT_LEFT_BUTTON ) 
	{
		if( state == GLUT_DOWN ) // when left mouse button goes down.
		{
			coord[0] = (x*4.0/XWindowSize)-2.0;
			coord[2] = 0.0;
			coord[1] = -(y*4.0/YWindowSize)+2.0;
			printf("The sphere is at (%.4f, %.4f, %.4f)\n",
				coord[0], coord[1], coord[2]);
			particle[N].p[0] = coord[0];
			particle[N].p[1] = coord[1];
			particle[N].p[2] = coord[2];			
		}
	}
}

	
void update(int value)
{
	n_body();

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

	initialize_bodies();
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glutDisplayFunc(Display);
	glutTimerFunc(16, update, 0);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}	
	
	
	
	

