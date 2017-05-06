/*
David & Mikaela
Sync - Test Human swarm

Output:
N humans walk toward center of circle.

*/

// gcc swarm2.c -o temp -lglut -lm -lGLU -lGL
// ./temp
// To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.141592654

#define XWindowSize 700 
#define YWindowSize 700

#define STOP_TIME 	100.0  
#define STOP_DISTANCE 	0.1	//Distance at which particles should stop.
#define DT        	0.0001 	//Step size, global variable
#define DRAW 10 		//Not sure?

#define N 100  // number of bodies


// Globals  //2-vectors for storing position & mass in x, y, and z directions.
double 	p[N][3],		// positions x,y,z
	v[N][3],		// velocities x,y,z
	r[N][N],		// distance between particles
	CENTER[3],		// Center of Bodies
	MINIMUM[N][2];		// Minimum of distances

void set_initail_conditions() // Initial conditions
{	
	int i, j, x, y;
	
	
	// Option 1: Particles are positioned in a circle.
	/*
	for(i = 0; i<N; i++){
		p[i][0] = sin(i*2*PI/N);
		p[i][1] = cos(i*2*PI/N);
		p[i][2] = 0.0;
		for(j=0; j<N; j++){
			v[i][j] = 0.0;
		}
	}
	*/
	

	// Option 2: Random;
	srand(time(NULL));
	for(i = 0; i<N; i++){
		p[i][0] = (((double)rand()/(double)RAND_MAX) - 0.5)*4;
		p[i][1] = (((double)rand()/(double)RAND_MAX) - 0.5)*4;
		p[i][2] = 0.0;
		for(j=0; j<N; j++){
			v[i][j] = 0.0;
		}
	}

}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	int i;
	for(i=0;i<N;i++){
		glColor3d(1.0,1.0,0.5); //yellow color
		//glColor3d(1.0,0.5,1.0); //pink color
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.05,10,10);  //First argument affects size.
		glPopMatrix();
	}
	glutSwapBuffers();
}


int n_body()
{
	double d[3];
	double dt;
	int    tdraw = 0; int   tprint = 0;
	double  time = 0.0;
	int i,j;
	double mag;
	
	dt = DT;

	while(time < STOP_TIME)
	{
		// Find new center
		CENTER[0] = 0.0;
		CENTER[1] = 0.0;
		CENTER[2] = 0.0; 
		 
		for(j=0;j<N;j++){
			CENTER[0] += p[j][0];
			CENTER[1] += p[j][1];
			CENTER[2] += 0.0; //pz[i];
		}
		CENTER[0] /= N;
		CENTER[1] /= N;
		CENTER[2] /= N;  // Find new center
		//printf("The center of masses is %.5f, %.5f, %.5f\n.", CENTER[0],CENTER[1],CENTER[2]);
				
		// Find distance between bodies
		for(i=0; i<N; i++)
		{
			
			for(j=0; j<N; j++){
				if ( i != j){
				d[0] = p[j][0] - p[i][0];
				d[1] = p[j][1] - p[i][1];
				d[2] = p[j][2] - p[i][2];
				r[i][j] = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
				}
				else {
					r[i][j] = N*N;
				}
			//printf(" The distance between  particle %d and particle %d is: %f\n", i, j, r[i][j]);
			}

		}
		
		// Finding minimum distance between each body
		for(i=0; i<N; i++){
			MINIMUM[i][0] = 100;
			for(j=0; j<N; j++){
				if (MINIMUM[i][0] > r[i][j]){
				MINIMUM[i][0] = r[i][j];
				MINIMUM[i][1] = (double)j;
				}
			}
			
			//printf(" The smallest distance for particle %d is: %f\n", i, MINIMUM[i]);
		}
		printf(" The smallest distance for particle 0 is: %f to particle %f\n", MINIMUM[0][0], MINIMUM[0][1]);
		


		//Get Velocities
		for(i=0; i<N; i++)
		{	
			for(j=0; j<3; j++){	
				//Velocity Direction
				v[i][j] = CENTER[j]-p[i][j];
			}
			

			
			// Find magnitude
			mag = sqrt(v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
			
			// Normalize Velocities to Mag 1
			for(j=0; j<3; j++){	
				v[i][j] /= mag;
			}
			
			// Set all "Z" velocities to 0
			v[i][2] = 0.0;

		}
		
		// Making distance affect velocity
		for(i=0; i<N; i++){
			if(MINIMUM[i][0] < STOP_DISTANCE){
				for(j=0;j<3;j++){
					v[i][j] *= -1;
				}
			}

			if(MINIMUM[i][0] < 2*STOP_DISTANCE){
				for(j=0; j<3; j++){
					int ind_nearest = (int)MINIMUM[i][1];
					v[i][j] = -(p[ind_nearest][j] - p[i][j]);
				}
			}
			
			if(MINIMUM[i][0] > STOP_DISTANCE){
				for(j=0;j<3;j++){
					v[i][j] *= MINIMUM[i][0]*10;
				}
			}
			
		}

		//Move elements
		for(i=0; i<N; i++)
		{			
			p[i][0]=p[i][0]+v[i][0]*dt; //update position
			p[i][1]=p[i][1]+v[i][1]*dt;
			p[i][2]=p[i][2]+v[i][2]*dt;

			//printf("The velocities are %.5f, %.5f, %.5f\n.", v[i][0],v[i][1],v[i][2]);
			
		}

		if(tdraw == DRAW) 
		{
			draw_picture();
			tdraw = 0;
		}

		time += dt;
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

	set_initail_conditions();
	
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
