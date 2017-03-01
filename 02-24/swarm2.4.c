/*
David & Mikaela
Sync - Test Human swarm

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
#define STOP_DISTANCE 	0.1	// Distance at which particles should stop.
#define DT        	0.0001 	// Step size, global variable
#define DRAW 		10 	// Not sure?
#define EPSILON	     	0.001 	// Small number
#define PI		3.1415926
#define SEARCH_DISTANCE	0.3	//Only care about neighbors that are within this radius
				// This value seems really important.

#define N 10  // number of bodies


// Globals  //2-vectors for storing position & mass in x, y, and z directions.
double 	p[N][3],		// positions x,y,z
	v[N][3],		// sum velocity x,y,z
	vc[N][3],		// velocities x,y,z toward center
	vn[N][3],		// velocities x,y,z away from closest neighbor
	r[N][N],		// distance between particles
	CENTER[3],		// Center of Bodies
	SENSITIVITY[N];		// How sensitive a body is to its neighbors.		
int MIN_POS[N][N];// Minimum of distances. Column 1 is distance; Column 2 is index 0 to N-1

void set_initail_conditions() // Initial conditions
{	
	int i, j;
	
/*
	// Option 1: Particles are positioned in a circle
	for(i = 0; i<N; i++){
		p[i][0] = sin(i*2*PI/N)*1.5;
		p[i][1] = cos(i*2*PI/N)*1.0;
		p[i][2] = 0.0;
		}
	}
//*/
	
//*
	// Option 2: Random
	srand(time(NULL));
	for(i = 0; i<N; i++)
	{
		p[i][0] = (((double)rand()/(double)RAND_MAX)-0.5)*4.0;
		p[i][1] = (((double)rand()/(double)RAND_MAX)-0.5)*4.0;
		p[i][2] = 0.0;
	}
//*/
}


void sort()
{
	double d[3];
	int i, j, k;
	printf("Original Array:\n");
	for(i=0; i<N; i++)
	{
		printf("Row %d:    ", i);
		for(j=0; j<N; j++)
		{
			if (i != j)
			{
				d[0] = p[j][0] - p[i][0];
				d[1] = p[j][1] - p[i][1];
				d[2] = p[j][2] - p[i][2];
				r[i][j] = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
			}
			else 
			{
				r[i][j] = N*N;
			}
		
			if(r[i][j] < EPSILON)
			{
				r[i][j] = EPSILON;
			}
			printf("%.2f    ", r[i][j]);
		}
		printf("\n");
		
		for(j=0; j<N; j++)
		{
			MIN_POS[i][j] = j;
		}
	
	}

	for(i=0; i<N; i++)
	{
		double temp;
		int temp2;
		for(j=0; j<N; j++)
		{
			for(k=j+1; k<N; k++)
			{
				if( r[i][j] > r[i][k])
				{
					temp = r[i][j];
					r[i][j] = r[i][k];
					r[i][k] = temp;
					temp2 = MIN_POS[i][j];
					MIN_POS[i][j] = MIN_POS[i][k];
					MIN_POS[i][k] = temp2;
				}
			}
		}
	}


	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nSorted Array\n\n");
	for(i=0; i<N; i++)
	{
		printf("Row %d:    ", i);
		for(j=0; j<N; j++)
		{
			printf("%.2f    ", r[i][j]);
		}
		printf("\n");
	}

	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nPosition Array\n\n");
	for(i=0; i<N; i++)
	{
		printf("Row %d:    ", i);
		for(j=0; j<N; j++)
		{
			printf("%d    ", MIN_POS[i][j]);
		}
		printf("\n");
	}
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	int i;
	for(i=0;i<1;i++){
		glColor3d(0.65,0.16,0.16); 	//brown color
		SENSITIVITY[i]=100.0;
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.05,10,10);  //First argument affects size.
		glPopMatrix();
	}
	for(i=1;i<2;i++){
		glColor3d(1.0,0.5,1.0); 	//pink color
		SENSITIVITY[i]=0.005;
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.05,10,10);  //First argument affects size.
		glPopMatrix();
	}
	for(i=2;i<N;i++){
		glColor3d(1.0,1.0,0.5); 	//yellow color
		SENSITIVITY[i]=1.0;
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
	int i,j, k;
	double magc, magn;
	
	dt = DT;

	while(time < STOP_TIME)
	{
		// Find new center
		CENTER[0] = 0.0;
		CENTER[1] = 0.0;
		CENTER[2] = 0.0; 
		 
		for(j=0;j<N;j++)
		{
			CENTER[0] += 0.0;
			CENTER[1] += 0.0;
			CENTER[2] += 0.0; //pz[i];
		}
		CENTER[0] /= N;
		CENTER[1] /= N;
		CENTER[2] /= N;  // Find new center
		//printf("The center of masses is %.5f, %.5f, %.5f\n.", CENTER[0],CENTER[1],CENTER[2]);
				
		// Find distance between bodies
		sort();

		/*for(i=0; i<N; i++)
		{
			
			for(j=0; j<N; j++)
			{
				if (i != j)
				{
				d[0] = p[j][0] - p[i][0];
				d[1] = p[j][1] - p[i][1];
				d[2] = p[j][2] - p[i][2];
				r[i][j] = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
				}
				else 
				{
					r[i][j] = N*N;
				}
				
				if(r[i][j] < EPSILON)
				{
					r[i][j] = EPSILON;
				}
				
				//printf(" The distance between particle %d and particle %d is: %3f\n", i, j, r[i][j]);
			}
			
			//printf(" The position of particle %d is (%.2f, %.2f, %.2f)\n", i, p[i][0], p[i][1], p[i][2]);

		}
		
		// Finding minimum distance between each body
		for(i=0; i<N; i++){
			MINIMUM[i][0] = 100;
			for(j=0; j<N; j++)
			{
				if (MINIMUM[i][0] > r[i][j])
				{
				MINIMUM[i][0] = r[i][j];
				MINIMUM[i][1] = (double)j;
				}
			}
			
			//printf(" The smallest distance for particle %d is: %f\n", i, MINIMUM[i]);
		}*/
		//printf(" The smallest distance for particle 0 is: %f to particle %f\n", MINIMUM[0][0], MINIMUM[0][1]);
		


		//Get Velocities
		for(i=0; i<N; i++)
		{	
			for(j=0; j<3; j++)
			{	
				//Velocity Direction
				vc[i][j] = CENTER[j]-p[i][j];
				//vn[i][j] = -(p[(int)MINIMUM[i][1]][j] - p[i][j]); // OLD, only looking at nearest neighbor
				vn[i][j] = 0.0;
				for(k=0; k<N; k++)
				{
					if(r[i][k] < SEARCH_DISTANCE && i != k)
					{
						vn[i][j] += -((p[k][j]-p[i][j])/(r[i][k]*r[i][k]+EPSILON));
					}
				}
			}
			//printf(" The vc of particle %d is (%.2f, %.2f, %.2f)\n", i, vc[i][0], vc[i][1], vc[i][2]);
			//printf(" The vn of particle %d is (%.2f, %.2f, %.2f)\n", i, vn[i][0], vn[i][1], vn[i][2]);

			
			// Find magnitude
			magc = sqrt(vc[i][0]*vc[i][0] + vc[i][1]*vc[i][1] + vc[i][2]*vc[i][2] + EPSILON); // THESE EPSILONS FREAKIN' MATTER!!!
			magn = sqrt(vn[i][0]*vn[i][0] + vn[i][1]*vn[i][1] + vn[i][2]*vn[i][2] + EPSILON);
			
			// Normalize Velocities to Mag 1
			for(j=0; j<3; j++)
			{	
				vc[i][j] /= magc;
				vn[i][j] /= magn;
			}
			
			// Set all "Z" velocities to 0
			vc[i][2] = 0.0;
			vn[i][2] = 0.0;
			v[i][2]  = 0.0;
		}
		//printf("The x componetns of vc, vn, and v for the first particle are: %.5f, %.5f, %.5f\n", vc[i][0], vn[i][0], v[i][0]);
		
		// Make distance affect velocity
		for(i=0; i<N; i++)
		{
		//*
			//printf(" The vc of particle %d is (%.2f, %.2f, %.2f)\n", i, vc[i][0], vc[i][1], vc[i][2]);
			//printf(" The vn of particle %d is (%.2f, %.2f, %.2f)\n", i, vn[i][0], vn[i][1], vn[i][2]);
			for(j=0; j<3; j++)
			{
				vc[i][j]*=(30.0*r[i][0]); 		// magnitude of center part
				vn[i][j]*=(SENSITIVITY[i]/r[i][0]);	// magnitude of neighbor part
				v[i][j] = vc[i][j]+vn[i][j]; 	// combine two velocities 
			}
					
			// Set all "Z" velocities to 0
			vc[i][2] = 0.0;
			vn[i][2] = 0.0;
			v[i][2]  = 0.0;
			//printf(" The vc of particle %d is (%.2f, %.2f, %.2f)\n", i, vc[i][0], vc[i][1], vc[i][2]);
			//printf(" The vn of particle %d is (%.2f, %.2f, %.2f)\n", i, vn[i][0], vn[i][1], vn[i][2]);
			//printf(" The v  of particle %d is (%.2f, %.2f, %.2f)\n", i, v[i][0],  v[i][1],  v[i][2]);
		//*/
		}
		//printf("The x componetns of vc, vn, and v for the first particle are: %.5f, %.5f, %.5f\n", vc[i][0], vn[i][0], v[i][0]);
			
		//Move elements
		for(i=0; i<N; i++)
		{			
			p[i][0]=p[i][0]+v[i][0]*dt; //update position
			p[i][1]=p[i][1]+v[i][1]*dt;
			p[i][2]=p[i][2]+v[i][2]*dt;
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
