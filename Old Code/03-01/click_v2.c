/*
David & Mikaela
Sync - Test Human swarm

Use point and click
1 March 2017
*/

// gcc click_v2.c -o temp -lglut -lm -lGLU -lGL & ./temp
// ./temp
// To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define XWindowSize 700 //700 initially 
#define YWindowSize 700 //700 initially

//#define STOP_TIME 	0.0005 	// Go few steps before stopping.  
#define STOP_TIME 	10.0	// Go lots of steps
#define STOP_DISTANCE 	0.5	// Distance at which particles should stop.
#define DT        	0.001 	// Step size, global variable
#define DRAW 		10 	// Not sure?
#define EPSILON	     	0.001 	// Small number
#define PI		3.1415926
#define SEARCH_DISTANCE	10	//Only care about neighbors that are within this radius
				// This value seems really important.

//NAMING TEMP FILE #defines			
#define DAMP 	200
#define K1 	10
#define N 	20  // number of bodies

// Global variable
double 	CENTER[3],	// Center or point of attraction;
	//r[N][N],	// distance between particles; 
	r;

double TIMERUNNING = 0.0;
double SPEED = 1.0;
	
struct Fish {
	double p[3];
	double vc[3];
	double vn[3];
	double v[3];
	double f[3];
	double radius;
	double color[3];
	double sensitivity;
	int    type;
} temp_f[N+1]; 

//Fish *temp_f;					//pointer

//Fish *temp_f = (Fish*)malloc(sizeof(Fish)*N);




void create_fish()
{
	TIMERUNNING = 0.0;
	int i;
	for(i=0; i<N; i++)
	{	
		//temp_f = (Fish*)malloc(N*sizeof(Fish));		//allocate memory
		
		/* Option to start in a CIRCLE
		temp_f[i].p[0] = sin(i*2*PI/N)*4.5;
		temp_f[i].p[1] = cos(i*2*PI/N)*4.5;
		temp_f[i].p[2] = 0.0;
		//*/
		
		//* Option to start in RANDOM positions
		temp_f[i].p[0] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		temp_f[i].p[1] = (((double)rand()/(double)RAND_MAX)-0.5)*10.0;
		temp_f[i].p[2] = (((double)rand()/(double)RAND_MAX)-0.5)*3.0;
		//*/
		
		// Initial Velocities
		temp_f[i].vc[0] = 0.0;
		temp_f[i].vc[1] = 0.0;
		temp_f[i].vc[2] = 0.0;
		temp_f[i].vn[0] = 0.0;
		temp_f[i].vn[1] = 0.0;
		temp_f[i].vn[2] = 0.0;
		 temp_f[i].v[0] = 0.0;
		 temp_f[i].v[1] = 0.0;
		 temp_f[i].v[2] = 0.0;
		
		// Fish forces:
		temp_f[i].f[0] = 0.0;
		temp_f[i].f[1] = 0.0;
		temp_f[i].f[2] = 0.0;
		
		// Radius and Color
		temp_f[i].radius = 0.025; // default was 0.05
		temp_f[i].color[0] = 1.0; // Default is yellow
		temp_f[i].color[1] = 1.0;
		temp_f[i].color[2] = 0.5;
		
		// Brown color:  (0.65,0.16,0.16)
		// Pink color:   (1.0,0.5,1.0)
		// Yellow color: (1.0,1.0,0.5)
		
		// Sensitivity
		temp_f[i].sensitivity = 1.0;
		
		// Type
		temp_f[i].type = 1;
		
		//printf("The starting position of particle %i is (%.4f, %.4f, %.4f)\n", 
		//	i, temp_f[i].p[0], temp_f[i].p[1], temp_f[i].p[2]);
	}
	//printf("\n\n");
	
	// Create Target
	temp_f[N].p[0] = 0.0;
	temp_f[N].p[1] = 0.0;
	temp_f[N].p[2] = 0.0;
	
	// Target Starting Velocities
	temp_f[N].vc[0] = 0.0; //ignore
	temp_f[N].vc[1] = 0.0; //ignore
	temp_f[N].vc[2] = 0.0; //ignore
	temp_f[N].vn[0] = 0.0; //ignore
	temp_f[N].vn[1] = 0.0; //ignore
	temp_f[N].vn[2] = 0.0; //ignore
	 temp_f[N].v[0] = 0.0; 
	 temp_f[N].v[1] = 0.0;
	 temp_f[N].v[2] = 0.0;
	
	// Target forces:
	temp_f[N].f[0] = 0.0;
	temp_f[N].f[1] = 0.0;
	temp_f[N].f[2] = 0.0;
	
	// Target Radius and Color
	temp_f[N].radius = 0.05; // default was 0.05
	temp_f[N].color[0] = 1.0;
	temp_f[N].color[1] = 0.5;
	temp_f[N].color[2] = 1.0;
	
	printf("The starting position of particle %i is (%.4f, %.4f, %.4f)\n", 
		N, temp_f[N].p[0], temp_f[N].p[1], temp_f[N].p[2]);

}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	int i;
	
	for(i=0;i<(N+1);i++){
		glColor3d(temp_f[i].color[0], temp_f[i].color[1], temp_f[i].color[2]); 	//Object color
		glPushMatrix();
		glTranslatef(temp_f[i].p[0], temp_f[i].p[1], temp_f[i].p[2]);
		glutSolidSphere(temp_f[i].radius, 10, 10);  //First argument affects size.
		glPopMatrix();
	}
	glutSwapBuffers();
}

int n_body()
{	
	// VARIABLES
	double d[3], dt;
	//double magc, magn, fmag, dist;
	int tdraw = 0;
	int tprint = 0;
	int i, j; // k;	
	dt = DT;
	
	

	// MOVE TARGET
	//temp_f[N].f[0] = 0.0;
	//temp_f[N].f[1] = 0.0;
	//temp_f[N].f[2] = 0.0;
	
	//temp_f[N].v[0] = 0.0;
	//temp_f[N].v[1] = 10.0;
	//temp_f[N].v[2] = 0.0;	

	//temp_f[N].p[0] += temp_f[N].v[0]*dt;
	//temp_f[N].p[1] += temp_f[N].v[1]*dt;
	//temp_f[N].p[2] += temp_f[N].v[2]*dt;

	//temp_f[N].p[0] = sin(10.0*TIMERUNNING);
	//temp_f[N].p[1] = cos(10.0*TIMERUNNING);
	//temp_f[N].p[2] = 0.0;

	//printf("The current position of particle %i is (%.4f, %.4f, %.4f)\n", 
	//	N, temp_f[N].p[0], temp_f[N].p[1], temp_f[N].p[2]);
	
	
	
	// MOVE FISH
	/*
	printf("The current position of particle 0 is (%.4f, %.4f, %.4f)\n", 
		temp_f[0].p[0], temp_f[0].p[1], temp_f[0].p[2]);
	printf("The current position of particle 1 is (%.4f, %.4f, %.4f)\n", 
		temp_f[1].p[0], temp_f[1].p[1], temp_f[1].p[2]);
	printf("The current position of particle 2 is (%.4f, %.4f, %.4f)\n", 
		temp_f[2].p[0], temp_f[2].p[1], temp_f[2].p[2]);
	printf("The current position of particle 3 is (%.4f, %.4f, %.4f)\n", 
		temp_f[3].p[0], temp_f[3].p[1], temp_f[3].p[2]);
	*/

				
	for(i=0; i<N; i++)
	{
		// reset the forces to 0.0 
		temp_f[i].f[0] = 0.0;
		temp_f[i].f[1] = 0.0;
		temp_f[i].f[2] = 0.0;
	}
	
	for(i=0; i<N; i++)
	{
		for(j=i+1; j<N; j++)
		{

			//d[0] = p[j][0] - p[i][0];
			//d[1] = p[j][1] - p[i][1];
			//d[2] = p[j][2] - p[i][2];
			d[0] = temp_f[j].p[0] - temp_f[i].p[0];
			d[1] = temp_f[j].p[1] - temp_f[i].p[1];
			d[2] = temp_f[j].p[2] - temp_f[i].p[2];

			//printf("force on i is %.4f \n", temp_f[i].f[0]);
			//printf("force on j is %.4f \n", temp_f[j].f[0]);
			//printf("position i is %.4f \n", temp_f[i].p[0]);
			//printf("position j is %.4f \n", temp_f[j].p[0]);
			//printf("d is %.4f \n", d[0]);
			
			r = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
			//printf("r is %.4f \n", r);
			
			if(r < SEARCH_DISTANCE && i != j)
			//if(i != j)
			{	
				temp_f[i].f[0] -= (STOP_DISTANCE - r)*d[0]*K1/r;
				temp_f[i].f[1] -= (STOP_DISTANCE - r)*d[1]*K1/r;
				temp_f[i].f[2] -= (STOP_DISTANCE - r)*d[2]*K1/r;
				
				temp_f[j].f[0] += (STOP_DISTANCE - r)*d[0]*K1/r;
				temp_f[j].f[1] += (STOP_DISTANCE - r)*d[1]*K1/r;
				temp_f[j].f[2] += (STOP_DISTANCE - r)*d[2]*K1/r;
				
				//printf("force is %.4f \n", temp_f[i].f[0]);
				// With bug, the search distance is exceeded quickly!
				// But that's not the fundamental problem.
				
				//f[i][0] -= (STOP_DISTANCE - r)*d[0]*K1/r;
				//f[i][1] -= (STOP_DISTANCE - r)*d[1]*K1/r;
				//f[i][2] -= (STOP_DISTANCE - r)*d[2]*K1/r;
				
				//f[j][0] += (STOP_DISTANCE - r)*d[0]*K1/r;
				//f[j][1] += (STOP_DISTANCE - r)*d[1]*K1/r;
				//f[j][2] += (STOP_DISTANCE - r)*d[2]*K1/r;
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
			
			temp_f[i].vc[j] = (temp_f[N].p[j] - temp_f[i].p[j]);
			temp_f[i].vn[j] += (temp_f[i].f[j] - DAMP*temp_f[i].vn[j]*dt);
		}
		
		//magc = sqrt(vc[i][0]*vc[i][0] + vc[i][1]*vc[i][1] + vc[i][2]*vc[i][2]);
		//magn = sqrt(vn[i][0]*vn[i][0] + vn[i][1]*vn[i][1] + vn[i][2]*vn[i][2]);
		// Ignored.
		
		for(j=0; j<3; j++)
		{
			//v[i][j] = vc[i][j] + vn[i][j];
			//v[i][j] *= 5.0;
			//p[i][j] += v[i][j]*dt;
			
			temp_f[i].v[j] = 5.0*temp_f[i].vc[j] + temp_f[i].vn[j];
			temp_f[i].v[j] *= SPEED;
			temp_f[i].p[j] += temp_f[i].v[j]*dt;
		}
		temp_f[i].p[2] = 0.0;
		
		// Diagnostics
		
		//printf("The position of particle %i is (%.4f, %.4f, %.4f)\n", 
		//		i, temp_f[i].p[0], temp_f[i].p[1], temp_f[i].p[2]);
				
		//printf("The velocity of particle %i is (%.4f, %.4f, %.4f)\n", 
		//		i, temp_f[i].v[0], temp_f[i].v[1], temp_f[i].v[2]);

		//printf("The forces on particle %i are (%.4f, %.4f, %.4f)\n", 
		//		i, temp_f[i].f[0], temp_f[i].f[1], temp_f[i].f[2]);
		
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
		if( state == GLUT_DOWN ) 
		{
		// when left mouse button goes down.
			coord[0] = (x*4.0/XWindowSize)-2.0;
			coord[2] = 0.0;
			coord[1] = -(y*4.0/YWindowSize)+2.0;
			printf("The sphere is at (%.4f, %.4f, %.4f)\n",
				coord[0], coord[1], coord[2]);
			temp_f[N].p[0] = coord[0];
			temp_f[N].p[1] = coord[1];
			temp_f[N].p[2] = coord[2];			
		}
	//	else if( state == GLUT_UP ) 
	//	{
	//		printf("Left button up!\n");
	//	}
	
	}

	//else if ( button == GLUT_RIGHT_BUTTON )
	//{
	//	/* when right mouse button down */
	//	if( state == GLUT_DOWN ) 
	//	{
	//		//printf("Right button down!\n");
	//	}
	//	else if( state == GLUT_UP ) 
	//	{
	//		//printf("Right button up!\n");
	//	}
	//}
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

	create_fish();
	gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glutDisplayFunc(Display);
	glutTimerFunc(16, update, 0);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}	
	
	
	
	

