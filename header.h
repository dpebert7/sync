//Header file for sync project.

#define EYEZ		50.0 // Effectively sets x- and y-coordinates from -EYEZ to +EYEZ
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



// Global Variables
double CENTER[3], // Center or point of attraction
	r;	  // Distance between particles
double 	TIMERUNNING = 0.0;
double 	SPEED = 20.0;

int 	FOODPOINTER = NFISH; // Our first food particle
int	PAUSE = 0;




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
			coord[0] = (x*EYEZ*2.0/XWindowSize)-EYEZ;
			coord[1] = -(y*EYEZ*2.0/YWindowSize)+EYEZ;
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
