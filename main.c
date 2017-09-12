//  rod.c
//  NuclearRod
//  Created by Dan Wilson on 20/12/2016.
//  Copyright © 2016 Dan Wilson. All rights reserved.

/*rod.c is a program that calculates the temperature distribution around a buried nuclear waster rod.
The original problem is a 4D heat equation, which, after using symmetry and the backwards euler method, can be converted into a simple 1D 'Ax=b' matrix equation
The rod emits heat given by 'double source', and this temperature distribution is modelled over TOTAL_LENGTH in N discrete locations.
This is then marched forwards in time, M times, for TOTAL_TIME.
The program uses GSL functions to solve the tridiagonal matrix A, and thus all relevant vectors must be defined as gsl_vector.
The original solution is featured on page 42 of http://espace.library.uq.edu.au/view/UQ:239427/Lectures_Book.pdf in which Dr Olsen-Kettle has produced a solution in Matlab.
This program seeks to reproduce her results in C.
*/

#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <limits.h>

#define KAPPA 2.0e7 //thermal diffusivity constant in cm^2
#define TOTAL_TIME 100.0 //in years
#define TOTAL_LENGTH 100.0 //in cm
#define T_ROD 1.0 //inital temp of rod in K
#define INIT_TEMP 300.0 //initial temp of the environment in K
#define TAU 100.0 //half life of rod in years
#define R_ROD 25.0 //radius of rod in cm

//fractions of total time that temperature is recorded
#define T1 1.0/100.0
#define T2 1.0/10.0
#define T3 1.0/2.0
#define T4 1.0

#define GNUPLOT_DATA "data.txt"
#define GNUPLOT_SCRIPT "script.txt"
#define GNUPLOT_EXE "gnuplot"

static const int N = 99; //number of discretisations of space
static const int M = 1000; //number of discretisations of time

//gsl vectors for use with the gsl_linalg_solve_tridiag() function
typedef struct matrix {
gsl_vector *temp;
gsl_vector *superdiag;
gsl_vector *subdiag;
gsl_vector *diag;
gsl_vector *b;
} Matrix;


//function to check if there is enough memory when a structure is allocated
static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "Out of memory!\n");
    exit(1);
  }
  return p;
}

//function to check the current time of the simulation, and if that corresonds to T1, T2, T3 or T4 times the total time, then the temperature is stored in one of the columns of the 4xN array storedTemp.
static void storeTemp(double storedTemp[][N], int k, Matrix *A) {
  if (k== (M * T1)) {
    for (int i=0; i<N; i++) storedTemp[0][i] = gsl_vector_get (A->temp, i);
  }
  if (k== (M * T2)) {
    for (int i=0; i<N; i++) storedTemp[1][i] = gsl_vector_get (A->temp, i);
  }
  if (k== (M * T3)) {
    for (int i=0; i<N; i++) storedTemp[2][i] = gsl_vector_get (A->temp, i);
  }
  if (k== (M * T4)) {
    for (int i=0; i<N; i++) storedTemp[3][i] = gsl_vector_get (A->temp, i);
  }
}

//function to print the temperatures at the 4 given times stored by storeTemp() into a file, and then plotted using gnuplot and script.txt.
static void plotData(double storedTemp[][N],double dr, FILE *foutput) {
  double r=0.0;
  for (int i=0; i<N; i++) {
    r+=dr;
    fprintf(foutput, "%g\t%g\t%g\t%g\t%g\n", r, storedTemp[0][i] , storedTemp[1][i], storedTemp[2][i], storedTemp[3][i]);
  }

  fclose(foutput);

  char command[PATH_MAX];
  snprintf(command, sizeof(command), "%s %s", GNUPLOT_EXE , GNUPLOT_SCRIPT );
  system( command );
}

//function to create the matrix for the problem. Within the for loop, the terms of the tridiagonal matrix are defined, with the  sub/super diagonal arrays one term shorter than the diagonal term array. The quantity 's' is the gain parameter that is present throughout the matrix. The initial temperature is set to INIT_TEMP.
//the array boundaryCond[] exists purely for the purposes of satisfying the Dirichlet boundary conditions (temp = 300K at r = TOTAL_LENGTH).
//the resetting of the first term of the diag array satisfies the Neumann boundary conditions at r = 0 (temperature cannot flow into r = 0 region).
static void initMatrix(double dt, double dr, double *boundaryCond, Matrix *A) {
  double s = KAPPA * dt / (pow(dr,2.0));

  for( int i = 0; i< N; i++) {
    if(i != N-1) {
      gsl_vector_set ( A->superdiag, i, -s -s/(2.0*(i+1.0)) );
      gsl_vector_set ( A->subdiag, i, -s +s/(2.0*(i+2.0)) );
    }
    gsl_vector_set ( A->diag, i, 1.0 + 2.0*s );
    gsl_vector_set ( A->temp, i, INIT_TEMP);
    boundaryCond[i] = 0.0;
  }
  //set boundary conditions
  gsl_vector_set ( A->diag, 0, 1 + (3.0/2.0)*s );
  boundaryCond[N-1] = -(-s-s/(2*N))*INIT_TEMP;
}

//function to define the b term for the Ax=b matrix problem. This term contains the temperature from the previous time step, the source term (if within the rod), and the Dirichlet boundary condtion array.
static void setb(double dr, double dt, double time, double *boundaryCond, Matrix *A) {
  double source = (T_ROD*exp(- time / TAU))/pow(R_ROD,2);

  for (int i=0; i<N; i++) {
    if ((i+1)*dr < R_ROD) {
      gsl_vector_set(A->b, i, gsl_vector_get (A->temp, i) + KAPPA*dt*source +boundaryCond[i]);
    }
    else {
      gsl_vector_set(A->b, i, gsl_vector_get (A->temp, i) +boundaryCond[i]);
    }
  }
}

//function to allocate the neccesary number of elements for each of the gsl vectors
static void allocateMatrix(Matrix *A) {
  A->temp = gsl_vector_alloc (N);
  A->superdiag = gsl_vector_alloc (N-1);
  A->subdiag = gsl_vector_alloc (N-1);
  A->diag = gsl_vector_alloc (N);
  A->b = gsl_vector_alloc (N);
}

//main function discretises time and distance into dt and dr respectively
//the gsl_vectors are allocated and freed, to be filled by initMatrix() and solved using gsl_linalg_solve_tridiag(), which updates the temperature to the next time step.
//the b term of the Ax=b matrix problem is updated every time step by setb() to include the updated temperature and source term.
int main(int argc, const char * argv[]) {

  FILE *foutput = fopen (GNUPLOT_DATA , "w");

  double dt, dr, time;
  dt = TOTAL_TIME / M;
  dr = TOTAL_LENGTH / (N+1);

  double boundaryCond[N];
  double storedTemp[4][N];

  Matrix *A = xmalloc(sizeof(Matrix));
  allocateMatrix(A);
  initMatrix(dt,dr, boundaryCond, A);

  time = 0.0;

  for(int k=1; k<=M; k++) {

    time += dt;
    setb(dr, dt, time, boundaryCond, A);
    gsl_linalg_solve_tridiag ( A->diag , A->superdiag, A->subdiag, A->b, A->temp);
    storeTemp(storedTemp, k, A);

  }

  plotData(storedTemp, dr, foutput);
  freeMatrix(A);

  return 0;
}
