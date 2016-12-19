/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 ** 'unwrapped' in row major order to give a 1D array:
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 **
 ** Grid indicies are:
 **
 **          ny
 **          ^       cols(jj)
 **          |  ----- ----- -----
 **          | | ... | ... | etc |
 **          |  ----- ----- -----
 ** rows(ii) | | 1,0 | 1,1 | 1,2 |
 **          |  ----- ----- -----
 **          | | 0,0 | 0,1 | 0,2 |
 **          |  ----- ----- -----
 **          ----------------------> nx
 **
 ** Note the names of the input parameter and obstacle files
 ** are passed on the command line, e.g.:
 **
 **   d2q9-bgk.exe input.params obstacles.dat
 **
 ** Be sure to adjust the grid dimensions in the parameter file
 ** if you choose a different obstacle file.
 */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"


#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<sys/time.h>
#include<sys/resource.h>

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  int    tot_cells;
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
  cl_device_id      device;
  cl_context        context;
  cl_command_queue  queue;

  cl_program program;
  cl_kernel  accelerate_flow;
  cl_kernel  prop_rbd_col;
  cl_kernel  reduce;

  cl_mem cells;
  cl_mem tmp_cells;
  cl_mem obstacles;
  cl_mem av_partial_sums;
  cl_mem av_vels;

} t_ocl;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speeds;

typedef struct
{
  float s0[1048576];
  float s1[1048576];
  float s2[1048576];
  float s3[1048576];
  float s4[1048576];
  float s5[1048576];
  float s6[1048576];
  float s7[1048576];
  float s8[1048576];

} t_cells;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_cells** cells_ptr, t_cells** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), prop_rbd_col() & collision()
*/
void timestep(const t_param params, t_cells* cells, t_cells* tmp_cells, int* obstacles, int tt, float* av_vels, t_ocl ocl);
int accelerate_flow(const t_param params, t_cells* cells, int* obstacles, t_ocl ocl);
int propagate(const t_param params, t_cells* cells, t_cells* tmp_cells, t_ocl ocl);
int prop_rbd_col(const t_param params, t_cells* cells, t_cells* tmp_cells, int* obstacles, int tt, float* av_vels, t_ocl ocl);
int collision(const t_param params, t_cells* cells, t_cells* tmp_cells, int* obstacles, t_ocl ocl);
int write_values(const t_param params, t_cells* cells, int* obstacles, float* av_vels);
void reduce(const t_param params, int tt, t_ocl ocl);


/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_cells** cells_ptr, t_cells** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */


/* calculate Reynolds number */

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

void clPrintDevInfo(cl_device_id device);

cl_device_id selectOpenCLDevice();