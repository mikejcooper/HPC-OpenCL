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

#include "d2q9-bgk.h"




/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_ocl    ocl;                 /* struct to hold OpenCL objects */
  t_cells* cells     = NULL;    /* grid containing fluid densities */
  t_cells* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  cl_int err;
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl);
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  // Write cells to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_cells), cells, 0, NULL, NULL);
  checkError(err, "writing cells data", __LINE__);


  // Write obstacles to OpenCL buffer
  err = clEnqueueWriteBuffer(
    ocl.queue, ocl.obstacles, CL_TRUE, 0,
    sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
  checkError(err, "writing obstacles data", __LINE__);


  for (int tt = 0; tt < params.maxIters; tt++)
  {
    // printf("s8 =%f\n", cells->s8[100]);

    timestep(params, cells, tmp_cells, obstacles, tt, av_vels, ocl);
    cl_mem temp = ocl.cells;
    ocl.cells = ocl.tmp_cells;
    ocl.tmp_cells = temp;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }  

  err = clFinish(ocl.queue);
  // Read av_vels from device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.av_vels, CL_TRUE, 0,
    sizeof(float) * params.maxIters, av_vels, 0, NULL, NULL);
  checkError(err, "reading tmp_cells data", __LINE__);

    // Read cells from device
  err = clEnqueueReadBuffer(
    ocl.queue, ocl.cells, CL_TRUE, 0,
    sizeof(t_cells), cells, 0, NULL, NULL);
  checkError(err, "reading cells data", __LINE__);


  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  // printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  
  clPrintDevInfo(selectOpenCLDevice());

  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

  return EXIT_SUCCESS;
}

void timestep(const t_param params, t_cells* cells, t_cells* tmp_cells, int* obstacles, int tt, float* av_vels, t_ocl ocl)
{
  accelerate_flow(params, cells, obstacles, ocl);
  prop_rbd_col(params, cells, tmp_cells, obstacles, tt, av_vels, ocl);
  reduce(params, tt, ocl);
}

int accelerate_flow(const t_param params, t_cells* cells, int* obstacles, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting accelerate_flow arg 0", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting accelerate_flow arg 1", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_int), &params.nx);
  checkError(err, "setting accelerate_flow arg 2", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_int), &params.ny);
  checkError(err, "setting accelerate_flow arg 3", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_float), &params.density);
  checkError(err, "setting accelerate_flow arg 4", __LINE__);
  err = clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_float), &params.accel);
  checkError(err, "setting accelerate_flow arg 5", __LINE__);

  // Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing accelerate_flow kernel", __LINE__);
  // err = clFinish(ocl.queue);
  // checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

int prop_rbd_col(const t_param params, t_cells* cells, t_cells* tmp_cells, int* obstacles, int tt, float* av_vels, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.prop_rbd_col, 0, sizeof(cl_mem), &ocl.cells);
  checkError(err, "setting prop_rbd_col arg 0", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 1, sizeof(cl_mem), &ocl.tmp_cells);
  checkError(err, "setting prop_rbd_col arg 1", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 2, sizeof(cl_mem), &ocl.obstacles);
  checkError(err, "setting prop_rbd_col arg 2", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 3, sizeof(cl_int), &params.nx);
  checkError(err, "setting prop_rbd_col arg 3", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 4, sizeof(cl_int), &params.ny);
  checkError(err, "setting prop_rbd_col arg 4", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 5, sizeof(cl_float), &params.omega);
  checkError(err, "setting prop_rbd_col arg 5", __LINE__);
  err = clSetKernelArg(ocl.prop_rbd_col, 6, sizeof(cl_int), &tt);
  checkError(err, "setting prop_rbd_col arg 5", __LINE__); 
  err = clSetKernelArg(ocl.prop_rbd_col, 7, sizeof(cl_mem), &ocl.av_partial_sums);
  checkError(err, "setting prop_rbd_col arg 7", __LINE__); 
  err = clSetKernelArg(ocl.prop_rbd_col, 8, sizeof(float) * params.nx * 1, NULL);
  checkError(err, "setting prop_rbd_col arg 7", __LINE__);   


  // Enqueue kernel
  size_t global[2] = {params.nx, params.ny};
  size_t local[2]  = {params.nx, 1};
   // size_t local[2]  = {1, 1};

  err = clEnqueueNDRangeKernel(ocl.queue, ocl.prop_rbd_col,
                               2, NULL, global, local, 0, NULL, NULL);
  checkError(err, "enqueueing prop_rbd_col kernel", __LINE__);
  // err = clFinish(ocl.queue);
  // checkError(err, "waiting for accelerate_flow kernel", __LINE__);

  return EXIT_SUCCESS;
}

void reduce(const t_param params, int tt, t_ocl ocl)
{
  cl_int err;

  // Set kernel arguments
  err = clSetKernelArg(ocl.reduce, 0, sizeof(cl_mem), &ocl.av_partial_sums);
  checkError(err, "setting reduce arg 0", __LINE__);
  err = clSetKernelArg(ocl.reduce, 1, sizeof(cl_mem), &ocl.av_vels);
  checkError(err, "setting reduce arg 1", __LINE__);
  err = clSetKernelArg(ocl.reduce, 2, sizeof(cl_int), &tt);
  checkError(err, "setting reduce arg 2", __LINE__);
  err = clSetKernelArg(ocl.reduce, 3, sizeof(cl_int), &params.tot_cells);
  checkError(err, "setting reduce arg 2", __LINE__);  
  err = clSetKernelArg(ocl.reduce, 4, sizeof(float) * params.nx * 1, NULL);
  checkError(err, "setting reduce arg 2", __LINE__);


  //Enqueue kernel
  size_t global[1] = {params.nx};
  err = clEnqueueNDRangeKernel(ocl.queue, ocl.reduce,
                               1, NULL, global, NULL, 0, NULL, NULL);
  checkError(err, "enqueueing reduce kernel", __LINE__);

    // err = clFinish(ocl.queue);

}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_cells** cells_ptr, t_cells** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  char*  ocl_src;        /* OpenCL kernel source */
  long   ocl_size;       /* size of OpenCL kernel source */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);


// Allocate memory 

/* main grid */
  *cells_ptr = (t_cells*)malloc(sizeof(t_cells));
  // (*cells_ptr)->s0 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s1 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s2 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s3 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s4 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s5 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s6 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s7 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*cells_ptr)->s8 = (float*)malloc(sizeof(float) * (params->ny * params->nx));

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_cells*)malloc(sizeof(t_cells));
  // (*tmp_cells_ptr)->s0 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s1 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s2 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s3 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s4 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s5 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s6 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s7 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  // (*tmp_cells_ptr)->s8 = (float*)malloc(sizeof(float) * (params->ny * params->nx));

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.0 / 9.0;
  float w1 = params->density      / 9.0;
  float w2 = params->density      / 36.0;

  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      

      (*cells_ptr)->s0[ii * params->nx + jj] = w0;
      /* axis directions */
      (*cells_ptr)->s1[ii * params->nx + jj] = w1;
      (*cells_ptr)->s2[ii * params->nx + jj] = w1;
      (*cells_ptr)->s3[ii * params->nx + jj] = w1;
      (*cells_ptr)->s4[ii * params->nx + jj] = w1;
      /* diagonals */
      (*cells_ptr)->s5[ii * params->nx + jj] = w2;
      (*cells_ptr)->s6[ii * params->nx + jj] = w2;
      (*cells_ptr)->s7[ii * params->nx + jj] = w2;
      (*cells_ptr)->s8[ii * params->nx + jj] = w2;

    }
  }

  /* first set all cells in obstacle array to zero */
  for (int ii = 0; ii < params->ny; ii++)
  {
    for (int jj = 0; jj < params->nx; jj++)
    {
      (*obstacles_ptr)[ii * params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  
  int cells_blocked = 0;
  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[yy * params->nx + xx] = blocked;
    cells_blocked ++;
  }

  /* and close the file */
  fclose(fp);

  params->tot_cells = params->nx * params->ny - cells_blocked;
  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


  cl_int err;

  ocl->device = selectOpenCLDevice();

  // Create OpenCL context
  ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
  checkError(err, "creating context", __LINE__);

  fp = fopen(OCLFILE, "r");
  if (fp == NULL)
  {
    sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
    die(message, __LINE__, __FILE__);
  }

  // Create OpenCL command queue
  ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
  checkError(err, "creating command queue", __LINE__);

  // Load OpenCL kernel source
  fseek(fp, 0, SEEK_END);
  ocl_size = ftell(fp) + 1;
  ocl_src = (char*)malloc(ocl_size);
  memset(ocl_src, 0, ocl_size);
  fseek(fp, 0, SEEK_SET);
  fread(ocl_src, 1, ocl_size, fp);
  fclose(fp);

  // Create OpenCL program
  ocl->program = clCreateProgramWithSource(
    ocl->context, 1, (const char**)&ocl_src, NULL, &err);
  free(ocl_src);
  checkError(err, "creating program", __LINE__);

  // Build OpenCL program
  err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t sz;
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
    char *buildlog = malloc(sz);
    clGetProgramBuildInfo(
      ocl->program, ocl->device,
      CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
    fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
    free(buildlog);
  }
  checkError(err, "building program", __LINE__);

  // Create OpenCL kernels
  ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
  checkError(err, "creating accelerate_flow kernel", __LINE__);
  ocl->prop_rbd_col = clCreateKernel(ocl->program, "prop_rbd_col", &err);
  checkError(err, "creating prop_rbd_col kernel", __LINE__);  
  ocl->reduce = clCreateKernel(ocl->program, "reduce", &err);
  checkError(err, "creating reduce kernel", __LINE__);

  // Allocate OpenCL buffers
  // LOOK AT CL_MEM_READ_WRITE!! only wirte or read

  ocl->cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_cells), NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
  ocl->tmp_cells = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(t_cells), NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
  ocl->obstacles = clCreateBuffer(
    ocl->context, CL_MEM_READ_ONLY,
    sizeof(cl_int) * params->nx * params->ny, NULL, &err);
  checkError(err, "creating obstacles buffer", __LINE__);
  ocl->av_partial_sums = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->ny, NULL, &err);
  checkError(err, "creating av_vel buffer", __LINE__);
  ocl->av_vels = clCreateBuffer(
    ocl->context, CL_MEM_READ_WRITE,
    sizeof(cl_float) * params->maxIters, NULL, &err);
  checkError(err, "creating av_vel buffer", __LINE__);
  

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_cells** cells_ptr, t_cells** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  clReleaseMemObject(ocl.cells);
  clReleaseMemObject(ocl.tmp_cells);
  clReleaseMemObject(ocl.obstacles);
  clReleaseKernel(ocl.accelerate_flow);
  clReleaseKernel(ocl.prop_rbd_col);
  clReleaseProgram(ocl.program);
  clReleaseCommandQueue(ocl.queue);
  clReleaseContext(ocl.context);

  return EXIT_SUCCESS;
}





int write_values(const t_param params, t_cells* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.0 / 3.0; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.ny; ii++)
  {
    for (int jj = 0; jj < params.nx; jj++)
    {
      /* an occupied cell */
      if (obstacles[ii * params.nx + jj])
      {
        u_x = u_y = u = 0.0;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.0;

        local_density += cells->s0[ii * params.nx + jj];
        local_density += cells->s1[ii * params.nx + jj];
        local_density += cells->s2[ii * params.nx + jj];
        local_density += cells->s3[ii * params.nx + jj];
        local_density += cells->s4[ii * params.nx + jj];
        local_density += cells->s5[ii * params.nx + jj];
        local_density += cells->s6[ii * params.nx + jj];
        local_density += cells->s7[ii * params.nx + jj];
        local_density += cells->s8[ii * params.nx + jj];

        /* compute x velocity component */
        u_x = (cells->s1[ii * params.nx + jj]
          + cells->s5[ii * params.nx + jj]
          + cells->s8[ii * params.nx + jj]
          -(cells->s3[ii * params.nx + jj]
            +cells->s6[ii * params.nx + jj]
            +cells->s7[ii * params.nx + jj]))
          /local_density;

        u_y = (cells->s2[ii * params.nx + jj]
          + cells->s5[ii * params.nx + jj]
          + cells->s6[ii * params.nx + jj]
          -(cells->s4[ii * params.nx + jj]
            +cells->s7[ii * params.nx + jj]
            +cells->s8[ii * params.nx + jj]))
          /local_density;

        /* compute norm of velocity */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", jj, ii, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
  cl_int err;
  cl_uint num_platforms = 0;
  cl_uint total_devices = 0;
  cl_platform_id platforms[8];
  cl_device_id devices[MAX_DEVICES];
  char name[MAX_DEVICE_NAME];

  // Get list of platforms
  err = clGetPlatformIDs(8, platforms, &num_platforms);
  checkError(err, "getting platforms", __LINE__);

  // Get list of devices
  for (cl_uint p = 0; p < num_platforms; p++)
  {
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-total_devices, devices+total_devices,
                         &num_devices);
    checkError(err, "getting device name", __LINE__);
    total_devices += num_devices;
  }

  // Print list of devices
  printf("\nAvailable OpenCL devices:\n");
  for (cl_uint d = 0; d < total_devices; d++)
  {
    clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
    printf("%2d: %s\n", d, name);
  }
  printf("\n");

  // Use first device unless OCL_DEVICE environment variable used
  cl_uint device_index = 0;
  char *dev_env = getenv("OCL_DEVICE");
  if (dev_env)
  {
    char *end;
    device_index = strtol(dev_env, &end, 10);
    if (strlen(end))
      die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
  }

  if (device_index >= total_devices)
  {
    fprintf(stderr, "device index set to %d but only %d devices available\n",
            device_index, total_devices);
    exit(1);
  }

  // Print OpenCL device name
  clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                  MAX_DEVICE_NAME, name, NULL);
  printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

  return devices[device_index];
}



void clPrintDevInfo(cl_device_id device) {
  // char device_string[1024];

  // // CL_DEVICE_NAME
  // clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  // printf("  CL_DEVICE_NAME: \t\t\t%s\n", device_string);

  // // CL_DEVICE_VENDOR
  // clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
  // printf("  CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

  // // CL_DRIVER_VERSION
  // clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
  // printf("  CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

  // // CL_DEVICE_INFO
  // cl_device_type type;
  // clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  // if( type & CL_DEVICE_TYPE_CPU )
  //   printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
  // if( type & CL_DEVICE_TYPE_GPU )
  //   printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
  // if( type & CL_DEVICE_TYPE_ACCELERATOR )
  //   printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
  // if( type & CL_DEVICE_TYPE_DEFAULT )
  //   printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

  // // CL_DEVICE_MAX_COMPUTE_UNITS
  // cl_uint compute_units;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  // printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

  // // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  // size_t workitem_dims;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
  // printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", workitem_dims);

  // // CL_DEVICE_MAX_WORK_ITEM_SIZES
  // size_t workitem_size[3];
  // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
  // printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", workitem_size[0], workitem_size[1], workitem_size[2]);

  // // CL_DEVICE_MAX_WORK_GROUP_SIZE
  // size_t workgroup_size;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
  // printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", workgroup_size);

  // // CL_DEVICE_MAX_CLOCK_FREQUENCY
  // cl_uint clock_frequency;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
  // printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

  // // CL_DEVICE_ADDRESS_BITS
  // cl_uint addr_bits;
  // clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
  // printf("  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

  // // CL_DEVICE_MAX_MEM_ALLOC_SIZE
  // cl_ulong max_mem_alloc_size;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
  // printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

  // // CL_DEVICE_GLOBAL_MEM_SIZE
  // cl_ulong mem_size;
  // clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  // printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

  // // CL_DEVICE_ERROR_CORRECTION_SUPPORT
  // cl_bool error_correction_support;
  // clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
  // printf("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

  // // CL_DEVICE_LOCAL_MEM_TYPE
  // cl_device_local_mem_type local_mem_type;
  // clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
  // printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

  // // CL_DEVICE_LOCAL_MEM_SIZE
  // clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  // printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

  // // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  // clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
  // printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

  // // CL_DEVICE_QUEUE_PROPERTIES
  // cl_command_queue_properties queue_properties;
  // clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
  // if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
  //   printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
  // if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
  //   printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

  // // CL_DEVICE_IMAGE_SUPPORT
  // cl_bool image_support;
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
  // printf("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

  // // CL_DEVICE_MAX_READ_IMAGE_ARGS
  // cl_uint max_read_image_args;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
  // printf("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

  // // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
  // cl_uint max_write_image_args;
  // clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
  // printf("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);

  // // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
  // size_t szMaxDims[5];
  // printf("\n  CL_DEVICE_IMAGE <dim>");
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
  // printf("\t\t\t2D_MAX_WIDTH\t %u\n", szMaxDims[0]);
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
  // printf("\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", szMaxDims[1]);
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
  // printf("\t\t\t\t\t3D_MAX_WIDTH\t %u\n", szMaxDims[2]);
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
  // printf("\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", szMaxDims[3]);
  // clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
  // printf("\t\t\t\t\t3D_MAX_DEPTH\t %u\n", szMaxDims[4]);

  // // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
  // printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
  // cl_uint vec_width [6];
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_float, sizeof(cl_uint), &vec_width[4], NULL);
  // clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_float, sizeof(cl_uint), &vec_width[5], NULL);
  // printf("CHAR %u, SHORT %u, INT %u, float %u, float %u\n\n\n",
  //  vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4]);


  //   printf("-------------------------NEW Fucntion----------------------\n\n");





}