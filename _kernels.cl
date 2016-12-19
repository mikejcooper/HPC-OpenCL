#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define blockSize 128
#define nIsPow2 1

void reduce_local(local float* shared_mem, int id, int work_items);


typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global write_only t_speed* cells,
                            global read_only int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{

 /* compute weighting factors */
  float w2 = density * accel / 36.0f;
  float w1 = 4 * w2;


  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[ii * nx + jj].speeds[3] - w1) > 0.0
      && (cells[ii * nx + jj].speeds[6] - w2) > 0.0
      && (cells[ii * nx + jj].speeds[7] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[ii * nx + jj].speeds[1] += w1;
    cells[ii * nx + jj].speeds[5] += w2;
    cells[ii * nx + jj].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii * nx + jj].speeds[3] -= w1;
    cells[ii * nx + jj].speeds[6] -= w2;
    cells[ii * nx + jj].speeds[7] -= w2;
  }
}

// -----------------------------------------------------------------------------------------
//  Add volatile types *****
kernel void prop_rbd_col(global write_only t_speed* cells,
                    global read_only t_speed* tmp_cells,
                    global read_only int* obstacles,
                    int nx, int ny, float omega, int tt, 
                    global float* av_partial_sums, local float* av_local_sums)
{
  float tot_u = 0.0;    /* accumulated magnitudes of velocity for each cell */
  const float d1 = 1 / 36.0;

  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  t_speed tmp_cells_local[1]; 

  /* determine indices of axis-direction   neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (ii + 1) % ny;
  int x_e = (jj + 1) % nx;
  int y_s = (ii == 0) ? (ii + ny - 1) : (ii - 1);
  int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);

  int index = ii * nx + jj;

  // for(int i = 0; i < NSPEEDS; i++){
  //   tmp_cells_local[0].speeds[i] = tmp_cells[index].speeds[i];
  // }

  /* if the cell contains an obstacle */
// -------------prop_rbd_col--------------------------------
      /* don't consider occupied cells */
      if (obstacles[index])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells_local[0].speeds[0] = cells[ii * nx + x_e].speeds[0];
        tmp_cells_local[0].speeds[1] = cells[ii * nx + x_e].speeds[3];
        tmp_cells_local[0].speeds[2] = cells[y_n * nx + jj].speeds[4];
        tmp_cells_local[0].speeds[3] = cells[ii * nx + x_w].speeds[1];
        tmp_cells_local[0].speeds[4] = cells[y_s * nx + jj].speeds[2];
        tmp_cells_local[0].speeds[5] = cells[y_n * nx + x_e].speeds[7];
        tmp_cells_local[0].speeds[6] = cells[y_n * nx + x_w].speeds[8];
        tmp_cells_local[0].speeds[7] = cells[y_s * nx + x_w].speeds[5];
        tmp_cells_local[0].speeds[8] = cells[y_s * nx + x_e].speeds[6];
      } 
// ----------------END--------------------------------------------
      else 
      {

        /* compute local density total */
        float local_density = 0.0;
        local_density += cells[ii * nx + jj].speeds[0];
        local_density += cells[ii * nx + x_e].speeds[3];
        local_density += cells[y_n * nx + jj].speeds[4];
        local_density += cells[ii * nx + x_w].speeds[1];
        local_density += cells[y_s * nx + jj].speeds[2];
        local_density += cells[y_n * nx + x_e].speeds[7];
        local_density += cells[y_n * nx + x_w].speeds[8];
        local_density += cells[y_s * nx + x_w].speeds[5];
        local_density += cells[y_s * nx + x_e].speeds[6];


        float local_density_invert = 1 / local_density;
        /* compute x velocity component */
        float u_x = (cells[ii * nx + x_w].speeds[1]
                      + cells[y_s * nx + x_w].speeds[5]
                      + cells[y_n * nx + x_w].speeds[8]
                      - (cells[ii * nx + x_e].speeds[3]
                         + cells[y_s * nx + x_e].speeds[6]
                         + cells[y_n * nx + x_e].speeds[7]))
                     * local_density_invert;
        /* compute y velocity component */
        float u_y = (cells[y_s * nx + jj].speeds[2]
                      + cells[y_s * nx + x_w].speeds[5]
                      + cells[y_s * nx + x_e].speeds[6]
                      - (cells[y_n * nx + jj].speeds[4]
                         + cells[y_n * nx + x_e].speeds[7]
                         + cells[y_n * nx + x_w].speeds[8]))
                     * local_density_invert;

        tmp_cells_local[0].speeds[0] = cells[ii * nx + jj].speeds[0]
        + omega
        * (local_density * d1 * (16.0f - (u_x * u_x + u_y * u_y) * 864.0f * d1)
           - cells[ii * nx + jj].speeds[0]);
        tmp_cells_local[0].speeds[1] = cells[ii * nx + x_w].speeds[1]
        + omega
        * (local_density * d1 * (4.0f + u_x * 12.0f + (u_x * u_x) * 648.0f * d1- (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[ii * nx + x_w].speeds[1]);
        tmp_cells_local[0].speeds[2] = cells[y_s * nx + jj].speeds[2]
        + omega
        * (local_density * d1 * (4.0f + u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * nx + jj].speeds[2]);
        tmp_cells_local[0].speeds[3] = cells[ii * nx + x_e].speeds[3]
        + omega
        * (local_density * d1 * (4.0f - u_x * 12.0f + (u_x * u_x) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[ii * nx + x_e].speeds[3]);
        tmp_cells_local[0].speeds[4] = cells[y_n * nx + jj].speeds[4]
        + omega
        * (local_density * d1 * (4.0f - u_y * 12.0f + (u_y * u_y) * 648.0f * d1 - (216.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * nx + jj].speeds[4]);
        tmp_cells_local[0].speeds[5] = cells[y_s * nx + x_w].speeds[5]
        + omega
        * (local_density * d1 * (1.0f + (u_x + u_y) * 3.0f + ((u_x + u_y) * (u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * nx + x_w].speeds[5]);
        tmp_cells_local[0].speeds[6] = cells[y_s * nx + x_e].speeds[6]
        + omega
        * (local_density * d1 * (1.0f + (- u_x + u_y) * 3.0f + ((- u_x + u_y) * (- u_x + u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_s * nx + x_e].speeds[6]);
        tmp_cells_local[0].speeds[7] = cells[y_n * nx + x_e].speeds[7]
        + omega
        * (local_density * d1 * (1.0f + (- u_x - u_y) * 3.0f + ((- u_x - u_y) * (- u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * nx + x_e].speeds[7]);
        tmp_cells_local[0].speeds[8] = cells[y_n * nx + x_w].speeds[8]
        + omega
        * (local_density * d1 * (1.0f + (u_x - u_y) * 3.0f + ((u_x - u_y) * (u_x - u_y)) * 162.0f * d1 - (54.0f * d1 * (u_x * u_x + u_y * u_y)))
           - cells[y_n * nx + x_w].speeds[8]);

        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
    }

  for(int i = 0; i < NSPEEDS; i++){
      tmp_cells[index].speeds[i] = tmp_cells_local[0].speeds[i];
  }


  // --------------Local REDUCTION -----------------

  int num_wrk_items  = get_local_size(0) * get_local_size(1);   // # work-items in work-group 
  int local_id       = get_local_size(0) * get_local_id(1) + get_local_id(0);     // ID of work-item within work-group          
  int group_id       = get_num_groups(0) * get_group_id(1) + get_group_id(0);     // ID of work-group

  av_local_sums[local_id] = tot_u;

  barrier(CLK_LOCAL_MEM_FENCE);

// -----------------REDUCTION ----------------------------------------

  // #pragma unroll 1
  for (int i = num_wrk_items / 2; i > 0; i >>= 1) {  
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < i){
          av_local_sums[local_id] += av_local_sums[local_id + i];
      }
  }

  if (local_id == 0){
      av_partial_sums[group_id] = av_local_sums[0];                               
  }

}

// ---------------- REDUCTION v3-------------------

kernel void reduce(global float* av_partial_sums,
                   global float* av_vels, int tt, int tot_cells, local float* shared_mem)
{
  int group_size  = get_global_size(0);  // # work-items   == # work-groups           
  int global_id    = get_global_id(0);   // ID of work-item
  shared_mem[global_id] = av_partial_sums[global_id];

  // reduce_local(shared_mem, global_id, group_size);
  
  barrier(CLK_GLOBAL_MEM_FENCE);

// -----------------REDUCTION ----------------------------------------

  // #pragma unroll 1
  for (int i = group_size / 2; i > 0; i >>= 1) {  
      barrier(CLK_GLOBAL_MEM_FENCE);
      if (global_id < i){
          shared_mem[global_id] += shared_mem[global_id + i];
      }
  }

  if (global_id == 0){
      av_vels[tt] = shared_mem[0]/tot_cells;                               
  }
}

void reduce_local(local float* shared_mem, int id, int group_size){
  barrier(CLK_LOCAL_MEM_FENCE);
  
  if (id == 0){
    float total = 0.0f;
    for (int i=0; i<group_size; i++) {        
      total += shared_mem[i];             
    }                                     
    shared_mem[0] = total;    
  } 

  // // #pragma unroll 1
  // for (int i = group_size / 2; i > 0; i >>= 1) {  
  //     if (id < i){
  //         shared_mem[id] += shared_mem[id + i];
  //     }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  // }

  // if (id < 32)
  // {
  //     if (blockSize >=  64) { shared_mem[id] += shared_mem[id + 32]; }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     if (blockSize >=  32) { shared_mem[id] += shared_mem[id + 16]; }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     if (blockSize >=  16) { shared_mem[id] += shared_mem[id +  8]; }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     if (blockSize >=   8) { shared_mem[id] += shared_mem[id +  4]; }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     if (blockSize >=   4) { shared_mem[id] += shared_mem[id +  2]; }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     if (blockSize >=   2) { shared_mem[id] += shared_mem[id +  1]; }
  // }


}


// ---------------- REDUCTION v2-------------------

// kernel void reduce(global float* av_partial_sums,
//                    global float* av_vels, int tt, int tot_cells, local float* shared_mem)
// {
//   int num_work_groups  = get_global_size(0);  // # work-items   == # work-groups           
//   int global_id    = get_global_id(0);   // ID of work-item
  

//   for (int i = num_work_groups / 2; i > 0; i /= 2) {  
//       if (global_id < i){
//           av_partial_sums[global_id] += av_partial_sums[global_id + i]; 
//       }
//       barrier(CLK_LOCAL_MEM_FENCE);
//   }   

//   if (global_id == 0){
//       av_vels[tt] = av_partial_sums[0]/tot_cells;                               
//   }
// }

// ---------------- REDUCTION v1-------------------

// kernel void reduce(global float* av_partial_sums,
//                    global float* av_vels, int tt, int tot_cells)
// {
//   int global_size  = get_global_size(0);    // number of items the work group (number of columns)              
//   int global_id    = get_global_id(0);   // ID of specific coloumn in the work group 

//   if (global_id == 0){
//     float total = 0.0f;
//     for (int i=0; i<global_size; i++) {        
//       total += av_partial_sums[i];             
//     }                                     
//     av_vels[tt] = total/tot_cells;    
//   } 
// }