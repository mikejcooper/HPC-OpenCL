#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9
#define blockSize 128
#define nIsPow2 1

void reduce_local(local float* shared_mem, int id, int work_items);

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

kernel void accelerate_flow(global write_only t_cells* cells,
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
      && (cells->s3[ii * nx + jj] - w1) > 0.0f
      && (cells->s6[ii * nx + jj] - w2) > 0.0f
      && (cells->s7[ii * nx + jj] - w2) > 0.0f)
  {

    /* increase 'east-side' densities */
    cells->s1[ii * nx + jj] += w1;
    cells->s5[ii * nx + jj] += w2;
    cells->s8[ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    cells->s3[ii * nx + jj] -= w1;
    cells->s6[ii * nx + jj] -= w2;
    cells->s7[ii * nx + jj] -= w2;
  }
}

// -----------------------------------------------------------------------------------------
//  Add volatile types *****
kernel void prop_rbd_col(global write_only t_cells* cells,
                    global read_only t_cells* tmp_cells,
                    global read_only int* obstacles,
                    int nx, int ny, float omega, int tt, 
                    global float* av_partial_sums, local float* av_local_sums)
{
  float tot_u = 0.0;    /* accumulated magnitudes of velocity for each cell */
  float val1 = 1.0f / 18.0f;
  float val2 = 1.0f / 72.0f;

  /* get column and row indices */
  int jj = get_global_id(0);
  int ii = get_global_id(1);

  // t_cells tmp_cells_local[1] = malloc(sizeof(t_cells) + 9 * sizeof(float))

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
        tmp_cells->s0[ii * nx + jj] = cells->s0[ii * nx + jj]; /* central cell, no movement */
        tmp_cells->s1[ii * nx + jj] = cells->s3[ii * nx + x_e]; /* east */
        tmp_cells->s2[ii * nx + jj] = cells->s4[y_n * nx + jj]; /* north */
        tmp_cells->s3[ii * nx + jj] = cells->s1[ii * nx + x_w]; /* west */
        tmp_cells->s4[ii * nx + jj] = cells->s2[y_s * nx + jj]; /* south */
        tmp_cells->s5[ii * nx + jj] = cells->s7[y_n * nx + x_e]; /* north-east */
        tmp_cells->s6[ii * nx + jj] = cells->s8[y_n * nx + x_w]; /* north-west */
        tmp_cells->s7[ii * nx + jj] = cells->s5[y_s * nx + x_w]; /* south-west */
        tmp_cells->s8[ii * nx + jj] = cells->s6[y_s * nx + x_e]; /* south-east */
      } 
// ----------------END--------------------------------------------
      else 
      {

            /* compute local density total */
        float local_density = 0.0f;
        local_density += cells->s0[ii * nx + jj];
        local_density += cells->s1[ii * nx + x_w];
        local_density += cells->s2[y_s * nx + jj];
        local_density += cells->s3[ii * nx + x_e];
        local_density += cells->s4[y_n * nx + jj];
        local_density += cells->s5[y_s * nx + x_w];
        local_density += cells->s6[y_s * nx + x_e];
        local_density += cells->s7[y_n * nx + x_e];
        local_density += cells->s8[y_n * nx + x_w];


        /* x-component of velocity */
        float u_x = (cells->s1[ii * nx + x_w]
                      + cells->s5[y_s * nx + x_w]
                      + cells->s8[y_n * nx + x_w]
                      - (cells->s3[ii * nx + x_e]
                         + cells->s6[y_s * nx + x_e]
                         + cells->s7[y_n * nx + x_e]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells->s2[y_s * nx + jj]
                      + cells->s5[y_s * nx + x_w]
                      + cells->s6[y_s * nx + x_e]
                      - (cells->s4[y_n * nx + jj]
                         + cells->s7[y_n * nx + x_e]
                         + cells->s8[y_n * nx + x_w]))
                     / local_density;

        /* velocity squared */
        float u_sq = 3 * (u_x * u_x + u_y * u_y);

        

        tmp_cells->s0[ii * nx + jj] = cells->s0[ii * nx + jj]
                                                  + omega
                                                  * (local_density * 4.0f * val1 * ( 2.0f - u_sq)
                                                  - cells->s0[ii * nx + jj]);
        tmp_cells->s1[ii * nx + jj] = cells->s1[ii * nx + x_w]
                                                  + omega
                                                  *  (((local_density * val1 * (3.0f * u_x * ( 3.0f * u_x + 2.0f) + 2.0f - u_sq )))
                                                  - cells->s1[ii * nx + x_w]);
        tmp_cells->s3[ii * nx + jj] = cells->s3[ii * nx + x_e]
                                                  + omega
                                                  *  (((local_density * val1 * (3.0f * u_x * (3.0f * u_x - 2.0f) + 2.0f - u_sq )))
                                                  - cells->s3[ii * nx + x_e]);
        tmp_cells->s2[ii * nx + jj] = cells->s2[y_s * nx + jj]
                                                  + omega
                                                  *  (((local_density * val1 * (3.0f * u_y * (3.0f * u_y + 2.0f) + 2.0f - u_sq )))
                                                  - cells->s2[y_s * nx + jj]);
        tmp_cells->s4[ii * nx + jj] = cells->s4[y_n * nx + jj]
                                                  + omega
                                                  *  (((local_density * val1 * (3.0f * u_y * (3.0f * u_y - 2.0f) + 2.0f - u_sq )))
                                                  - cells->s4[y_n * nx + jj]);
        tmp_cells->s5[ii * nx + jj] = cells->s5[y_s * nx + x_w]
                                                  + omega
                                                  * (((local_density * val2 * (3.0f * (u_x + u_y) * (3.0f * (u_x + u_y) + 2.0f) + 2.0f - u_sq )))
                                                  - cells->s5[y_s * nx + x_w]);
        tmp_cells->s7[ii * nx + jj] = cells->s7[y_n * nx + x_e]
                                                  + omega
                                                  * (((local_density * val2 * (3.0f * (u_x + u_y) * (3.0f * (u_x + u_y) - 2.0f) + 2.0f - u_sq )))
                                                  - cells->s7[y_n * nx + x_e]);
        tmp_cells->s6[ii * nx + jj] = cells->s6[y_s * nx + x_e]
                                                  + omega
                                                  * (((local_density * val2 * (3.0f * (u_x - u_y) * (3.0f * (u_x - u_y) - 2.0f) + 2.0f - u_sq )))
                                                  - cells->s6[y_s * nx + x_e]);
        
        tmp_cells->s8[ii * nx + jj] = cells->s8[y_n * nx + x_w]
                                                  + omega
                                                  * (((local_density * val2 * (3.0f * (u_x - u_y) * (3.0f * (u_x - u_y) + 2.0f) + 2.0f - u_sq )))
                                                  - cells->s8[y_n * nx + x_w]);
    

        tot_u += sqrt((u_x * u_x) + (u_y * u_y));
  }

    // tmp_cells->s0[ii * nx + jj] = tmp_cells_local->s0[ii * nx + jj]];
    // tmp_cells->s1[ii * nx + jj] = tmp_cells_local->s1[ii * nx + jj]];
    // tmp_cells->s2[ii * nx + jj] = tmp_cells_local->s2[ii * nx + jj]];
    // tmp_cells->s3[ii * nx + jj] = tmp_cells_local->s3[ii * nx + jj]];
    // tmp_cells->s4[ii * nx + jj] = tmp_cells_local->s4[ii * nx + jj]];
    // tmp_cells->s5[ii * nx + jj] = tmp_cells_local->s5[ii * nx + jj]];
    // tmp_cells->s6[ii * nx + jj] = tmp_cells_local->s6[ii * nx + jj]];
    // tmp_cells->s7[ii * nx + jj] = tmp_cells_local->s7[ii * nx + jj]];
    // tmp_cells->s8[ii * nx + jj] = tmp_cells_local->s8[ii * nx + jj]];
    


  // --------------Local REDUCTION -----------------

  int num_wrk_items  = get_local_size(0) * get_local_size(1);   // # work-items in work-group 
  int local_id       = get_local_size(0) * get_local_id(1) + get_local_id(0);     // ID of work-item within work-group          
  int group_id       = get_num_groups(0) * get_group_id(1) + get_group_id(0);     // ID of work-group

  av_local_sums[local_id] = tot_u;

  reduce_local(av_local_sums, local_id, num_wrk_items);

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


  reduce_local(shared_mem, global_id, group_size);

  // // #pragma unroll 1
  // for (int i = group_size / 2; i > 0; i >>= 1) {  
  //     if (id < i){
  //         shared_mem[global_id] += shared_mem[global_id + i];
  //     }
  //     barrier(CLK_LOCAL_MEM_FENCE);
  // }

  if (global_id == 0){
      av_vels[tt] = shared_mem[0]/tot_cells;  
      // printf("%d\n", av_vels[tt]);                             
  }
}

void reduce_local(local float* shared_mem, int id, int group_size){
  barrier(CLK_LOCAL_MEM_FENCE);
  
  // if (id == 0){
  //   float total = 0.0f;
  //   for (int i=0; i<group_size; i++) {        
  //     total += shared_mem[i];             
  //   }                                     
  //   shared_mem[0] = total;    
  // } 

  // #pragma unroll 1
  for (int i = group_size / 2; i > 0; i >>= 1) {  
      if (id < i){
          shared_mem[id] += shared_mem[id + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
  }

  //do reduction in shared mem
  // if (blockSize >= 512) { if (id < 256) { shared_mem[id] += shared_mem[id + 256]; } barrier(CLK_LOCAL_MEM_FENCE); }
  // if (blockSize >= 256) { if (id < 128) { shared_mem[id] += shared_mem[id + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
  // if (blockSize >= 128) { if (id <  64) { shared_mem[id] += shared_mem[id +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }
    
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