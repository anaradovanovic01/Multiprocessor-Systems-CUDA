#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NUM_OF_GPU_THREADS 1024

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

/*double potential(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}*/

__device__ double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

__global__ void feyman(int* ni_gpu, int* nj_gpu, int* nk_gpu, int* N_gpu, double* stepsz_gpu, float* err, int* n_inside) {

  int ni = *ni_gpu;
  int nj = *nj_gpu;
  int nk = *nk_gpu;
  int N = *N_gpu;
  double stepsz = *stepsz_gpu;
  if(threadIdx.x + (blockDim.x * blockIdx.x) < ni*nj*nk) {
    err[threadIdx.x + (blockDim.x * blockIdx.x)] = 0.0;
    n_inside[threadIdx.x + (blockDim.x * blockIdx.x)] = 0;
  }
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  double dx;
  double dy;
  double dz;
  double h = 0.001;
  int i;
  int j;
  int k;
  int steps;
  int steps_ave;
  int trial;
  double us;
  double ut;
  double vh;
  double vs;
  double x;
  double x1;
  double x2;
  double x3;
  double y;
  double w;
  double w_exact;
  double we;
  double wt;
  double z;
  int seed = 123456789;

  i = ((threadIdx.x + (blockDim.x * blockIdx.x) ) / (nj * nk) ) % ni + 1;
  j = ((threadIdx.x + (blockDim.x * blockIdx.x) ) /  nk ) % nj + 1;
  k = (threadIdx.x + (blockDim.x * blockIdx.x) ) % nk + 1;
  x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);
  y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);
  z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);
  
  if(threadIdx.x + (blockDim.x * blockIdx.x) < ni*nj*nk){
     chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

          return;
        }

        n_inside[threadIdx.x + (blockDim.x * blockIdx.x)] += 1;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;
        for (trial = 0; trial < N; trial++)
        {
          x1 = x;
          x2 = y;
          x3 = z;
          w = 1.0;
          chk = 0.0;
          while (chk < 1.0)
          {
            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dx = -stepsz;
              else
                dx = stepsz;
            }
            else
              dx = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dy = -stepsz;
              else
                dy = stepsz;
            }
            else
              dy = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
              us = r8_uniform_01(&seed) - 0.5;
              if (us < 0.0)
                dz = -stepsz;
              else
                dz = stepsz;
            }
            else
              dz = 0.0;

            vs = 2.0 * (pow(x1 / a / a, 2) + pow(x2 / b / b, 2) + pow(x3 / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
            x1 = x1 + dx;
            x2 = x2 + dy;
            x3 = x3 + dz;

            steps++;

            vh = 2.0 * (pow(x1 / a / a, 2) + pow(x2 / b / b, 2) + pow(x3 / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;

            we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = pow(x1 / a, 2) + pow(x2 / b, 2) + pow(x3 / c, 2);
          }
          wt = wt + w;
        }
        wt = wt / (double)(N);
        steps_ave = steps / (double)(N);

        err[threadIdx.x + (blockDim.x * blockIdx.x)] += pow(w_exact - wt, 2);
        
  }

}

// print na stdout upotrebiti u validaciji paralelnog resenja
int main(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double stepsz;
  double h = 0.001;
  int dim = 3;
  int ni;
  int nj;
  int nk;
  int* ni_gpu;
  int* nj_gpu;
  int* nk_gpu;
  int* N_gpu;
  double* stepsz_gpu;
  
  int N = atoi(argv[1]);
  timestamp();

  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t stop = cudaEvent_t();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c))
  {
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c))
  {
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else
  {
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }
  const int size = ni*nj*nk;

  float err[2*NUM_OF_GPU_THREADS], *err_gpu;
  int n_inside[2*NUM_OF_GPU_THREADS], *n_inside_gpu;

  cudaMalloc(&ni_gpu, sizeof(int));
  cudaMalloc(&nj_gpu, sizeof(int));
  cudaMalloc(&nk_gpu, sizeof(int));
  cudaMalloc(&N_gpu, sizeof(int));
  cudaMalloc(&stepsz_gpu, sizeof(double));
  cudaMalloc(&n_inside_gpu, size*sizeof(int));
  cudaMalloc(&err_gpu, size*sizeof(double));
  
  cudaMemcpy(ni_gpu, &ni, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nj_gpu, &nj, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nk_gpu, &nk, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(N_gpu, &N, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(stepsz_gpu, &stepsz, sizeof(double), cudaMemcpyHostToDevice);

  feyman<<<(2), (NUM_OF_GPU_THREADS)>>>(ni_gpu, nj_gpu, nk_gpu, N_gpu, stepsz_gpu, err_gpu, n_inside_gpu);
  
  cudaMemcpy(n_inside, n_inside_gpu, size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(err, err_gpu, size*sizeof(double), cudaMemcpyDeviceToHost);

  int n_inside_sum = 0;
  double err_sum = 0;

  for(int i=0; i < size; i++){
    n_inside_sum += n_inside[i];
    err_sum += err[i];
  }

  err_sum = sqrt(err_sum / (double)(n_inside_sum));

  printf("\n\nRMS absolute error in solution = %e\n", err_sum);
  timestamp();

  cudaFree(ni_gpu);
  cudaFree(nj_gpu);
  cudaFree(nk_gpu);
  cudaFree(n_inside_gpu);
  cudaFree(err_gpu);
  cudaFree(N_gpu);
  cudaFree(stepsz_gpu);

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsed = 0.f;
  cudaEventElapsedTime(&elapsed, start, stop);

  printf("Time: %0.2f\n", elapsed);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
