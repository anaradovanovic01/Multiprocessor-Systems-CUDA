#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define NUM_OF_GPU_THREADS 1024

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

__global__ void prime_number(int* n, int* total)
{
  int j;
  int prime;

  int i = threadIdx.x + 2;
  int total_sum = 0;

  while(i <= *n)
  {
    prime = 1;
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    total_sum += prime;
    i += NUM_OF_GPU_THREADS;
  }
  total[threadIdx.x] = total_sum;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor);

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

  timestamp();
  printf("\n");
  printf("PRIME TEST\n");

  if (argc != 4)
  {
    n_lo = 1;
    n_hi = 131072;
    n_factor = 2;
  }
  else
  {
    n_lo = atoi(argv[1]);
    n_hi = atoi(argv[2]);
    n_factor = atoi(argv[3]);
  }

  test(n_lo, n_hi, n_factor);

  printf("\n");
  printf("PRIME_TEST\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();

  return 0;
}

void test(int n_lo, int n_hi, int n_factor)
{
  int n;
  double ctime;
  int* n_gpu;
  int* total_gpu;
  int total[NUM_OF_GPU_THREADS];
  int total_sum;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");
  
  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t stop = cudaEvent_t();
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  n = n_lo;

  cudaMalloc(&n_gpu, sizeof(int));

  cudaMalloc(&total_gpu, NUM_OF_GPU_THREADS * sizeof(int));

  while (n <= n_hi)
  {
    total_sum = 0;

    ctime = cpu_time();

    dim3 gridDim(1);
    dim3 blockDim(NUM_OF_GPU_THREADS);

    cudaMemcpy(n_gpu, &n, sizeof(int), cudaMemcpyHostToDevice);

    prime_number<<< gridDim, blockDim >>>(n_gpu, total_gpu);

    cudaMemcpy(&total, total_gpu,NUM_OF_GPU_THREADS *  sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < NUM_OF_GPU_THREADS; i++) {
      total_sum += total[i];
    }

    ctime = cpu_time() - ctime;

    printf("  %8d  %8d  %14f\n", n, total_sum, ctime);
    n = n * n_factor;
  }

  cudaFree(n_gpu);
  cudaFree(total_gpu);

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);

  float elapsed = 0.f;
  cudaEventElapsedTime(&elapsed, start, stop);

  printf("Time: %0.2f\n", elapsed);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return;
}
