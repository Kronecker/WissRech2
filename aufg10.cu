#include <stdio.h>
#include <cuda.h>
//device code
__global__ void helloCUDA(float f) { //beliebige Argumente möglich
printf("ThreadId: %d\tProzessId: %d\n", threadIdx.x, blockIdx.x);
}
//host code
int main() {
helloCUDA<<<15, 32>>>(1.2345f); // 15 Prozesse zu je 32 Threads
cudaThreadExit();
return 0;
}
