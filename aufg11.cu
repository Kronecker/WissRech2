#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;
#define CUDA_THREAD_NUM 1024
// must be a multiply of 2



void dotProductCPU();
__global__ void dotProductCuda(float *a, float *b, float *c);


//host code
int main() {

dotProductCPU();
cudaThreadExit();
return 0;
}
void dotProductCPU() {
int n=524288;
float *a,*b,*c;
float *cudaA,*cudaB,*cudaC, sumSeq=0,sumCuda=0;


int numBlocks=n/CUDA_THREAD_NUM;
cudaMalloc(&cudaA,(int)sizeof(float)*n);
cudaMalloc(&cudaB,(int)sizeof(float)*n);
cudaMalloc(&cudaC,(int)sizeof(float)*numBlocks);
a=new float[n];
b=new float[n];
c=new float[numBlocks];
for(int i=0;i<n;i++) {
a[i]=i+1;
b[i]=n-i;
sumSeq+=a[i]*b[i];
}
cudaMemcpy(cudaA,a,sizeof(float)*n,cudaMemcpyHostToDevice);
cudaMemcpy(cudaB,b,sizeof(float)*n,cudaMemcpyHostToDevice);



dotProductCuda<<<numBlocks,CUDA_THREAD_NUM>>>(cudaA,cudaB,cudaC);

cudaMemcpy(c,cudaC,sizeof(float)*numBlocks,cudaMemcpyDeviceToHost);


cudaFree(cudaA);
cudaFree(cudaB);
cudaFree(cudaC);

for(int k=0;k<numBlocks;k++) {
	sumCuda+=c[k];
}

std::cout<<sumSeq<<std::endl;
std::cout<<sumCuda<<std::endl;

delete(a);
delete(b);
delete(c);
printf("All done");
} 

//device code

__global__ void dotProductCuda(float *a, float *b, float *c) {
__shared__ float se[CUDA_THREAD_NUM];

// Calculate a.*b
se[threadIdx.x]=a[threadIdx.x+blockIdx.x*CUDA_THREAD_NUM]*b[threadIdx.x+blockIdx.x*CUDA_THREAD_NUM];
__syncthreads();

// Sum Reducto
int numActiveThreads=CUDA_THREAD_NUM/2;
while(numActiveThreads>0) {
   if(threadIdx.x<numActiveThreads) {
      se[threadIdx.x]=se[threadIdx.x]+se[threadIdx.x+numActiveThreads];
   }
   numActiveThreads=numActiveThreads/2;
   __syncthreads();
}


if(threadIdx.x==0) {
   c[blockIdx.x]=se[0];
//printf("BlockId: %d,  ThreadID: %d,  %f \n",blockIdx.x,threadIdx.x,c[blockIdx.x]);
}

return;
}



