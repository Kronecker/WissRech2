#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;
#define CUDA_THREAD_NUM 1024

// must be a multiply of 2




float* prefSumSeq(int n, float *a);  // -> Aufgabe a)
float* prefSumBinTree(int n, float* a) ;  // -> Aufgabe b)
float* prefSumBinTreeMulti(int n, float* a) ;   // -> Aufgabe d)
float* gilesmScan(int n, float* a);   // -> Aufgabe c)

__global__ void scan(float *d_data);
__global__ void prefSumBinTreeCuda(float* a, int n) ;
__global__ void prefSumBinTreeCudaMulti(float* a, int n) ;
//__global__ void prefSumBinTreeCudaMeijerAkl(float* a, int n) ;


//host code
int main() {
	int n=524288;n=CUDA_THREAD_NUM*5;
	float *a=new float[n];
	float *pSum;
	float *pSumSeq;
	for (int k=0;k<n;k++) {
	a[k]=k+1;
	}

	printf("Sequentiell\n_____________\n");
	pSumSeq=prefSumSeq(n,a);
	for(int k=0;k<15;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSumSeq[k]);
	}
	printf("\n\n");
	for(int k=n-10;k<n;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSumSeq[k]);
	}
	printf("\n\n");


	pSum=prefSumBinTreeMulti(n,a);
	printf("Tree_Multi\n_____________\n");
	for(int k=0;k<15;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSum[k]);
	}
	printf("\n\n");

	for(int k=n-10;k<n;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSum[k]);
	}
	printf("\n\n");
	delete(pSum);

	printf("Warp\n_____________\n");
	pSum=gilesmScan(n,a);
	for(int k=0;k<15;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSum[k]);
	}
	printf("\n\n");

	for(int k=n-10;k<n;k++) {
		printf("%d_%d : %f <  ",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSum[k]);
	}
	printf("\n\n");

	//for(int k=0;k<n;k++) {
	//printf("%d_%d : %f <> %f\n",k/CUDA_THREAD_NUM,k%CUDA_THREAD_NUM,pSumSeq[k],pSum[k]);
	//}

	delete(pSum);





	delete(pSumSeq);
	delete(a);


	return 0;
}

float* prefSumSeq(int n,float* a) {
	float* pSum=new float[n];
	pSum[0]=a[0];
	for(int k=1;k<n;k++) {
	pSum[k]=pSum[k-1]+a[k];
	}

	return pSum;
}
float* prefSumBinTree(int n, float* a) {

	float *pSum;
	float *cudaA;


	pSum=new float[n];

	cudaMalloc(&cudaA,(int)sizeof(float)*n);


	cudaMemcpy(cudaA, a, sizeof(float)*n, cudaMemcpyHostToDevice);


	// Invoke Kernel
	prefSumBinTreeCuda<<<1,CUDA_THREAD_NUM>>>(cudaA,n);

	cudaMemcpy(pSum,cudaA, sizeof(float)*n, cudaMemcpyDeviceToHost);

	cudaFree(cudaA);
	cudaThreadExit();


	return pSum;

}
float* prefSumBinTreeMulti(int n, float* a) {

	float *pSum;
	float *cudaA;
	float prefPSum;
	int numOfBlocks=n/CUDA_THREAD_NUM;
	pSum=new float[n];

	cudaMalloc(&cudaA,(int)sizeof(float)*n);


	cudaMemcpy(cudaA, a, sizeof(float)*n, cudaMemcpyHostToDevice);


	// Invoke Kernel
	prefSumBinTreeCudaMulti<<<numOfBlocks,CUDA_THREAD_NUM>>>(cudaA,n);

	cudaMemcpy(pSum,cudaA, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cudaFree(cudaA);
	cudaThreadExit();

	for(int k=1;k<numOfBlocks;k++) {
	   prefPSum=pSum[k*CUDA_THREAD_NUM-1];	
	   for(int j=0;j<CUDA_THREAD_NUM;j++) {
		pSum[k*CUDA_THREAD_NUM+j]+=prefPSum;
	   }
	   
	}
	return pSum;

}



//CUDA_THREAD_NUM
/* All following algos are based on an archived Princeton Lecture:
Math 18.337, Computer Science 6.338, SMA 5505, Spring 2004
Up the tree: Compute pairwise sums
Down the tree: even idx => value of parant node
	       odd  idx => add value of the node left of parent node to the current value
*/


__global__ void prefSumBinTreeCuda(float *a, int n) {
__shared__ float shm[CUDA_THREAD_NUM];
	int tid=threadIdx.x;
	int dot=2;//depth of tree

	if((tid+1)%dot==0) {
          shm[tid]=a[tid]+a[tid-1];	   	      
	}
	dot*=2;	
	__syncthreads();
	while(dot<=n)  {
	   if((tid+1)%dot==0) {
              shm[tid]=shm[tid]+shm[tid-dot/2];	   	      
	   }
	   dot*=2;
	   __syncthreads();	
	}
	dot/=2;
	while(dot>2) {
	   if((tid+1)%dot==0) {
	      if((tid+1)/dot!=1) {
                 shm[tid-dot/2]=shm[tid-dot/2]+shm[tid-dot];	   	  
              }
	   }
	   dot/=2;
	   __syncthreads();
	}

  	   if((tid+1)%2==0) {
		a[tid]=shm[tid];
	   } else if(tid>0) {	     
                 a[tid]=a[tid]+shm[tid-1];		 	   	  
              
           }

}

/*
	   if((tid+1)%dot==0) {
	      if((tid+1)/dot!=1) {
                 a[tid-dot/2]=a[tid-dot/2]+shm[tid-dot];
		 	   	  
              }
		a[tid]=shm[tid];
	   }
*/

__global__ void prefSumBinTreeCudaMulti(float *a, int n) {
__shared__ float shm[CUDA_THREAD_NUM];
	int tid=threadIdx.x;
	int bid=blockIdx.x;
	int dot=2;//depth of tree

	if((tid+1)%dot==0) {
          shm[tid]=a[CUDA_THREAD_NUM*bid+tid]+a[CUDA_THREAD_NUM*bid+tid-1];	   	      
	}
	dot*=2;	
	__syncthreads();
	while(dot<=n)  {
	   if((tid+1)%dot==0) {
              shm[tid]=shm[tid]+shm[tid-dot/2];	   	      
	   }
	   dot*=2;
	   __syncthreads();	
	}
	dot/=2;
	while(dot>2) {
	   if((tid+1)%dot==0) {
	      if((tid+1)/dot!=1) {
                 shm[tid-dot/2]=shm[tid-dot/2]+shm[tid-dot];	   	  
              }
	   }
	   dot/=2;
	   __syncthreads();
	}
	
 	   if((tid+1)%2==0) {
		a[CUDA_THREAD_NUM*bid+tid]=shm[tid];
	   } else if(tid>0) {	     
                 a[CUDA_THREAD_NUM*bid+tid]=a[CUDA_THREAD_NUM*bid+tid]+shm[tid-1];		 	   	  
              
           }
	
}
__global__ void prefSumBinTreeCudaMultiCollect(float *a, int n, int numBlocks) {
__shared__ float shm[CUDA_THREAD_NUM];
	int tid=threadIdx.x;
	int dot=2;//depth of tree

	if((tid+1)%dot==0) {
          shm[tid]=a[tid]+a[tid-1];	   	      
	}
	dot*=2;	
	__syncthreads();
	while(dot<=n)  {
	   if((tid+1)%dot==0) {
              shm[tid]=shm[tid]+shm[tid-dot/2];	   	      
	   }
	   dot*=2;
	   __syncthreads();	
	}
	dot/=2;
	while(dot>2) {
	   if((tid+1)%dot==0) {
	      if((tid+1)/dot!=1) {
                 shm[tid-dot/2]=shm[tid-dot/2]+shm[tid-dot];	   	  
              }
	   }
	   dot/=2;
	   __syncthreads();
	}
	   if((tid+1)%2==0) {
		a[tid]=shm[tid];
	   } else if(tid>0) {	     
                 a[tid]=a[tid]+shm[tid-1];		 	   	                
           }
}




float* gilesmScan(int n,float* a) {




	float *pSum;
	float *cudaA;
	float prefPSum;
	int cudaThreads=CUDA_THREAD_NUM;
	int numOfBlocks=n/cudaThreads;
	pSum=new float[n];
 

	cudaMalloc(&cudaA,(int)sizeof(float)*n);


	cudaMemcpy(cudaA, a, sizeof(float)*n, cudaMemcpyHostToDevice);


	// Invoke Kernel
	scan<<<numOfBlocks,cudaThreads>>>(cudaA);

	cudaMemcpy(pSum,cudaA, sizeof(float)*n, cudaMemcpyDeviceToHost);
	cudaFree(cudaA);
	cudaThreadExit();

	//for(int k=1;k<n;k++) printf(">>%d   %f ",k,pSum[k]);

/*	for(int k=1;k<numOfBlocks;k++) {

	   prefPSum=pSum[k*CUDA_THREAD_NUM-1];	
	   for(int j=0;j<CUDA_THREAD_NUM;j++) {
	   
		pSum[k*CUDA_THREAD_NUM+j]+=prefPSum;
	   }
	   printf("\n");
	}*/
	return pSum;




}




// from https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
__global__ void scan(float *d_data) {
	__shared__ float temp[32];
	float temp1=0, temp2=0, temp3=0;
	int tid = threadIdx.x;
	temp1   = d_data[tid+blockIdx.x*blockDim.x];

	for (int d=1; d<32; d<<=1) {
		  temp2 = __shfl_up(temp1,d);
		  if (tid%32 >= d) temp1 += temp2;
	}

	if (tid%32 == 31) temp[tid/32] = temp1;
	__syncthreads();


	if (tid < 32) {
		temp2 = 0.0f;
		if (tid < blockDim.x/32)
			temp2 = temp[tid];
		for (int d=1; d<32; d<<=1) {
			temp3 = __shfl_up(temp2,d);
			if (tid%32 >= d) temp2 += temp3;
		}
		if (tid < blockDim.x/32) temp[tid] = temp2;
	}
	__syncthreads();
	if (tid >= 32) temp1 += temp[tid/32 - 1];
	//printf("<> %f",temp[tid]);
	d_data[tid+blockIdx.x*blockDim.x]=temp[tid];

}





