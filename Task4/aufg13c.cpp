//
// Created by grabiger on 08.06.2018.
//

using namespace std;

flouble* jacobiIterCuda_1Core_CPU(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h);
__global__ void initMatrixRightHandSideCuda_1Core_CUDA(flouble h, flouble* matrix, int n);
__global__ void initSolutionVectors_1Core_CUDA(flouble *actualIteration, flouble valBoundary, int n);
__global__ void jacoboIteration_1Core_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                           flouble valMainDiag, flouble *f,flouble *resi ,int *iteration);
__device__ void calculateResidual_1Core_CUDA(double *a, double *b, double *c, int n);
__device__ void calculateResidual_1Core_CUDA(float *a, float *b, float *c, int n);



void aufg13c() {

    int n=1024;
    int nn=n*n;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;
    flouble *cuda_fun;
    cudaMalloc(&cuda_fun,sizeof(flouble)*nn);

    flouble *result=new flouble[nn];

    int doneIterations=0;

    initMatrixRightHandSideCuda_1Core_CUDA<<<1,n>>>(h,cuda_fun,n);

    result=jacobiIterCuda_CPU(n, cuda_fun, boundaryValue, &doneIterations,h);
    cudaThreadExit();

    saveMyMatrix(result, n,n,h,2);

}





flouble* jacobiIterCuda_1Core_CPU(int n, flouble *cudaF, flouble valBoundary, int* numberOfIterations, flouble h) {

    int nn=n*n;
    flouble* actualIteration=new flouble[nn]();


    flouble *cuda_actualIteration, *cuda_lastIterSol;
    cudaMalloc(&cuda_actualIteration,sizeof(flouble)*nn);;
    cudaMalloc(&cuda_lastIterSol,sizeof(flouble)*nn);;

    initSolutionVectors_CUDA <<<n,n>>> (cuda_actualIteration, valBoundary);
  // a


    flouble resi=1;
    flouble *resi_Cuda;
    cudaMalloc(&resi_Cuda,sizeof(flouble));
    int *itas;
    cudaMalloc(&itas,sizeof(int));


    flouble hsquare=h*h;
    flouble valSubDiag=-1/hsquare;
    flouble valMainDiag=4/hsquare;

    jacoboIteration_1Core_CUDA<<<1,n>>>(cuda_actualIteration,cuda_lastIterSol,n,valSubDiag,valMainDiag,cudaF,resi_Cuda, itas);

    cudaMemcpy(numberOfIterations,itas,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(resi,resi_Cuda,sizeof(flouble),cudaMemcpyDeviceToHost);




    std::cout << "Calculation finished after "<<*numberOfIterations<<" Iterations.(%"<<step<<")"<<std::endl;
    cudaMemcpy(actualIteration,cuda_actualIteration, sizeof(flouble)*nn, cudaMemcpyDeviceToHost);

    return actualIteration;

}

__global__ void initMatrixRightHandSideCuda_1Core_CUDA(flouble h, flouble* matrix, int n) {
    // Version for n==1024
    int tid=threadIdx.x;
    int bid;

    for(bid=0;bid<n;bid++) {
        flouble x=h*bid;
        flouble y=h*tid;
        matrix[bid * blockDim.x + tid] = x * (1 - x) + y * (1 - y);

    }
    calculateResidual_1Core_CUDA(matrix,matrix,matrix,1);
}

__global__ void initSolutionVectors_1Core_CUDA(flouble *actualIteration, flouble valBoundary, int n) {
    int tid = threadIdx.x;

    actualIteration[tid] = valBoundary;
    actualIteration[n * (n-1) + tid] = valBoundary;

    for(int bid=1;bid<n-1;bid++) {
            if ((tid == 0) || tid == n - 1) {
                actualIteration[n * bid + tid] = valBoundary;
            } else {
                actualIteration[bid * n + tid] = 0;
            }

    }
}



__global__ void jacoboIteration_1Core_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                     flouble valMainDiag, flouble *f,flouble *resi ,int *iteration) {
    *iteration=0;
    *resi=0;
    int index;  //index=k*n+i;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int step=100;
    flouble tol=0.0001;
    flouble *temp;
    int maxIter=2000;


    while(*iteration<maxIter) {
        for (bid = 1; bid < n - 1; bid++) {

            if (tid == 0 || tid == n - 1) {  // Boundaries, nothing to do here
                continue;
            }

            index = bid * n + tid;
            actualIteration[index] = 1 / valMainDiag *
                                     (f[index] - valSubDiag * lastIterSol[index - n] -
                                      valSubDiag * lastIterSol[index - 1] -
                                      valSubDiag * lastIterSol[index + 1] - valSubDiag * lastIterSol[index + n]);

        }


        if (tid == 0) {
            *iteration=*iteration+1;
        }
        __syncthreads();
        temp=lastIterSol;
        lastIterSol=actualIteration;
        actualIteration=temp;
        if (*iteration % step == 0) {
            calculateResidual_CUDA(actualIteration, lastIterSol, resi, n);
            if (*resi < tol) {
                break;
            }
            *resi = 0;
        }
    }


}


__device__ void calculateResidual_1Core_CUDA(float *a, float *b, float *c, int n) {
    __shared__ float se[1024];

    int tid=threadIdx.x;
    int bid;



    for(bid=0;bid<n;bid++) {
        //   Calculate
        se[tid] = fabsf(a[tid + bid * n] - b[tid + bid * n]);
        __syncthreads();

        //   Reducto
        int numActiveThreads = n / 2;
        while (numActiveThreads > 0) {
            if (tid < numActiveThreads) {
                se[tid] = se[tid] + se[tid + numActiveThreads];
            }
            numActiveThreads = numActiveThreads / 2;
            __syncthreads();
        }


        if (tid == 0) {
            atomicAdd(c, se[0]);
        }
    }
}

__device__ void calculateResidual_1Core_CUDA(double *a, double *b, double *c, int n) {
    __shared__ float se[1024];

    int tid=threadIdx.x;
    int bid;



    for(bid=0;bid<n;bid++) {
        //   Calculate
        se[tid] = fabsf(a[tid + bid * n] - b[tid + bid * n]);
        __syncthreads();

        //   Reducto
        int numActiveThreads = n / 2;
        while (numActiveThreads > 0) {
            if (tid < numActiveThreads) {
                se[tid] = se[tid] + se[tid + numActiveThreads];
            }
            numActiveThreads = numActiveThreads / 2;
            __syncthreads();
        }


        if (tid == 0) {
            atomicAdd(c, se[0]);
        }
    }
}























































































