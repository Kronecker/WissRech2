//
// Created by grabiger on 08.06.2018.
//

using namespace std;

flouble* jacobiIterCuda_1Core_CPU(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h);
__global__ void initMatrixRightHandSideCuda_1Core_CUDA(flouble h, flouble* matrix);
__global__ void initSolutionVectors_1Core_CUDA(flouble *actualIteration, flouble valBoundary, int n);
__global__ void jacoboIteration_1Core_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                     flouble valMainDiag, flouble *f);
__global__ void calculateResidual_1Core_CUDA(double *a, double *b, double *c);
__global__ void calculateResidual_1Core_CUDA(float *a, float *b, float *c);



void aufg13c() {

    int n=1024;
    int nn=n*n;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;
    flouble *cuda_fun;
    cudaMalloc(&cuda_fun,sizeof(flouble)*nn);

    flouble *result=new flouble[nn];

    int doneIterations=0;

    initMatrixRightHandSideCuda_CUDA<<<1,n>>>(h,cuda_fun,n);
    cudaMemcpy(result, cuda_fun,sizeof(flouble)*n*n,cudaMemcpyDeviceToHost)
  //  result=jacobiIterCuda_CPU(n, cuda_fun, boundaryValue, &doneIterations,h);
    cudaThreadExit();

    saveMyMatrix(result, n,n,h,1);

}





flouble* jacobiIterCuda_1Core_CPU(int n, flouble *cudaF, flouble valBoundary, int* numberOfIterations, flouble h) {

    int nn=n*n;
    flouble* actualIteration=new flouble[nn]();


    flouble *cuda_actualIteration, *cuda_lastIterSol;
    cudaMalloc(&cuda_actualIteration,sizeof(flouble)*nn);;
    cudaMalloc(&cuda_lastIterSol,sizeof(flouble)*nn);;

    initSolutionVectors_CUDA <<<n,n>>> (cuda_actualIteration, valBoundary);

    flouble tol=0.0001;
    int iteration=0;
    flouble resi=tol+1;
    flouble *resiCuda;
    cudaMalloc(&resiCuda,sizeof(flouble));
    int step=100;  // 2 Iterations
    int maxDoubleIter=MAXITERATIONS/2;

    flouble hsquare=h*h;
    flouble valSubDiag=-1/hsquare;
    flouble valMainDiag=4/hsquare;


    while(iteration<maxDoubleIter) {
        // consecutive blocks

        jacoboIteration_CUDA <<<n,n>>>(cuda_actualIteration,cuda_lastIterSol,n,valSubDiag,valMainDiag,cudaF);
        jacoboIteration_CUDA <<<n,n>>>(cuda_lastIterSol,cuda_actualIteration,n,valSubDiag,valMainDiag,cudaF);
        iteration++;
        if(iteration%step==0) {
            calculateResidual_CUDA <<<n,n>>>(cuda_actualIteration, cuda_lastIterSol, resiCuda);
            cudaMemcpy(&resi,resiCuda,sizeof(flouble),cudaMemcpyDeviceToHost);

            cout<<iteration*2<<": "<<resi<<endl;
            if(resi<tol) {
                break;
            }
            resi=0;  // Reset resiCuda.....is there any better way?
            cudaMemcpy(resiCuda,&resi,sizeof(flouble),cudaMemcpyHostToDevice);
        }
    }
    std::cout << "Calculation finished after "<<2*iteration<<" Iterations.(%"<<step<<")"<<std::endl;
    *numberOfIterations=iteration*2;
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

}

__global__ void initSolutionVectors_1Core_CUDA(flouble *actualIteration, flouble valBoundary) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int n = blockDim.x;

    if ((bid == 0)||(bid == n-1)) {  // boundary values init (outer)
        actualIteration[n * bid + tid] = valBoundary;
    } else {
        if((tid==0)||tid==n-1) {
            actualIteration[n * bid + tid] = valBoundary;
        }else {
            actualIteration[bid*n+tid] = 0;
        }
    }
}



__global__ void jacoboIteration_1Core_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                     flouble valMainDiag, flouble *f) {
    int index;  //index=k*n+i;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int bdim=blockDim.x;

    if(bid==0||bid==gridDim.x-1) {  // Boundaries, nothing to do here
        return;
    }
    if(tid==0||tid==gridDim.x-1) {  // Boundaries, nothing to do here
        return;
    }

    index=bid*bdim+tid;
    actualIteration[index]=1/valMainDiag*(f[index]-valSubDiag*lastIterSol[index-bdim]-valSubDiag*lastIterSol[index-1]-valSubDiag*lastIterSol[index+1]-valSubDiag*lastIterSol[index+bdim]);
}


__global__ void calculateResidual_1Core_CUDA(float *a, float *b, float *c) {
    __shared__ float se[1024];

    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int n=blockDim.x;
    //   Calculate
    se[tid]=fabsf(a[tid+bid*n]-b[tid+bid*n]);
    __syncthreads();

    //   Reducto
    int numActiveThreads=n/2;
    while(numActiveThreads>0) {
        if(tid<numActiveThreads) {
            se[tid]=se[tid]+se[tid+numActiveThreads];
        }
        numActiveThreads=numActiveThreads/2;
        __syncthreads();
    }


    if(tid==0) {
        atomicAdd(c,se[0]);
    }
}

__global__ void calculateResidual_1Core_CUDA(double *a, double *b, double *c) {
    __shared__ double se[1024];

    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int n=blockDim.x;

    // Calculate a.*b
    se[tid]=fabsf(a[tid+bid*n]-b[tid+bid*n]);
    __syncthreads();

    // Sum Reducto
    int numActiveThreads=n/2;
    while(numActiveThreads>0) {
        if(tid<numActiveThreads) {
            se[tid]=se[tid]+se[tid+numActiveThreads];
        }
        numActiveThreads=numActiveThreads/2;
        __syncthreads();
    }


    if(tid==0) {
        atomicAdd(c,se[0]);
    }
}
























































































