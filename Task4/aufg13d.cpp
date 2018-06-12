//
// Created by grabiger on 08.06.2018.
//

using namespace std;

void aufg13b();
flouble* jacobiIterCuda_MultiGPU_CPU(int n, flouble valBoundary, int* numberOfIterations, flouble h);
__global__ void initMatrixRightHandSideCuda_MultiGPU_CUDA(flouble h, flouble* matrix, int offset);
__global__ void initSolutionVectors_MultiGPU_CUDA(flouble *actualIteration, flouble valBoundary, int n, int offset);
__global__ void jacoboIteration_MultiGPU_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                     flouble valMainDiag, flouble *f);
__global__ void calculateResidual_MultiGPU_CUDA(double *a, double *b, double *c);
__global__ void calculateResidual_MultiGPU_CUDA(float *a, float *b, float *c);




void aufg13d() {
    // Init Chrono
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed;

    start = std::chrono::high_resolution_clock::now();

    int n=1024;
    int nn=n*n;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;


    flouble *result;

    int doneIterations=0;


    result=jacobiIterCuda_MultiGPU_CPU(n, boundaryValue, &doneIterations,h);
    cudaThreadExit();
    finish = std::chrono::high_resolution_clock::now();
    elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start);

    cout<< "Jacobi Iteration mit zwei GPU's(multicore): "<< elapsed.count() * 1000 << "ms"<<endl;

    saveMyMatrix(result, n,n,h,3);
}

flouble* jacobiIterCuda_MultiGPU_CPU(int n, flouble valBoundary, int* numberOfIterations, flouble h) {

    int nn=n*n;
    int m=(nn/2+n);
    flouble* actualIteration=new flouble[nn]();


    flouble *cuda_actualIterationD0, *cuda_lastIterSolD0, *temp;
    flouble *cuda_actualIterationD1, *cuda_lastIterSolD1;
    cudaSetDevice(0);
    cudaMalloc(&cuda_actualIterationD0,sizeof(flouble)*m);
    cudaMalloc(&cuda_lastIterSolD0,sizeof(flouble)*m);
    cudaSetDevice(1);
    cudaMalloc(&cuda_actualIterationD1,sizeof(flouble)*m);
    cudaMalloc(&cuda_lastIterSolD1,sizeof(flouble)*m);

    flouble *cuda_funD0,*cuda_funD1;

    cudaSetDevice(0);
    cudaMalloc(&cuda_funD0,sizeof(flouble)*m);
    cudaSetDevice(1);
    cudaMalloc(&cuda_funD1,sizeof(flouble)*m);

    cudaSetDevice(0);
    initMatrixRightHandSideCuda_MultiGPU_CUDA<<<n/2+1,n>>>(h,cuda_funD0,0);
    cudaSetDevice(1);
    initMatrixRightHandSideCuda_MultiGPU_CUDA<<<n/2+1,n>>>(h,cuda_funD1,n/2-1);
    cudaSetDevice(1);


    cudaSetDevice(0);
    initSolutionVectors_MultiGPU_CUDA <<<n/2+1,n>>> (cuda_lastIterSolD0, valBoundary,n,0);
    cudaSetDevice(1);
    initSolutionVectors_MultiGPU_CUDA <<<n/2+1,n>>> (cuda_lastIterSolD1, valBoundary,n,n/2-1);

//    cudaSetDevice(0);
//    cudaMemcpy(actualIteration,cuda_actualIterationD0,sizeof(flouble)*m,cudaMemcpyDeviceToHost);
//    saveMyMatrix(actualIteration, n/2+1,n,1,0);
//
//    cudaSetDevice(1);
//    cudaMemcpy(actualIteration,cuda_actualIterationD1,sizeof(flouble)*m,cudaMemcpyDeviceToHost);
//    saveMyMatrixAppend(actualIteration, n/2+1,n,1,1,n/2-1);



//    cudaSetDevice(0);
//    cudaMemcpy(actualIteration,cuda_actualIterationD0,sizeof(flouble)*m,cudaMemcpyDeviceToHost);
//    saveMyMatrix(actualIteration, n/2+1,n,1,3);


    flouble tol=0.0001;
    int iteration=0;
    flouble resi=tol+1;
    flouble *resiCudaD0,*resiCudaD1;

    cudaSetDevice(0);
    cudaMalloc(&resiCudaD0,sizeof(flouble));
    cudaSetDevice(1);
    cudaMalloc(&resiCudaD1,sizeof(flouble));

    int step=100;  // 2 Iterations

    flouble hsquare=h*h;
    flouble valSubDiag=-1/hsquare;
    flouble valMainDiag=4/hsquare;





    while(iteration<MAXITERATIONS) {
        // consecutive blocks
        cudaSetDevice(0);
        jacoboIteration_MultiGPU_CUDA <<<n/2+1,n>>>(cuda_actualIterationD0,cuda_lastIterSolD0,n,valSubDiag,valMainDiag,cuda_funD0);
        cudaSetDevice(1);
        jacoboIteration_MultiGPU_CUDA <<<n/2+1,n>>>(cuda_actualIterationD1,cuda_lastIterSolD1,n,valSubDiag,valMainDiag,cuda_funD1);

//        cudaSetDevice(0);
//        cudaMemcpy(actualIteration,&cuda_lastIterSolD0[m-2*n], sizeof(flouble)*n, cudaMemcpyDeviceToHost);
//        cudaSetDevice(1);
//        cudaMemcpy(cuda_lastIterSolD1,actualIteration, sizeof(flouble)*n, cudaMemcpyHostToDevice);
//        cudaMemcpy(actualIteration,&cuda_lastIterSolD1[n], sizeof(flouble)*n, cudaMemcpyDeviceToHost);
//        cudaSetDevice(0);
//        cudaMemcpy(&cuda_lastIterSolD0[m-n],actualIteration, sizeof(flouble)*n, cudaMemcpyHostToDevice);

          cudaSetDevice(0);
          cudaMemcpy(cuda_actualIterationD1,&cuda_actualIterationD0[m-2*n], sizeof(flouble)*n, cudaMemcpyDeviceToDevice);
          cudaSetDevice(1);
          cudaMemcpy(&cuda_actualIterationD0[m-n],&cuda_actualIterationD1[n], sizeof(flouble)*n, cudaMemcpyDeviceToDevice);



        // Swap
        temp=cuda_actualIterationD0;
        cuda_actualIterationD0=cuda_lastIterSolD0;
        cuda_lastIterSolD0=temp;

        temp=cuda_actualIterationD1;
        cuda_actualIterationD1=cuda_lastIterSolD1;
        cuda_lastIterSolD1=temp;


        iteration++;
        if(false&&iteration%step==0) {// War nicht gefragt, und Häufigkeit kann Geschwindigkeitsvergleich beeinflussen
       /*     calculateResidual_MultiGPU_CUDA <<<n,n>>>(cuda_actualIteration, cuda_lastIterSol, resiCuda);
            cudaMemcpy(&resi,resiCuda,sizeof(flouble),cudaMemcpyDeviceToHost);

            cout<<iteration*2<<": "<<resi<<endl;
            if(resi<tol) {
                break;
            }
            resi=0;  // Reset resiCuda.....is there any better way?
            cudaMemcpy(resiCuda,&resi,sizeof(flouble),cudaMemcpyHostToDevice);*/
        }
    }
  //  std::cout << "Calculation finished after "<<iteration<<" Iterations.(%"<<step<<")"<<std::endl;
    *numberOfIterations=iteration;

   cudaSetDevice(0);
   cudaMemcpy(actualIteration,cuda_lastIterSolD0,sizeof(flouble)*m,cudaMemcpyDeviceToHost);
 //  saveMyMatrix(actualIteration, n/2,n,1,3);

   cudaSetDevice(1);
   cudaMemcpy(&actualIteration[m-n],&cuda_lastIterSolD1[n],sizeof(flouble)*(m-n),cudaMemcpyDeviceToHost);
   //saveMyMatrixAppend(&actualIteration[n],n/2,n,1,3,n/2);

    // saveMyMatrix(actualIteration, n/2,n,1,3);

    return actualIteration;

}
/*
 *         cudaSetDevice(0);
        jacoboIteration_MultiGPU_CUDA <<<n/2+1,n>>>(cuda_lastIterSolD0,cuda_actualIterationD0,n,valSubDiag,valMainDiag,cuda_funD0);
        cudaSetDevice(1);
        jacoboIteration_MultiGPU_CUDA <<<n/2+1,n>>>(cuda_lastIterSolD1,cuda_actualIterationD1,n,valSubDiag,valMainDiag,cuda_funD1);

 */

__global__ void initMatrixRightHandSideCuda_MultiGPU_CUDA(flouble h, flouble* matrix, int offset) {
    // Version for n==1024
    int tid=threadIdx.x;
    int bid=blockIdx.x;

    flouble x=h*(bid+offset);
    flouble y=h*tid;
    matrix[bid*blockDim.x+tid]=x*(1-x)+y*(1-y);

}

__global__ void initSolutionVectors_MultiGPU_CUDA(flouble *actualIteration, flouble valBoundary, int n, int offset) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int threads= blockDim.x;
    int blockId=bid+offset;


    if ((blockId == 0)||(blockId == n-1)) {  // boundary values init (outer)
        actualIteration[threads * bid + tid] = valBoundary;
    } else {
        if((tid==0)||tid==threads-1) {
            actualIteration[threads * bid + tid] = valBoundary;
        }else {
            actualIteration[bid*threads+tid] = 0;
        }
    }
}



__global__ void jacoboIteration_MultiGPU_CUDA(flouble *actualIteration, flouble *lastIterSol, int n, flouble valSubDiag,
                                     flouble valMainDiag, flouble *f) {
    int index;  //index=k*n+i;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int bdim=blockDim.x;

    if(bid==0||bid==gridDim.x-1) {  // Boundaries, nothing to do here
        return;
    }
    if(tid==0||tid==n-1) {  // Boundaries, nothing to do here
        return;
    }

    index=bid*bdim+tid;
    actualIteration[index]=1/valMainDiag*(f[index]-valSubDiag*lastIterSol[index-bdim]-valSubDiag*lastIterSol[index-1]-valSubDiag*lastIterSol[index+1]-valSubDiag*lastIterSol[index+bdim]);
}


__global__ void calculateResidual_MultiGPU_CUDA(float *a, float *b, float *c) {
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

__global__ void calculateResidual_MultiGPU_CUDA(double *a, double *b, double *c) {
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




























