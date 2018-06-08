#include <iostream>
#include <math.h>
#include <fstream>
#include <cuda.h>

//#define flouble float
#define flouble double
#define MAXITERATIONS 2000


using namespace std;



void aufg13a();
flouble* initMatrixRightHandSide(int n, flouble h  );
flouble* jacobiIter(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h);

void aufg13b();
flouble* jacobiIterCuda_CPU(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h);
__global__ void initMatrixRightHandSideCuda_CUDA(flouble h, flouble* matrix);
__global__ void cuda_initSolutionVectors (flouble *actualIteration, flouble valBoundary)


void aufg13c();

void aufg13d();


// Utility
flouble* initMatrixKonstant(int m,int n, flouble value  ) ;
void displayMyMatrix(flouble* matrix, int m,int n);
void saveMyMatrix(flouble* matrix, int m,int n, flouble h);


int main() {
    std::cout << "Hello, World!" << std::endl;
    aufg13b();
    return 0;
}

void aufg13a() {

    int n=1024;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;
    flouble *fun;
    flouble *result;

    int doneIterations=0;


    fun=initMatrixRightHandSide(n,h);
    result=jacobiIter(n, fun, boundaryValue, &doneIterations,h);

    std::cout<<fun[1]<<std::endl;

   // displayMyMatrix(result,n,n);

    saveMyMatrix(result, n,n,h);

    delete(fun);
    delete(result);
}

flouble* initMatrixRightHandSide(int n, flouble h  ) {
    flouble*matrix=new flouble[n*n];
    flouble x;
    flouble y;
    for (int i=0;i<n;i++) {

        for (int j=0;j<n;j++) {
            x=h*i;
            y=h*j;
            matrix[i*n+j]=x*(1-x)+y*(1-y);
   //         printf("<%f %f> %f\n",x,y,matrix[i*m+j]);
        }
    }
    return matrix;
}

// zu Aufgabe 13a
flouble* jacobiIter(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h) {

    flouble* actualIteration=new flouble[n*n]();
    flouble* lastIterSol=new flouble[n*n]();
    flouble *temp;

    flouble tol=0.0001;
    int iteration=0;
    flouble resi=tol+1;
    int step=100;

    flouble hsquare=h*h;
    flouble valLowBlockDiag=-1/hsquare;
    flouble valUpBlockDiag=-1/hsquare;
    flouble valLowMinDiag=-1/hsquare;
    flouble valUpDiag=-1/hsquare;
    flouble valMainDiag=4/hsquare;



    // boundary values init (outer)
    for(int i=0;i<n;i++) {
        actualIteration[i]=valBoundary;
        lastIterSol[i]=valBoundary;
        actualIteration[n*(n-1)+i]=valBoundary;
        lastIterSol[n*(n-1)+i]=valBoundary;
    }
    for(int k=1;k<n-1;k++) { // iterate through blocks
        actualIteration[k*n]=valBoundary;
        lastIterSol[k*n]=valBoundary;
        actualIteration[(k+1)*n-1]=valBoundary;
        lastIterSol[(k+1)*n-1]=valBoundary;
    }


    int nm1=n-1;
    int index;
    while(iteration<MAXITERATIONS&&resi>tol) {
        // consecutive blocks

        for(int k=1;k<nm1;k++) { // iterate through blocks

            for(int i=1;i<nm1;i++) {  // iterate in block
                index=k*n+i;
                actualIteration[index]=1/valMainDiag*(f[index]-valLowBlockDiag*lastIterSol[index-n]-valLowMinDiag*lastIterSol[index-1]-valUpDiag*lastIterSol[index+1]-valUpBlockDiag*lastIterSol[index+n]);
            }

        }


        if (!(iteration % step)) {
            resi=0;
            for(int i=0;i<n*n;i++) {
                resi+=fabs(actualIteration[i]- lastIterSol[i]);
            }
            //   std::cout << iteration <<": "<< resi<< std::endl;
        }


        temp=lastIterSol;
        lastIterSol=actualIteration;
        actualIteration=temp;
        iteration++;


    }
    std::cout << "Calculation finished after "<<iteration<<" Iterations.(%"<<step<<")"<<std::endl;
    *numberOfIterations=iteration;

    delete(lastIterSol);
    return actualIteration;
}






__global__ void initMatrixRightHandSideCuda_CUDA(flouble h, flouble* matrix) {
    // Version for n==1024
    int tid=threadIdx.x;
    int bid=blockIdx.x;

    flouble x=h*bid;
    flouble y=h*tid;
    matrix[bid*blockDim.x+tid]=x*(1-x)+y*(1-y);

}



// _________________________________________________________________________________________ //
//
//                                Aufgabe b
// _________________________________________________________________________________________ //




flouble* jacobiIterCuda_CPU(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h) {

    int nn=n*n;
    flouble* actualIteration=new flouble[nn]();


    flouble *cuda_actualIteration, *cuda_lastIterSol;
    cudaMalloc(&cuda_actualIteration,sizeof(flouble)*nn);;
    cudaMemset(cuda_actualIteration, 0, nn*sizeof(flouble));
    cudaMalloc(&cuda_lastIterSol,sizeof(flouble)*nn);;

    cuda_initSolutionVectors<<<n,n>>> (cuda_actualIteration, valBoundary);

    cudaMemCopy(actualIteration,cuda_actualIteration,n*sizeof(flouble),cudaMemcpyDeviceToHost);
//
//    flouble tol=0.0001;
//    int iteration=0;
//    flouble resi=tol+1;
//    int step=100;
//
//    flouble hsquare=h*h;
//    flouble valSubDiag=-1/hsquare;
//    flouble valMainDiag=4/hsquare;
//
//
//
//
//
//
//    int nm1=n-1;
//    int index;
//    while(iteration<MAXITERATIONS&&resi>tol) {
//        // consecutive blocks
//
//
//
//
//        if (!(iteration % step)) {
//            resi=0;
//            for(int i=0;i<n*n;i++) {
//                resi+=fabs(actualIteration[i]- lastIterSol[i]);
//            }
//            //   std::cout << iteration <<": "<< resi<< std::endl;
//        }
//
//
//        temp=lastIterSol;
//        lastIterSol=actualIteration;
//        actualIteration=temp;
//        iteration++;
//
//
//    }
//    std::cout << "Calculation finished after "<<iteration<<" Iterations.(%"<<step<<")"<<std::endl;
//    *numberOfIterations=iteration;
//

    return actualIteration;

}

__global__ void cuda_initSolutionVectors (flouble *actualIteration, flouble valBoundary) {
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



__global__ void cuda_jacoboIteration (flouble* actualIteration,flouble*  lastIterSol,int n,flouble valSubDiag,flouble  valMainDiag, flouble* f ) {
/*    int index;  //index=k*n+i;
    int tid=threadIdx.x;
    int bid=blockIdx.x;
    int bdim=blockDim.x;

    for(int k=1;k<nm1;k++) { // iterate through blocks

        for(int i=1;i<nm1;i++) {  // iterate in block
            actualIteration[index]=1/valMainDiag*(f[index]-valSubDiag*lastIterSol[index-n]-valSubDiag*lastIterSol[index-1]-valSubDiag*lastIterSol[index+1]-valSubDiag*lastIterSol[index+n]);
        }

    }
*/
}





void aufg13b() {

    int n=1024;
    int nn=n*n;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;
    flouble *cuda_fun;
    cudaMalloc(&cuda_fun,sizeof(flouble)*nn);

    flouble *result=new flouble[nn];

    int doneIterations=0;

    initMatrixRightHandSideCuda_CUDA<<<n,n>>>(h,cuda_fun);
    result=jacobiIter(n, cuda_fun, boundaryValue, &doneIterations,h);




   // result=jacobiIter(n, fun, boundaryValue, &doneIterations,h);

    //std::cout<<fun[1]<<std::endl;

    // displayMyMatrix(result,n,n);

    saveMyMatrix(result, n,n,h);






}





















// Utility functions

flouble* initMatrixKonstant(int m,int n, flouble value  ) {
    flouble*matrix=new flouble[n*m];
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            matrix[i*m+j]=value;
        }
    }
    return matrix;
}


void displayMyMatrix(flouble* matrix, int m,int n) {
    printf(" \n");
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            //printf("<%d %d %f>",i,j,matrix[i*m+j]);
            printf("%f ",matrix[i*m+j]);
        }
        printf(" \n");
    }
}

void saveMyMatrix(flouble* matrix, int m,int n, flouble h) {
    // h=1 for save indices
    std::ofstream myfile;
    myfile.open ("./../T13a.dat");
    flouble x;
    flouble y;
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            x=h*i;
            y=h*j;
            // printf("<%d %d %f>",x,y,matrix[i*m+j]);
            myfile<<x<<" "<<y<<" "<<matrix[i*m+j]<<"\n";
        }
        myfile<<std::endl;
        // printf(" \n");
    }
    myfile.close();
}
