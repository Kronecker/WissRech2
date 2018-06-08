//
// Created by grabiger on 08.06.2018.
//

using namespace std;

void aufg13a();
flouble* initMatrixRightHandSide(int n, flouble h  );
flouble* jacobiIter(int n, flouble *f, flouble valBoundary, int* numberOfIterations, flouble h);


void aufg13a() {

    int n=1024;
    flouble h = 1./(n-1);

    flouble boundaryValue=0;
    flouble *fun;
    flouble *result;

    int doneIterations=0;


    fun=initMatrixRightHandSide(n,h);
    result=jacobiIter(n, fun, boundaryValue, &doneIterations,h);

    saveMyMatrix(result, n,n,h);

    delete(fun);
    delete(result);
}

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
            //std::cout << iteration <<": "<< resi<< std::endl;
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



