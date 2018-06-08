#include <iostream>
#include <math.h>
#include <fstream>
#include <cuda.h>

//#define flouble float
#define flouble double
#define MAXITERATIONS 20000

// Utility
flouble* initMatrixKonstant(int m,int n, flouble value  ) ;
void displayMyMatrix(flouble* matrix, int m,int n);
void saveMyMatrix(flouble* matrix, int m,int n, flouble h);

#include "aufg13a.cpp"
#include "aufg13b.cpp"







int main() {
    aufg13a();
    std::cout << "Hello, World!" << std::endl;
    aufg13b();
    return 0;
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
    myfile.open ("./results.dat");
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
























