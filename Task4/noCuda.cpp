#include <iostream>
#include <math.h>
#include <fstream>
//#include <cuda.h>

//#define flouble float
#define flouble double
#define MAXITERATIONS 2000


// Utility
flouble* initMatrixKonstant(int m,int n, flouble value  ) ;
void displayMyMatrix(flouble* matrix, int m,int n);
void saveMyMatrix(flouble* matrix, int m,int n, flouble h, int numberTask);

#include "aufg13a.cpp"





int main() {
    std::cout << "Hello, World!" << std::endl;
    aufg13a();
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

void saveMyMatrix(flouble* matrix, int m,int n, flouble h, int numberTask) {
    // h=1 for save indices
    std::ofstream myfile;
    if (numberTask == 0)
        myfile.open("./results_a.dat");
    else if (numberTask == 1)
        myfile.open("./results_b.dat");
    else if (numberTask == 2)
        myfile.open("./results_c.dat");
    else if (numberTask == 3)
        myfile.open("./results_d.dat");
    else
        myfile.open("./results_temp.dat");

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













