#include <iostream>
#include <math.h>
#include <fstream>
#include <cuda.h>

//#define flouble float
#define flouble double
#define MAXITERATIONS 2000


// Utility
flouble* initMatrixKonstant(int m,int n, flouble value  ) ;
void displayMyMatrix(flouble* matrix, int m,int n);
void saveMyMatrix(flouble* matrix, int m,int n, flouble h, int numberTask);
void saveMyMatrixAppend(flouble* matrix, int m,int n, flouble h, int numberTask, int offset);

//#include "aufg13a.cpp"
//#include "aufg13b.cpp"  // An include-disaster waiting to happen ....
//#include "aufg13c.cpp"
#include "aufg13d.cpp"






int main() {

    std::cout << "Hello, World!" << std::endl;
   // aufg13a();

    std::cout << "Hello, World!" << std::endl;
  //  aufg13b();

    std::cout << "Hello, World!" << std::endl;
//    aufg13c();

    std::cout << "Hello, World!" << std::endl;
    aufg13d();


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
            myfile<<x<<" "<<y<<" "<<matrix[i*n+j]<<"\n";
        }
        myfile<<std::endl;
        // printf(" \n");
    }
    myfile.close();
}




void saveMyMatrixAppend(flouble* matrix, int m,int n, flouble h, int numberTask, int offset) {
    // h=1 for save indices
    std::ofstream myfile;
    if (numberTask == 0)
        myfile.open("./results_a.dat",std::ofstream::out | std::ofstream::app);
    else if (numberTask == 1)
        myfile.open("./results_b.dat",std::ofstream::out | std::ofstream::app);
    else if (numberTask == 2)
        myfile.open("./results_c.dat",std::ofstream::out | std::ofstream::app);
    else if (numberTask == 3)
        myfile.open("./results_d.dat",std::ofstream::out | std::ofstream::app);
    else
        myfile.open("./results_temp.dat",std::ofstream::out | std::ofstream::app);

    flouble x;
    flouble y;
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            x=h*(i+offset);
            y=h*j;
            // printf("<%d %d %f>",x,y,matrix[i*m+j]);
            myfile<<x<<" "<<y<<" "<<matrix[i*n+j]<<"\n";
        }
        myfile<<std::endl;
        // printf(" \n");
    }
    myfile.close();
}


















