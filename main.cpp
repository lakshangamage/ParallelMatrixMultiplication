#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 	
#include <omp.h>
#define CHUNKSIZE 100	
using namespace std;

int nthreads, tid;

int N;
double *matrixA;
double *matrixB;

void printMatrix(double *matrix);
double* multiplyMatrices(double *matrix_1, double *matrix_2);
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2);

int main ()
{
	string user_input;
	cout << "Please enter N: ";
	getline(cin, user_input);
	stringstream(user_input) >> N;

	//int array_size = N;
	matrixA = new double[N*N];
	matrixB  = new double[N*N];

  /* initialize random seed: */
	srand (time(NULL));

	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			matrixA[i*N+j] = rand();
			matrixB[i*N+j] = rand();  		
		}  	
	}
	// printMatrix(matrixA);
	// printMatrix(matrixB);

	cout << "==============Sequential===========\n"<< endl;
	cout << "timing start..."<< endl;
	clock_t start_time = clock();

	double *resultMatrix = multiplyMatrices(matrixA, matrixB);

	float time_spent = (double)(clock() - start_time) * 1000.0 /  CLOCKS_PER_SEC;
	cout << "timing end..."<< endl;
	//printMatrix(resultMatrix);
	delete []resultMatrix;
	cout << "Total time spent for multiplication = " << time_spent << endl;

	cout << "\n==============Parallel===========\n"<< endl;
	cout << "timing start..."<< endl;
	start_time = clock();

	double *parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB);

	time_spent = (double)(clock() - start_time) * 1000.0 /  CLOCKS_PER_SEC;
	cout << "timing end..."<< endl;
	//printMatrix(resultMatrix);
	delete []parallelResultMatrix;
	cout << "Total time spent for multiplication = " << time_spent << endl;
	delete []matrixA; 
	delete []matrixB; 
	 
	return 0;
}


void printMatrix(double *matrix) {
	for(int i = 0; i < N; i++) {
	  	for (int j = 0; j < N; j++)
	  	{
	  		cout << matrix[i*N+j] << "\t";
	  	}  	
	  	cout << "\n";
  	}	
  	cout << "\n";
}

double* multiplyMatrices(double *matrix_1, double *matrix_2){
	double *outputMatrix = new double[N*N];	
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
				{
					double total = 0;
					for (int k = 0; k < N; k++)
					{
						total += matrix_1[i*N+k] * matrix_2[k*N+j];
					}
					outputMatrix[i*N+j] = total;
				}		
	}
	return outputMatrix;
} 

double* multiplyMatricesParallel(double *matrix_1, double *matrix_2){
	double *outputMatrix = new double[N*N];	
	int chunk = CHUNKSIZE;
	int i,j,k = 0;
	int size = N;
	double total = 0;
	#pragma omp parallel shared(matrix_1,matrix_2,outputMatrix,chunk) private(i,j,k,total,size)
	{
		#pragma omp for schedule(dynamic,chunk) nowait
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				total = 0;
				for (k = 0; k < size; k++)
				{
					total += matrix_1[i*size+k] * matrix_2[k*size+j];
				}
				outputMatrix[i*size+j] = total;
			}		
		}

	}   /* end of parallel region */ 
	return outputMatrix;
} 





