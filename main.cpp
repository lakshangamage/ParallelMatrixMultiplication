#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 	
#include <omp.h>
#define CHUNKSIZE 100	
#define GAP 200	
using namespace std;

int nthreads, tid;

int N = 200;
int max_N;
double *matrixA;
double *matrixB;

double start_time;
double end_time;
double time_spent;

void printMatrix(double *matrix);
double* multiplyMatrices(double *matrix_1, double *matrix_2);
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2);

int main ()
{
	string user_input;
	cout << "Please enter N: ";
	getline(cin, user_input);
	stringstream(user_input) >> max_N;

	while(N <= max_N) {

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

		//cout << "\n==============Sequential===========\n"<< endl;
		start_time = omp_get_wtime();

		double *resultMatrix = multiplyMatrices(matrixA, matrixB);
		end_time = omp_get_wtime();

		time_spent = (end_time - start_time) * 1000.0;

		cout << "Matrix size = " << N << endl;
		cout << "Total time spent for sequential multiplication = " << time_spent <<" ms" << endl;
		delete []resultMatrix;

		//cout << "\n==============Parallel===========\n"<< endl;
		start_time = omp_get_wtime();

		double *parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB);
		//cout << "clock end" << endl;
		end_time = omp_get_wtime();
		time_spent = (end_time - start_time) * 1000.0;

		cout << "Total time spent for parallel multiplication = " << time_spent <<" ms"<< endl << endl;

		//printMatrix(parallelResultMatrix);

		delete []parallelResultMatrix;
		delete []matrixA; 
		delete []matrixB;
		N += GAP;
	} 
	 
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
	#pragma omp parallel shared(matrix_1,matrix_2,outputMatrix,chunk,size) private(i,j,k,total)
	{
		#pragma omp for schedule(dynamic,chunk)
		for (i = 0; i < size; i++)
		{
			//printf("%d\n",i);
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





