#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 	
#include <omp.h>
#include <math.h> 
#define CHUNKSIZE 100	
#define MAX_SAMPLE_SIZE 50	
#define MIN_SAMPLE_SIZE 10	
#define GAP 200	
#define CONFIDENCE 1.96	
#define ACCURACY 0.05	
using namespace std;

int N = 200;
int max_N;
double *matrixA;
double *matrixB;

void printMatrix(double *matrix);
double* multiplyMatrices(double *matrix_1, double *matrix_2);
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2);
double calcStd(double* totArrray, double mean, int noOfSamples);
double calcSampleCount(double std, double mean);

int main ()
{
	string user_input;
	cout << "Please enter N: ";
	getline(cin, user_input);
	stringstream(user_input) >> max_N;


	while(N <= max_N) {
		int currentSample = 1;
		int samplesNeeded = MAX_SAMPLE_SIZE;
		double *serialTotArray = new double[MAX_SAMPLE_SIZE];
		double serial_tot = 0 ;
		double serial_mean = 0;
		double *parallelTotArray = new double[MAX_SAMPLE_SIZE];
		double parallel_tot = 0;
		double parallel_mean = 0;
		while( samplesNeeded <= MAX_SAMPLE_SIZE && samplesNeeded > currentSample){
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
			double serial_time = omp_get_wtime();

			double *resultMatrix = multiplyMatrices(matrixA, matrixB);

			double serial_time_spent = (omp_get_wtime() - serial_time);
			
			delete []resultMatrix;

			//cout << "\n==============Parallel===========\n"<< endl;
			double parallel_time = omp_get_wtime();

			double *parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB);
			//cout << "clock end" << endl;
			double parallel_time_spent = (omp_get_wtime() - parallel_time);

			//printMatrix(parallelResultMatrix);

			delete []parallelResultMatrix;
			delete []matrixA; 
			delete []matrixB;

			serial_tot += serial_time_spent;
			serialTotArray[currentSample-1] = serial_time_spent;
			
			
			parallel_tot += parallel_time_spent;
			parallelTotArray[currentSample-1] = parallel_time_spent;
			

			if (currentSample >= MIN_SAMPLE_SIZE)
			{
				serial_mean = serial_tot/currentSample;
				double serialStd = calcStd(serialTotArray, serial_mean, currentSample);
				int serialSampleCount = calcSampleCount(serialStd, serial_mean);

				parallel_mean = parallel_tot / currentSample;
				double parallelStd = calcStd(parallelTotArray, parallel_mean, currentSample);
				int parallelSampleCount = calcSampleCount(parallelStd, parallel_mean);

				samplesNeeded = serialSampleCount;
				if (serialSampleCount < parallelSampleCount)
				{
					samplesNeeded = parallelSampleCount;
				}
			}

			currentSample++;
		}

		cout << "Matrix size              = " << N << endl;
		cout << "No. of Samples           = " << currentSample - 1 << endl;
		cout << "Average time sequential  = " << serial_mean <<" s" << endl;		
		cout << "Average time parallel    = " << parallel_mean <<" s"<< endl ;
		cout << "Speedup Achieved         = " << serial_mean / parallel_mean << endl<< endl;

		delete []serialTotArray;
		delete []parallelTotArray;
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

double calcStd(double* totArrray, double mean, int noOfSamples){
	double total = 0;
	for (int i = 0; i < noOfSamples; i++)
	{
		total += pow((totArrray[i] - mean),2.0);
	}
	return sqrt(total/noOfSamples);
}

double calcSampleCount(double std, double mean){
	//return (int)round(pow(CONFIDENCE,2.0) * std * (1-std) / pow(ACCURACY,2.0)); 
	int samplecount = (int)round(pow((CONFIDENCE * std / ACCURACY)/mean,2.0));
	
	return samplecount; 
}

