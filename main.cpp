#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 	
#include <omp.h>
#include <math.h> 
#define MAX_N 2000 //Maximum number of n
#define CHUNKSIZE 100	//Chunk size of openmp
#define MAX_SAMPLE_SIZE 500	//maximum number of samples
#define MIN_SAMPLE_SIZE 10	//minimum number of samples
#define GAP 200		//gap between each configuration of n
#define CONFIDENCE 1.96	// z-score relevant to confidence of 95%
#define ACCURACY 0.05	//accuracy(margin or error)
using namespace std; 

int N = 200;
double **matrixA; 
double **matrixB;

void printMatrix(double **matrix); //function to print a matrix 
double** multiplyMatrices(double **matrix_1, double **matrix_2); //function for multiplying 2 matrices sequentially
double** multiplyMatricesParallel(double **matrix_1, double **matrix_2); //function for multiplying 2 matrices parallely.
double calcStd(double* totArrray, double mean, int noOfSamples); //function for calculating standard deviation of the sample
double calcSampleCount(double std, double mean); //function for calculating number of samples 
												 //needed to achieve expected accuracy and confidence

int main ()
{
	while(N <= MAX_N) { //run for each configuraion of n (200 to 2000 with a step of 200)
		int currentSample = 1;
		int samplesNeeded = MAX_SAMPLE_SIZE;
		double *serialTotArray = new double[MAX_SAMPLE_SIZE]; //keeps the total time of each sample -sequential
		double serial_tot = 0 ; //current total of samples - sequential 
		double serial_mean = 0; //current mean of samples - sequential 
		double *parallelTotArray = new double[MAX_SAMPLE_SIZE]; //keeps the total time of each sample -parallel
		double parallel_tot = 0; //current total of samples - parallel
		double parallel_mean = 0; //current mean of samples - parallel 
		while( currentSample <= MAX_SAMPLE_SIZE && samplesNeeded > currentSample){
			matrixA = new double*[N];
			matrixB = new double*[N];

			for (int i = 0; i < N; i++)
			{
				matrixA[i] = new double[N];
				matrixB[i] = new double[N];
			}
		  
			srand (time(NULL)); //initialize random seed

			/*
				populate matrices with rendom doubles
			*/
			for(int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++)
				{
					matrixA[i][j] = rand();
					matrixB[i][j] = rand();  		
				}  	
			}
			double serial_time = omp_get_wtime(); //record starting time of sequential multiplication

			double **resultMatrix = multiplyMatrices(matrixA, matrixB); //multiply matrices A and B sequenially

			double serial_time_spent = (omp_get_wtime() - serial_time); //calculate time spent for sequential multiplication.
			
			double parallel_time = omp_get_wtime(); //record starting time of parallel multiplication

			double **parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB); //multiply matrices A and B parallelly

			double parallel_time_spent = (omp_get_wtime() - parallel_time); //calculate time spent for parallel multiplication

			/*
				Deallocate the memory of created matrices
			*/
			for(int i = 0; i < N; i++) {
			    delete [] resultMatrix[i];
			    delete [] parallelResultMatrix[i];
			    delete [] matrixA[i];
			    delete [] matrixB[i];
			}
			delete []resultMatrix;
			delete []parallelResultMatrix;
			delete []matrixA; 
			delete []matrixB;

			/*
				add up the results to relevant totals and store results in relevant arrays
			*/
			serial_tot += serial_time_spent;
			serialTotArray[currentSample-1] = serial_time_spent;
			
			
			parallel_tot += parallel_time_spent;
			parallelTotArray[currentSample-1] = parallel_time_spent;
			
			/*
				number of samples needed for confidence 
				and accuracy is calculated only after 
				minimum number of samples exceeded.
				mean and standard deviations are calculated
				and number of needed samples is calculated 
				for each of the methods(serial/parallel).
			*/

			if (currentSample >= MIN_SAMPLE_SIZE) 
			{
				serial_mean = serial_tot/currentSample;
				double serialStd = calcStd(serialTotArray, serial_mean, currentSample);
				int serialSampleCount = calcSampleCount(serialStd, serial_mean);

				parallel_mean = parallel_tot / currentSample;
				double parallelStd = calcStd(parallelTotArray, parallel_mean, currentSample);
				int parallelSampleCount = calcSampleCount(parallelStd, parallel_mean);

				/*
					Maximum of required number of samples 
					in bot cases is taken as the number of 
					samples needed for the whole iteration
				*/
				samplesNeeded = serialSampleCount;
				if (serialSampleCount < parallelSampleCount)
				{
					samplesNeeded = parallelSampleCount;
				}
			}

			currentSample++;
		}

		/*
			Print the calculated information.
		*/

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


void printMatrix(double **matrix) {
	for(int i = 0; i < N; i++) {
	  	for (int j = 0; j < N; j++)
	  	{
	  		cout << matrix[i][j] << "\t";
	  	}  	
	  	cout << "\n";
  	}	
  	cout << "\n";
}

double** multiplyMatrices(double **matrix_1, double **matrix_2){
	double **outputMatrix = new double*[N];	
	for (int x = 0; x < N; x++)
	{
		outputMatrix[x] = new double[N];
	}
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
				{
					outputMatrix[i][j] = 0;

					for (int k = 0; k < N; k++)
					{
						outputMatrix[i][j] += matrix_1[i][k] * matrix_2[k][j];
					}
				}		
	}
	return outputMatrix;
} 

double** multiplyMatricesParallel(double **matrix_1, double **matrix_2){
	double **outputMatrix = new double*[N];	
	for (int x = 0; x < N; x++)
	{
		outputMatrix[x] = new double[N];
	}
	int chunk = CHUNKSIZE;
	int i,j,k = 0;
	int size = N;

	#pragma omp parallel shared(matrix_1,matrix_2,outputMatrix,chunk,size) private(i,j,k)
	{
		#pragma omp for schedule(dynamic,chunk)
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{		
				outputMatrix[i][j] = 0;		
				for (k = 0; k < size; k++)
				{
					outputMatrix[i][j] += matrix_1[i][k] * matrix_2[k][j];
				}				
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
	int samplecount = (int)round(pow((CONFIDENCE * std )/ (ACCURACY*mean), 2.0));
	return samplecount; 
}

