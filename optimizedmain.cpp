#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 	
#include <omp.h>
#include <math.h> 
#include <pmmintrin.h>
#include <immintrin.h> 
#include <assert.h> 

#define MAX_N 2000
#define CHUNKSIZE 100	
#define BLOCKSIZE 100	
#define MAX_SAMPLE_SIZE 500	
#define MIN_SAMPLE_SIZE 10
#define GAP 200	
#define CONFIDENCE 1.96	
#define ACCURACY 0.05	
using namespace std;

int N = 200;
double *matrixA;
double *matrixB;
double *matrixC;

void printMatrix(double *matrix);
double* multiplyMatrices(double *matrix_1, double *matrix_2);
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2, double *matrix_3);
double calcStd(double* totArrray, double mean, int noOfSamples);
double calcSampleCount(double std, double mean);
void transpose(double *matrix);

int main ()
{
	while(N <= MAX_N) {
		int currentSample = 1;
		int samplesNeeded = MAX_SAMPLE_SIZE;		
		double *parallelTotArray = new double[MAX_SAMPLE_SIZE];
		double parallel_tot = 0;
		double parallel_mean = 0;
		while( currentSample <= MAX_SAMPLE_SIZE && samplesNeeded > currentSample){		

			matrixA = (double *) aligned_alloc(16, N*N*sizeof(double));
			matrixB = (double *) aligned_alloc(16, N*N*sizeof(double));
			matrixC = (double *) aligned_alloc(16, N*N*sizeof(double));
		  
			srand (time(NULL));

			for(int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++)
				{
					matrixA[i*N+j] = rand();
					matrixB[i*N+j] = rand();  		
					matrixC[i*N+j] = 0.0;  		
				}  	
			}			
			
			double parallel_time = omp_get_wtime();

			double *parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB, matrixC);
			
			double parallel_time_spent = (omp_get_wtime() - parallel_time);

			free(matrixA);
			free(matrixB);
			free(parallelResultMatrix);
						
			parallel_tot += parallel_time_spent;
			parallelTotArray[currentSample-1] = parallel_time_spent;
			parallel_mean = parallel_tot / currentSample;

			if (currentSample >= MIN_SAMPLE_SIZE)
			{				
				
				double parallelStd = calcStd(parallelTotArray, parallel_mean, currentSample);
				int parallelSampleCount = calcSampleCount(parallelStd, parallel_mean);

				samplesNeeded = parallelSampleCount;				
			}

			currentSample++;
		}

		cout << "Matrix size              = " << N << endl;
		cout << "No. of Samples           = " << currentSample - 1 << endl;				
		cout << "Average time optimized   = " << parallel_mean <<" s"<< endl << endl;
		
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

double* multiplyMatricesParallel(double *matrix_1, double *matrix_2, double *matrix_3){
	int i,j,k,x,y, jmax,kmax = 0;
	int size = N;
	int blockSize = BLOCKSIZE;
	int blockCount = N/blockSize;
	__m128d c = _mm_setzero_pd();	
	
	transpose(matrix_2);

	#pragma omp parallel shared(matrix_1,matrix_2,matrix_3) private(j,i,c,k,jmax,kmax)	
	{		
		#pragma omp for schedule(dynamic,CHUNKSIZE) collapse(2)
		
		for (x = 0; x < blockCount; x++)
		{
			for (y = 0; y < blockCount; y++)
			{				
				for (i = 0; i < size; i++) {
					jmax = ((x+1)*blockSize)>N?N:((x+1)*blockSize);
			        for (j = x*blockSize; j < jmax; j++) {
			        	kmax = ((y+1)*blockSize)>N?N:((y+1)*blockSize);
			            c = _mm_setzero_pd();
			            for (k = y*blockSize; k < kmax; k += 2) {         	
			                c = _mm_add_pd(c, _mm_mul_pd(_mm_load_pd(&matrix_1[i*size+k]), _mm_load_pd(&matrix_2[j*size+k])));	                
			            }    
			            c = _mm_hadd_pd(c, c); 			           
		             	c = _mm_add_sd(c, _mm_load_sd(&matrix_3[i*size+j]));     
			            _mm_store_sd(&matrix_3[i*size+j], c);
			        }
			    }
			}
		}		
	}   /* end of parallel region */ 
	return matrix_3;
} 

void transpose(double *matrix) {
	
	int i, j = 0;
	double temp = 0;
	#pragma omp parallel shared(matrix) private(j,i, temp)	
	{		
		#pragma omp for schedule(dynamic,CHUNKSIZE) collapse(2)	
	    for (i = 0; i < N; i++) {
	        for (j = i + 1; j < N; j++) {
	        	temp = matrix[i*N+j];
	        	matrix[i*N+j] = matrix[j*N+i];
	        	matrix[j*N+i] = temp;
	        }
	    }
	}
	
	
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

