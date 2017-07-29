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

#define MAX_N 2000 //Maximum number of n
#define CHUNKSIZE 100 //Chunk size of openmp
#define BLOCKSIZE 100 //Size of a block in "tiling" technique
#define MAX_SAMPLE_SIZE 500	//maximum number of samples
#define MIN_SAMPLE_SIZE 10 //minimum number of samples
#define GAP 200	//gap between each configuration of n
#define CONFIDENCE 1.96	// z-score relevant to confidence of 95%
#define ACCURACY 0.05 //accuracy(margin or error)	
using namespace std;

int N = 200;
double *matrixA;
double *matrixB;
double *matrixC;

void printMatrix(double *matrix); //function to print a matrix 
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2, double *matrix_3); //optimized function for multiplying 2 matrices parallely
double calcStd(double* totArrray, double mean, int noOfSamples); //function for calculating standard deviation of the sample
double calcSampleCount(double std, double mean);//function for calculating number of samples 
												//needed to achieve expected accuracy and confidence
void transpose(double *matrix); //function to get the transpose of a matrix

int main ()
{
	while(N <= MAX_N) { //run for each configuraion of n (200 to 2000 with a step of 200)
		int currentSample = 1;
		int samplesNeeded = MAX_SAMPLE_SIZE;		
		double *parallelTotArray = new double[MAX_SAMPLE_SIZE]; //keeps the total time of each sample
		double parallel_tot = 0; //current total of samples 
		double parallel_mean = 0; //current mean of samples
		while( currentSample <= MAX_SAMPLE_SIZE && samplesNeeded > currentSample){ //while maximum number of samples OR required number of samples not run		

			/*
				Initialize and allocate memory for matrices A, B and C(outputmatrix)
				16 byte aligned arrays are allocated since it is requiered for SIMD optimization.
			 */
			matrixA = (double *) aligned_alloc(16, N*N*sizeof(double));
			matrixB = (double *) aligned_alloc(16, N*N*sizeof(double));
			matrixC = (double *) aligned_alloc(16, N*N*sizeof(double));
		  
			srand (time(NULL)); //initialize random seed

			/*
				populate matrices with rendom doubles
			*/
			for(int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++)
				{
					matrixA[i*N+j] = rand();
					matrixB[i*N+j] = rand();  		
					matrixC[i*N+j] = 0.0;  		
				}  	
			}			
			
			double parallel_time = omp_get_wtime(); //record starting time of optimized parallel multiplication

			double *parallelResultMatrix = multiplyMatricesParallel(matrixA, matrixB, matrixC); //multiply matrices A and B parallelly
			
			double parallel_time_spent = (omp_get_wtime() - parallel_time); //calculate time spent for optimized parallel multiplication

			/*
				Deallocate the memory of created matrices
			*/
			free(matrixA);
			free(matrixB);
			free(parallelResultMatrix);

			/*
				add the result to total and store result in array
			*/			
			parallel_tot += parallel_time_spent;
			parallelTotArray[currentSample-1] = parallel_time_spent;

			/*
				number of samples needed for confidence 
				and accuracy is calculated only after 
				minimum number of samples exceeded.
				mean and standard deviations are calculated
				and the the number of needed samples is calculated.
			*/
			if (currentSample >= MIN_SAMPLE_SIZE)
			{				
				
				parallel_mean = parallel_tot / currentSample;
				double parallelStd = calcStd(parallelTotArray, parallel_mean, currentSample);
				int parallelSampleCount = calcSampleCount(parallelStd, parallel_mean);

				samplesNeeded = parallelSampleCount;				
			}

			currentSample++;
		}


		/*
			Print the calculated information.
		*/
		cout << "Matrix size              = " << N << endl;
		cout << "No. of Samples           = " << currentSample - 1 << endl;				
		cout << "Average time optimized   = " << parallel_mean <<" s"<< endl << endl;
		
		delete []parallelTotArray;
		N += GAP;
	} 
	 
	return 0;
}

/**
 * function to print a matrix 
 * 
 * @param double** matrix [matrix to be printed]
 */
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

/**
 * optimized function for multiplying 2 matrices parallely.
 * 
 * @param double** matrix1 [first matrix]
 * @param double** matrix2 [second matrix]
 * @param double** matrix3 [output matrix]
 * @return double** matrix3 [output matrix]
 */
double* multiplyMatricesParallel(double *matrix_1, double *matrix_2, double *matrix_3){
	int i,j,k,x,y, jmax,kmax = 0;
	int size = N;
	int blockSize = BLOCKSIZE;
	int blockCount = N/blockSize; //number of blocks in a row or a column
	__m128d c = _mm_setzero_pd(); //double vector
	
	transpose(matrix_2); //transpose the second matrix(matrix_2) for optimization

	/*
		Use OpenMp for parallelizing the multiplication.
		i,j,k,c,jmax,kmax variables are kept private for each threads 
		matrices are kept shared.
	 */
	#pragma omp parallel shared(matrix_1,matrix_2,matrix_3) private(j,i,c,k,jmax,kmax) //start of parallel region
	{		
		#pragma omp for schedule(dynamic,CHUNKSIZE) collapse(2) // OpenMp parallel-for upto 2 loops - definition
		
		/*
			Calculate the multipliction of matrices
			with blocking and SIMD(SSE3) for optimization
		 */
		for (x = 0; x < blockCount; x++) //for each block in a row
		{
			for (y = 0; y < blockCount; y++) // for each block in a column
			{				
				for (i = 0; i < size; i++) {
					jmax = ((x+1)*blockSize)>N?N:((x+1)*blockSize); //maximum value of j
			        for (j = x*blockSize; j < jmax; j++) {
			        	kmax = ((y+1)*blockSize)>N?N:((y+1)*blockSize); //maximum value of k
			            c = _mm_setzero_pd();
			            for (k = y*blockSize; k < kmax; k += 2) {      
			            	/*
			            		load values from matrices, multiply them and add them to c.
			            		These operations happen for 2 consecutive doubles parallely.
			            		Therefore the loop runs in step 2. 
			            	 */   	
			                c = _mm_add_pd(c, _mm_mul_pd(_mm_load_pd(&matrix_1[i*size+k]), _mm_load_pd(&matrix_2[j*size+k])));	                
			            }    
			            c = _mm_hadd_pd(c, c); 	//add the resulting 2 doubles(in vector c) to each other.Now both the values of c are same.		           
		             	c = _mm_add_sd(c, _mm_load_sd(&matrix_3[i*size+j]));  //load from current location of matrix element and add c to it.    
			            _mm_store_sd(&matrix_3[i*size+j], c); //store c in current element of matrix.
			        }
			    }
			}
		}		
	}   /* end of parallel region */ 
	return matrix_3;
} 

/**
 * function to get the transpose of a matrix
 * 
 * @param double* matrix [matrix to be transposed]
 */
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

/**
 * function for calculating standard deviation of the sample
 * 
 * @param double* totArrray [Array containing results of each sample]
 * @param double mean [Mean of the sample]
 * @param int noOfSamples [Number of samples upto now]
 * @return double [Standard deviation of the sample]
 */
double calcStd(double* totArrray, double mean, int noOfSamples){
	double total = 0;
	for (int i = 0; i < noOfSamples; i++)
	{
		total += pow((totArrray[i] - mean),2.0);
	}
	return sqrt(total/noOfSamples);
}

/**
 * function for calculating number of samples 
 * needed to achieve expected accuracy and confidence
 * 
 * @param double std [starndard deviation of the sample]
 * @param double mean [mean of the sample]
 * @return double [number of samples needed to achieve expected confidence and accuracy]
 */
double calcSampleCount(double std, double mean){
	int samplecount = (int)round(pow((CONFIDENCE * std )/ (ACCURACY*mean), 2.0));
	return samplecount; 
}

