# ParallelMatrixMultiplication

## main.cpp

*This file contains the code for sequential matrix multiplication and parallel-for matrix multiplication.
*Once this is run, it will run for all the configurations of n (from 200 to 2000 with a step of 200).
*For each n configuration it will run sufficient number of samples until a confidence level of 95% and an 
 accuracy of 5% is achieved for both sequential and parallel methods. 
*Minimum number of samples will be 10.

### Compilation Command: 
  g++ -fopenmp main.cpp -o main
### Run Command: 
  ./main

### Sample Output:
	Matrix size              = 200
	No. of Samples           = 21
	Average time sequential  = 0.0588636 s
	Average time parallel    = 0.0439755 s
	Speedup Achieved         = 1.33855

	Matrix size              = 400
	No. of Samples           = 16
	Average time sequential  = 0.453656 s
	Average time parallel    = 0.259677 s
	Speedup Achieved         = 1.747

	Matrix size              = 600
	No. of Samples           = 10
	Average time sequential  = 2.03639 s
	Average time parallel    = 1.12129 s
	Speedup Achieved         = 1.81611

	---------------------------------
	---------------------------------
	---------------------------------

 ## optimizedmain.cpp

*This file contains the code for optimized parallel matrix multiplication.
*Once this is run, it will run for all the configurations of n (from 200 to 2000 with a step of 200).
*For each n configuration it will run sufficient number of samples until a confidence level of 95% and an 
 accuracy of 5% is achieved. 
*Minimum number of samples will be 10.

### Compilation Command: 
  g++ -O3 -fopenmp -march=native -std=gnu++11 optimizedmain.cpp -o optimizedmain
### Run Command: 
  ./optimizedmain

### Sample Output:

	Matrix size              = 200
	No. of Samples           = 10
	Average time optimized   = 0.00781162 s

	Matrix size              = 400
	No. of Samples           = 10
	Average time optimized   = 0.039424 s

	Matrix size              = 600
	No. of Samples           = 10
	Average time optimized   = 0.126876 s

	Matrix size              = 800
	No. of Samples           = 10
	Average time optimized   = 0.234729 s

	Matrix size              = 1000
	No. of Samples           = 10
	Average time optimized   = 0.295726 s

	Matrix size              = 1200
	No. of Samples           = 10
	Average time optimized   = 0.448353 s

	Matrix size              = 1400
	No. of Samples           = 10
	Average time optimized   = 0.617094 s

	Matrix size              = 1600
	No. of Samples           = 10
	Average time optimized   = 1.1097 s

	Matrix size              = 1800
	No. of Samples           = 10
	Average time optimized   = 1.34761 s

	Matrix size              = 2000
	No. of Samples           = 10
	Average time optimized   = 1.69308 s
