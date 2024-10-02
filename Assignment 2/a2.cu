#include <iostream>
#include <string>
#include <cassert>
#include <ctime>
#include <vector>

using namespace std;

/* ******************************
*  Exception handler functions
*
* Reference: sample_example
****************************** */
struct cuda_exception
{
	explicit cuda_exception(const char* err) : error_info(err) {}
	explicit cuda_exception(const string& err) : error_info(err) {}
	string what() const throw() { return error_info; }

private:
	string error_info;
};

void checkCudaError(const char* msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		string error_info(msg);
		error_info += " : ";
		error_info += cudaGetErrorString(err);
		throw cuda_exception(error_info);
	}
}

/**
* CUDA kernel for univariate polynomial multiplication
*
* @a, the first input coefficients of polynomial
* @b, the second input coefficients of polynomial
* @c, the output coefficients of polynomial
* @n, degree of polynomial
*
* reference: Dependence_Analysis_and_Parallelization.pdf (slide 41)
*/
__global__ void polynomials_mul(int* a, int* b, int* c, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < 2 * n + 1)
	{
		c[idx] = 0;
		for (int t = max(0, idx - n); t <= min(idx, n); t++)
		{
			c[idx] = c[idx] + a[t] * b[idx - t];
		}
	}
}

/*
* Check result on the CPU (C function)
*/
void verify_result(int* a, int* b, int* c, int n)
{
	clock_t t1 = clock();
	for (int i = 0; i < 2 * n + 1; i++)
	{
		int k = i;
		int result = 0;
		for (int j = 0; j <= k; j++)
		{
			if (j <= n && k - j <= n)
			{
				result = result + a[j] * b[k - j];
			}
		}
		assert(result == c[i]);
	}
	clock_t t2 = clock();
	cout << "   C Function Verification Passed! - takes " << (t2 - t1) / double(CLOCKS_PER_SEC) * 1000 << " ms" << endl;
}

/*
* Initialize c with random coefficients from {-1, 0, 1}
*
* @c, :an array of integers representing the coefficients of a polynomial
*  that is randomly generated with coefficients chosen from the set {-1, 0, 1}
*
*   n: the size of polynomial terms
*/
void random_polynomial(int* c, int n)
{
	srand(time(NULL));
	for (int i = 0; i < n; i++) {
		c[i] = rand() % 3 - 1;  // Generate random coefficient from {-1, 0, 1}
	}
}

/*
*  print out passed polynomial of size n
*
* c: coefficients of polynomial
* n: size of polynomial terms

*/
void print_polynomial(int* c, int n)
{
	for (int i = 0; i < n; i++)
	{
		cout << "   " << c[i] << " ";
	}
	cout << endl;
}
/*
*  Testing functon
*	Retun the testing running time for tracking the performance
*
*	@e, the exponent
*	@B, the threads per block
*/
int run_test(int e, int B) {
	double time;
	cout << "   --------------------------------------------------" << endl;
	cout << "   Case: B=" << B << " and n=" << "2^" << e;

	clock_t t1 = clock();
	size_t n = 1ULL << e; // calculate n based on the exponent
	size_t bytes = (n + 1) * sizeof(int); // size of a and b
	size_t bytes_c = (2 * n + 1) * sizeof(int); //  size of c

	int* a_h, * b_h, * c_h; // host arrays
	int* a_d, * b_d, * c_d; // device arrays
	try
	{
		// allocate memory
		a_h = (int*)malloc(bytes);
		b_h = (int*)malloc(bytes);
		c_h = (int*)malloc(bytes_c);

		// Initialize host random coefficients from {-1, 0, 1}
		random_polynomial(a_h, n + 1);
		random_polynomial(b_h, n + 1);

		// initialize polynomial c_h with size 2n+1
		int cSize = 2 * n + 1;
		for (int i = 0; i < cSize; i++)
		{
			c_h[i] = 0;
		}

		// Allocate device memory
		cudaMalloc((void**)&a_d, bytes);
		cudaMalloc((void**)&b_d, bytes);
		cudaMalloc((void**)&c_d, bytes_c);

		// Copy data to the device
		cudaMemcpy(a_d, a_h, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b_h, bytes, cudaMemcpyHostToDevice);

		// compute the execution configure
		// number of blocks (round up to the nearest whole number of blocks)
		int nBlocks = (2 * n + 1) / B + ((2 * n + 1) % B == 0 ? 0 : 1);

		// block size
		int bSize = B;

		polynomials_mul <<<nBlocks, bSize >>> (a_d, b_d, c_d, n);
		cudaDeviceSynchronize();

		// read c from the device 
		cudaMemcpy(c_h, c_d, bytes_c, cudaMemcpyDeviceToHost);

		clock_t t2 = clock();
		time = (t2 - t1) / double(CLOCKS_PER_SEC) * 1000;
		cout << " takes " << time << " ms\n" << endl;

		// print polynomial
		if (n < 10)
		{
			cout << "   Input array a:" << endl;
			print_polynomial(a_h, n + 1);
			cout << "   Input array b:" << endl;
			print_polynomial(b_h, n + 1);
			cout << endl;
			cout << "   Output array c:" << endl;
			print_polynomial(c_h, 2 * n + 1);
		}

		//Check result with C function
		verify_result(a_h, b_h, c_h, n);
	}
	catch (cuda_exception& err)
	{
		cout << err.what() << endl;
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);

		free(a_h);
		free(b_h);
		free(c_h);
		return EXIT_FAILURE;
	}
	catch (...)
	{
		cout << "unknown exeception" << endl;
		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);

		free(a_h);
		free(b_h);
		free(c_h);
		return EXIT_FAILURE;
	}


	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

	free(a_h);
	free(b_h);
	free(c_h);

	cout << "   --------------------------------------------------" << endl;
	return time;
}


/**
*	Implement a CUDA program using ⌈(2n + 1)/B⌉
*	 thread-blocks with B = 32 threads.
*/
void run_question_1() {
	int exponent;
	cout << "\n   ***************   Question 1   ***************" << endl;
		cout << "   The polynomial degree n = 2^exponent\n" << endl;
		cout << "   The input/output coefficients will be \n   printed out only if n < 10 (exponent < 4)\n" << endl;
		cout << "   Thread per block B = 32" << endl;
		cout << "   Defaut exponent = 16 if input value < 0" << endl;
		cout << "   ********************************************* " << endl;

	while (true) {
		cout << "   Please input an exponent or type a non-number to exit: ";
		if (!(cin >> exponent)) {
			// Input is not a number, clear input buffer and exit loop
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
			break;
		}

		cout << endl;

		// Default exponent is 16 if input is less than or equal to 1
		int e = (exponent < 0) ? 16 : exponent;
		int B = 32; // thread-blocks

		run_test(e, B);
	}
}

void run_question_2()
{
	cout << "\n   ***************   Question 2   ***************" << endl;
	int exponents[] = { 14, 16 };
	int Bs[] = { 32, 64, 128, 256, 512 };
	int e_len = sizeof(exponents) / sizeof(exponents[0]);
	int b_len = sizeof(Bs) / sizeof(Bs[0]);

	vector<pair<int, int>> bestPerformances(e_len, { numeric_limits<int>::max(), 0 });

	for (int i = 0; i < e_len; i++)
	{
		int e = exponents[i];

		for (int j = 0; j < b_len; j++)
		{
			int B = Bs[j];

			int time = run_test(e, B);

			// Update best performance B for this exponent if needed
			if (time < bestPerformances[i].first)
			{
				bestPerformances[i].first = time;
				bestPerformances[i].second = B;
			}
		}
	}
	cout << "\n   Best performance B for n = 2^" << exponents[0] << ": " << bestPerformances[0].second << endl;
	cout << "   Best performance B for n = 2^" << exponents[1] << ": " << bestPerformances[1].second << endl;
}

int main(int argc, char** argv)
{
	int input;
	if (argc >= 2)
	{
		input = atoi(argv[1]);
	}
	else
	{
		cout << "\n  Enter 1 for Question 1 or 2 for Question 2: ";
		scanf("%d", &input);
		cout << endl;

		// Consume the newline character from the input buffer
		while (getchar() != '\n');

		while (input != 1 && input != 2)
		{
			cout << "\n  Invalid input. Enter 1 for Question 1 or 2 for Question 2: ";
			scanf("%d", &input);
			cout << endl;

			// Consume the newline character from the input buffer
			while (getchar() != '\n');
		}
	}

	switch (input)
	{
	case 1:
		run_question_1();
		break;
	case 2:
		run_question_2();
		break;
	default:
		printf("Invalid number. Running Question 1 by default.\n");
		run_question_1();
		break;
	}

	return 0;
}
