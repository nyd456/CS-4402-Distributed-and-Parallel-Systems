#include <iostream>
#include <assert.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <time.h>
#include<vector>
using namespace std;

/* reference: class example: DNC_MM.cpp */
long print_time(std::ostream &out, struct timespec start,
                struct timespec end)
{
    time_t s = end.tv_sec - start.tv_sec;
    long ns = end.tv_nsec - start.tv_nsec;
    if (ns < 0)
    {
        ns += 1000000000;
        s -= 1;
    }
    return (s * 1000 + ((ns + 500000) / 1000000));
}

/*
    random_matrix creates a random n-by-n matrix
    reference: class example: DNC_MM.cpp
*/
void random_matrix(double *A, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        // A[i] = rand()%n;
        A[i] = static_cast<double>(std::rand() % n) + 0.1L;
    }
}

/*
    Print an n-by-n matrix
    reference: class example: DNC_MM.cpp
*/
void print_matrix(double *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", a[n * i + j]);
            if (j == n - 1)
                printf("\n");
        }
    }

    printf("\n");
}

/*
    return true if passed matrix A is an identity matrix,
    false otherwise
*/
bool is_identity_matrix(double *A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (i == j && A[i * n + j] != 1.0)
            {
                return false; // Diagonal elements should be 1
            }
            else if (i != j && A[i * n + j] != 0.0)
            {
                return false; // Non-diagonal elements should be 0
            }
        }
    }
    return true; // Matrix passed all checks
}
/*   mA = -A */
void minus_matrix(double *A, double *mA, int n)
{

    for (int i = 0; i < n; i++)
    {
        mA[i] = A[i] == 0 ? 0 : -A[i];
    }
}
/* copy value of a into A */
void copy_matrix(double *A, double *a, int i, int j, int k, int l, int N)
{
    int n = j - i + 1;
    for (int p = 0; p < n; p++)
    {
        for (int q = 0; q < n; q++)
        {
            A[((p + i) * N) + q + k] = a[p * n + q];
        }
    }
}
/* populate sub matrix value*/
void get_sub_matrix(double *A, double *a, int i, int j, int k, int l, int N)
{
    int n = j - i + 1;
    for (int p = 0; p < n; p++)
    {
        for (int q = 0; q < n; q++)
        {
            a[p * n + q] = A[((p + i) * N) + q + k];
        }
    }
}

/*
   lower triangular matrix inversion by forward substitution
   A - input matrix
   IA - inverse of A
*/
void loop_serial_inverse(double *A, double *IA, int n)
{
    cilk_for(int i = 0; i < n; i++)
    {
        IA[i * n + i] = 1.0 / A[i * n + i];
        for (int j = i + 1; j < n; j++)
        {
            double sum = 0.0;
            for (int k = i; k < j; k++)
            {
                sum += A[j * n + k] * IA[k * n + i];
            }
            IA[j * n + i] = sum == 0 ? 0 : -sum / A[j * n + j];
        }
    }
}

/*
    matrix multiplication loop serial
    C = A * B
    reference: class example: DNC_MM.cpp
*/
void mm_loop_serial(double *C, int k0, int k1, double *A, int i0, int i1, double *B, int j0, int j1, int n)
{
    /* MODIFY THIS CODE TO MAKE IT PARALLEL */
    for (int i = i0; i < i1; i++)
    {
        for (int j = j0; j < j1; j++)
            for (int k = k0; k < k1; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
    }
}

/*
    matrix multiplication (serial) - divide-and-conquer approch
    C = A * B
    reference: class example: DNC_MM.cpp
*/
void serial_dandc(int i0, int i1, int j0, int j1, int k0, int k1, double *A, int lda, double *B, int ldb, double *C, int ldc, int X)
{
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;
    if (di >= dj && di >= dk && di >= X)
    {
        int mi = i0 + di / 2;
        serial_dandc(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
        serial_dandc(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
    }
    else if (dj >= dk && dj >= X)
    {

        int mj = j0 + dj / 2;
        serial_dandc(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc, X);
        serial_dandc(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
    }
    else if (dk >= X)
    {

        int mk = k0 + dk / 2;
        serial_dandc(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc, X);
        serial_dandc(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc, X);
    }
    else
    {
        mm_loop_serial(C, k0, k1, A, i0, i1, B, j0, j1, lda);
    }
}

/*
    matrix multiplication (parallel) - divide-and-conquer approch
    C = A * B
    reference: class example: DNC_MM.cpp
*/
void parallel_dandc(int i0, int i1, int j0, int j1, int k0, int k1, double *A, int lda, double *B, int ldb, double *C, int ldc, int X)
{
    int di = i1 - i0;
    int dj = j1 - j0;
    int dk = k1 - k0;
    if (di >= dj && di >= dk && di >= X)
    {
        int mi = i0 + di / 2;
        cilk_spawn parallel_dandc(i0, mi, j0, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
        parallel_dandc(mi, i1, j0, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
        cilk_sync;
    }
    else if (dj >= dk && dj >= X)
    {

        int mj = j0 + dj / 2;
        cilk_spawn parallel_dandc(i0, i1, j0, mj, k0, k1, A, lda, B, ldb, C, ldc, X);
        parallel_dandc(i0, i1, mj, j1, k0, k1, A, lda, B, ldb, C, ldc, X);
        cilk_sync;
    }
    else if (dk >= X)
    {

        int mk = k0 + dk / 2;
        cilk_spawn parallel_dandc(i0, i1, j0, j1, k0, mk, A, lda, B, ldb, C, ldc, X);
        parallel_dandc(i0, i1, j0, j1, mk, k1, A, lda, B, ldb, C, ldc, X);
        cilk_sync;
    }
    else
    {
        mm_loop_serial(C, k0, k1, A, i0, i1, B, j0, j1, lda);
    }
}

/*
   the serial elision of inverse function when cilk spawn and cilk sync are erased
   A - input matrix
   IA - inverse of A
   B - threshold
*/
void serial_inverse(double *A, double *IA, int i, int j, int k, int l, int N, int B)
{
    if (N <= B)
    {
        loop_serial_inverse(A, IA, N);
    }
    else
    { // n > B

        int mid = (j - i) / 2;
        int n = mid + 1;
        int size = n * n;
        // sub matrix
        double *a1 = (double *)malloc(size * sizeof(double));   //  A1
        double *a2 = (double *)malloc(size * sizeof(double));   //  A2
        double *a3 = (double *)malloc(size * sizeof(double));   //  A3
        double *IA1 = (double *)malloc(size * sizeof(double));  // inverse of A1
        double *IA2 = (double *)malloc(size * sizeof(double));  // inverse of A2
        double *IA3 = (double *)malloc(size * sizeof(double));  // inverse of A3
        double *mIA3 = (double *)malloc(size * sizeof(double)); // minus(IA3) i.e. -IA3
        double *W = (double *)malloc(size * sizeof(double));    // W = -IA3 * IA2

        // populate sub-matrix values
        get_sub_matrix(A, a1, i, i + mid, k, k + mid, N);
        get_sub_matrix(A, a2, i + n, j, k, k + mid, N);
        get_sub_matrix(A, a3, i + n, j, k + n, l, N);

        // (1) IA1 - inverse of A1
        serial_inverse(a1, IA1, i, i + mid, k, k + mid, n, B);

        // (2) IA3 - inverse of A3
        serial_inverse(a3, IA3, i + n, j, k + n, l, n, B);

        // (3) IA2 - inverse of A2 = -IA3 * A2 * IA1
        minus_matrix(IA3, mIA3, size); // -IA3

        mm_loop_serial(W, 0, n, mIA3, 0, n, a2, 0, n, n); // W = -IA3 * A2
        mm_loop_serial(IA2, 0, n, W, 0, n, IA1, 0, n, n); // IA2 = W * IA1

        // cout << "\n***********IA1**********" << endl;
        // print_matrix(IA1, n);

        // cout << "\n***********IA2**********" << endl;
        // print_matrix(IA2, n);

        // cout << "\n***********IA3**********" << endl;
        // print_matrix(IA3, n);

        // (4) Copy value into IA
        copy_matrix(IA, IA1, i, i + mid, k, k + mid, N); // copy IA1 into A
        copy_matrix(IA, IA2, i + n, j, k, k + mid, N);   // copy IA2 into A
        copy_matrix(IA, IA3, i + n, j, k + n, l, N);     // copy IA3 into A
    }
}

/*
   the multi-threaded (parallel) version of inverse function
   A - input matrix
   IA - inverse of A
   B - threshold
*/
void parallel_inversse(double *A, double *IA, int i, int j, int k, int l, int N, int B)
{
    if (N <= B)
    {
        loop_serial_inverse(A, IA, N);
    }
    else
    { // n > B

        int mid = (j - i) / 2;
        int n = mid + 1;
        int size = n * n;
        // sub matrix
        double *a1 = (double *)malloc(size * sizeof(double));   //  A1
        double *a2 = (double *)malloc(size * sizeof(double));   //  A2
        double *a3 = (double *)malloc(size * sizeof(double));   //  A3
        double *IA1 = (double *)malloc(size * sizeof(double));  // inverse of A1
        double *IA2 = (double *)malloc(size * sizeof(double));  // inverse of A2
        double *IA3 = (double *)malloc(size * sizeof(double));  // inverse of A3
        double *mIA3 = (double *)malloc(size * sizeof(double)); // minus(IA3) i.e. -IA3
        double *W = (double *)malloc(size * sizeof(double));    // W = -IA3 * IA2

        // populate sub-matrix values
        cilk_spawn get_sub_matrix(A, a1, i, i + mid, k, k + mid, N);
        cilk_spawn get_sub_matrix(A, a2, i + n, j, k, k + mid, N);
        get_sub_matrix(A, a3, i + n, j, k + n, l, N);
        cilk_sync;
        // (1) IA1 - inverse of A1
        parallel_inversse(a1, IA1, i, i + mid, k, k + mid, n, B);

        // (2) IA3 - inverse of A3
        parallel_inversse(a3, IA3, i + n, j, k + n, l, n, B);

        // (3) IA2 - inverse of A2 = -IA3 * A2 * IA1
        minus_matrix(IA3, mIA3, size); // -IA3

        cilk_spawn mm_loop_serial(W, 0, n, mIA3, 0, n, a2, 0, n, n); // W = -IA3 * A2
        mm_loop_serial(IA2, 0, n, W, 0, n, IA1, 0, n, n);            // IA2 = W * IA1
        cilk_sync;

        // (4) Copy value into IA
        cilk_spawn copy_matrix(IA, IA1, i, i + mid, k, k + mid, N); // copy IA1 into A
        copy_matrix(IA, IA2, i + n, j, k, k + mid, N);              // copy IA2 into A
        copy_matrix(IA, IA3, i + n, j, k + n, l, N);                // copy IA3 into A
        cilk_sync;
    }
}

/*
    correctness tests: a couple examples with n = 4 (with B taking values 1, 2, 4)
    for which your code verifies that AA−1 equals the identity matrix;

    Regerence: Class Example: DnC_MM_With_analysis.cpp
*/
void correctness_tests()
{
    std::cout << "\n*** Correctness Tests (n = 4)***" << std::endl;

    int n = 4;
    int size = n * n;
    double *a;  // random created matrix
    double *ia; // invers of a
    double *id; // identity matrix

    vector<int> B_values = {1, 2, 4}; // test case B values

    int index = 0;
    for (const int B : B_values)
    {
        index++;

        printf("\n *** Test  Case # (%d) ***", index);
        printf("\n B =%d:\n", B);
        a = (double *)malloc(size * sizeof(double));
        ia = (double *)malloc(size * sizeof(double));
        id = (double *)malloc(size * sizeof(double));

        // Random createing matrix
        random_matrix(a, n);
        std::cout << "Input matrix:" << endl;
        print_matrix(a, n);

        struct timespec start_stl, end_stl;
        if (clock_gettime(CLOCK_MONOTONIC, &start_stl) < 0)
        {
            perror("clock_gettime");
            // return 1;
        }
        // test inverse function
        parallel_inversse(a, ia, 0, n - 1, 0, n - 1, n, B);

        if (clock_gettime(CLOCK_MONOTONIC, &end_stl) < 0)
        {
            perror("clock_gettime");
            // return 1;
        }

        std::cout << "Inversed matrix ia:" << endl;
        print_matrix(ia, 4);

        // r = a * ia
        mm_loop_serial(id, 0, n, a, 0, n, ia, 0, n, n);
        std::cout << "Check if id is identity:" << endl;
        is_identity_matrix(id, n);
        long run_time = print_time(std::cout, start_stl, end_stl);
        std::cout << " parallel_inversse took " << run_time << " mseconds." << endl;
    }
    free(a);
    free(ia);
    free(id);
}

/*
performance tests: tests for which n takes successive powers of 2, namely
4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 and B varies in the range 32, 64, 128.
*/
void performance_tests()
{

    int index = 0;

    double *a;  // random created matrix
    double *ia; // invers of a
    double *id; // identity matrix

     vector<vector<int>> test_data = {{32, 4}, {32, 8}, {64, 16}, {64, 32}, {128, 64}, {128, 128}, {512, 256}, {512, 512}, {256, 1024}, {256, 2048}};
    for (const vector<int> data : test_data)
    {
        int B = data[0];
        int n = data[1];

        int size = n * n;
        index++;

        printf("\n *** Test  Case # (%d) ***", index);
        printf("\n B =%d:\n", B);
        a = (double *)malloc(size * sizeof(double));
        ia = (double *)malloc(size * sizeof(double));
        id = (double *)malloc(size * sizeof(double));

        // Random createing matrix
        a = (double *)malloc(size * sizeof(double));
        random_matrix(a, n);
        std::cout << "Input matrix:" << endl;
        print_matrix(a, n);

        struct timespec start_stl, end_stl;
        if (clock_gettime(CLOCK_MONOTONIC, &start_stl) < 0)
        {
            perror("clock_gettime");
            // return 1;
        }
        // test inverse function
        parallel_inversse(a, ia, 0, n - 1, 0, n - 1, n, B);

        if (clock_gettime(CLOCK_MONOTONIC, &end_stl) < 0)
        {
            perror("clock_gettime");
            // return 1;
        }

        std::cout << "Inversed matrix ia:" << endl;
        print_matrix(ia, 4);

        // r = a * ia
        mm_loop_serial(id, 0, n, a, 0, n, ia, 0, n, n);
        std::cout << "Check if id is identity:" << endl;
        is_identity_matrix(id, n);
        long run_time = print_time(std::cout, start_stl, end_stl);
        std::cout << "parallel_inversse took " << run_time << " mseconds." << endl;
    }

    free(a);
    free(ia);
    free(id);
}

/*
    Test case for below functions:
        - loop_serial_inverse
        - mm_loop_serial
        - print_matrix

    Test Case (3 matrix) is from assignment
*/
void unit_tests_1(int n)
{
    cout << "\n*** Unit testing ***" << endl;
    int s = n * n;

    // input matrix: examples from assignment Problem 1 Question 3
    double a1[] = {1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 1.0};
    double a2[] = {1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0};
    double a3[] = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    // inversed matrix
    double *ia1 = (double *)malloc(s * sizeof(double));
    double *ia2 = (double *)malloc(s * sizeof(double));
    double *ia3 = (double *)malloc(s * sizeof(double));

    // r = a * ia
    double *r1 = (double *)malloc(s * sizeof(double));
    double *r2 = (double *)malloc(s * sizeof(double));
    double *r3 = (double *)malloc(s * sizeof(double));

    //  Test Case #1
    cout << "\nTest Case #1" << endl;
    cout << "Input matrix a1:" << endl;
    print_matrix(a1, 4);
    // test inversion function
    loop_serial_inverse(a1, ia1, n);
    cout << "Inversed matrix ia1:" << endl;
    print_matrix(ia1, 4);
    // r = a * ia
    mm_loop_serial(r1, 0, n, a1, 0, n, ia1, 0, n, n);
    cout << "Check if r1 is identity (r1 = a1 * ia1):" << endl;
    if (is_identity_matrix(r1, n))
    {
        cout << "Test case 1 passed! print r1:" << endl;
    }
    else
    {
        cout << "Test case 1 failed! print rr1:" << endl;
    }

    print_matrix(r1, 4);

    //  Test Case #2
    cout << "\nTest Case #2" << endl;
    cout << "Input matrix 2:" << endl;
    print_matrix(a2, 4);
    // test inversion function
    loop_serial_inverse(a2, ia2, n);
    cout << "Inversed matrix 2:" << endl;
    print_matrix(ia2, 4);
    // r = a * ia
    mm_loop_serial(r2, 0, n, a2, 0, n, ia2, 0, n, n);
    cout << "Check if r2 is identity: r2 = a 2* ia2:" << endl;
    if (is_identity_matrix(r1, n))
    {
        cout << "Test case 2 passed! print r2" << endl;
    }
    else
    {
        cout << "Test case 2 failed! print r2" << endl;
    }
    print_matrix(r2, 4);

    //  Test Case #3
    cout << "\nTest Case #3" << endl;
    cout << "Input matrix 3:" << endl;
    print_matrix(a3, 4);
    // test inversion function
    loop_serial_inverse(a3, ia3, n);
    cout << "Inversed matrix 1:" << endl;
    print_matrix(ia3, 4);
    // r = a * ia
    mm_loop_serial(r3, 0, n, a3, 0, n, ia3, 0, n, n);
    cout << "Check if r3 is identity: r3 = a 3* ia3:" << endl;
    if (is_identity_matrix(r1, n))
    {
        cout << "Test case 3 passed! print r3" << endl;
    }
    else
    {
        cout << "Test case 3 failed! print r3" << endl;
    }
    print_matrix(r3, 4);

    free(r1);
    free(r2);
    free(r3);
    free(ia1);
    free(ia2);
    free(ia3);
}

/*
    Testing function serial_inverse()
    and its invoked functions

    Test Case (3 matrix) is from assignment
*/
void unit_tests_2(int n, int B)
{
    std::cout << "\n*** Unit testing ***" << endl;
    int s = n * n;

    // input matrix: examples from assignment Problem 1 Question 3
    double a1[] = {1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, -1.0, -1.0, -1.0, 1.0};
    double a2[] = {1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0};
    double a3[] = {1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0};

    // inversed matrix
    double *ia1 = (double *)malloc(s * sizeof(double));
    double *ia2 = (double *)malloc(s * sizeof(double));
    double *ia3 = (double *)malloc(s * sizeof(double));

    // r = a * ia
    double *r1 = (double *)malloc(s * sizeof(double));
    double *r2 = (double *)malloc(s * sizeof(double));
    double *r3 = (double *)malloc(s * sizeof(double));

    //  Test Case #1
    std::cout << "\nTest Case #1" << endl;
    std::cout << "Input matrix a1:" << endl;
    print_matrix(a1, 4);
    // test inversion function
    serial_inverse(a1, ia1, 0, n - 1, 0, n - 1, n, B);
    std::cout << "Inversed matrix ia1:" << endl;
    print_matrix(ia1, 4);
    // r = a * ia
    mm_loop_serial(r1, 0, n, a1, 0, n, ia1, 0, n, n);
    std::cout << "Check if r1 is identity (r1 = a1 * ia1):" << endl;
    if (is_identity_matrix(r1, n))
    {
        std::cout << "Test case 1 passed! print r1:" << endl;
    }
    else
    {
        std::cout << "Test case 1 failed! print rr1:" << endl;
    }

    print_matrix(r1, 4);

    //  Test Case #2
    // cout << "\nTest Case #2" << endl;
    cout << "Input matrix 2:" << endl;
    print_matrix(a2, 4);
    // test inversion function
    serial_inverse(a2, ia2, 0, n - 1, 0, n - 1, n, B);
    cout << "Inversed matrix 2:" << endl;
    print_matrix(ia2, 4);
    // r = a * ia
    mm_loop_serial(r2, 0, n - 1, a2, 0, n - 1, ia2, 0, n, n);
    cout << "Check if r2 is identity: r2 = a 2* ia2:" << endl;
    if (is_identity_matrix(r1, n))
    {
        cout << "Test case 2 passed! print r2" << endl;
    }
    else
    {
        cout << "Test case 2 failed! print r2" << endl;
    }
    print_matrix(r2, 4);

    // Test Case #3
    cout << "\nTest Case #3" << endl;
    cout << "Input matrix 3:" << endl;
    print_matrix(a3, 4);
    // test inversion function
    serial_inverse(a3, ia3, 0, n - 1, 0, n - 1, n, B);
    cout << "Inversed matrix 1:" << endl;
    print_matrix(ia3, 4);
    // r = a * ia
    mm_loop_serial(r3, 0, n - 1, a3, 0, n - 1, ia3, 0, n, n);
    cout << "Check if r3 is identity: r3 = a 3* ia3:" << endl;
    if (is_identity_matrix(r1, n))
    {
        cout << "Test case 3 passed! print r3" << endl;
    }
    else
    {
        cout << "Test case 3 failed! print r3" << endl;
    }
    print_matrix(r3, 4);

    free(r1);
    free(r2);
    free(r3);
    free(ia1);
    free(ia2);
    free(ia3);
}

int main(int argc, char *argv[])
{
    correctness_tests();
    performance_tests();

    // unit_tests_1(4);
    // unit_tests_2(4, 2);
}
