/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <math.h>
#include <string.h>
#include "utils.h"
#include <iostream>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < n; j++) {
            y[i] += A[i*n + j] * x[j];
        }
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        y[i] = 0;
        for (int j = 0; j < m; j++) {
            y[i] += A[i*m + j] * x[j];
        }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // init x to zero just in case
    memset(x, 0, sizeof(double) * n);

    // to use in while loop
    double tmp_Rx[n];
    int iter = 0;

    // while ||Ax−b|| > l update x ← D−1(b−Rx)
    while(sqrt(l2norm(n, A, x, b)) > l2_termination && iter < max_iter) {
        for (int i = 0; i < n; i++) {
            tmp_Rx[i] = b[i];

            // b - R*x
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    tmp_Rx[i] -= A[i*n + j] * x[j];
                }
            }

            // D^(-1) * (b - R*x)
            tmp_Rx[i] /= A[i*n + i];
        }

        iter += 1;

        // update x
        memcpy(x, tmp_Rx, sizeof(double) * n);
    }
}
