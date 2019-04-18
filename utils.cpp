/**
 * @file    utils.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements common utility/helper functions.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "utils.h"
#include <iostream>

/*********************************************************************
 *                 Implement your own functions here                 *
 *********************************************************************/
#include "jacobi.h"
#include <math.h>


double l2norm(const int n, double* A, double* x, double* b) {
    double answer = 0;
    double Ax[n];

    matrix_vector_mult(n, A, x, Ax);

    for (int i = 0; i < n; i++) {
        answer += pow((Ax[i] - b[i]), 2.0);
    }
    return answer;
}
// ...

