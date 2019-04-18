/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <algorithm>

/*
 * TODO: Implement your solutions here
 */
#include <cstring>

void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);

    //need separate comm just for column
    // THIS TOOK ME TWO YEARS TO FIND
    // WHY THE FUCK CANT THIS BE INSIDE THE IF STATEMENT????
    // WHY THO WHY.
    MPI_Comm col_comm;
    int tf[2] = {true, false};
    MPI_Cart_sub(comm, tf, &col_comm);

    if (coords[1] == 0) {
        int sendcounts[dims[0]];
        int displs[dims[0]] = {0};
        sendcounts[0] = block_decompose(n, dims[0], 0);
        for (int i = 1; i < dims[0]; i++) {
            sendcounts[i] = block_decompose(n, dims[0], i);
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

        // How the fuck was I supposed to know local_vector is uninitialized?
        *local_vector = new double[sendcounts[coords[0]]];

        //finish with scatterv
        MPI_Scatterv(input_vector, sendcounts, displs, MPI_DOUBLE,
            *local_vector, sendcounts[coords[0]], MPI_DOUBLE, 0, col_comm);
    }
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);

    //need separate comm just for column
    MPI_Comm col_comm;
    int tf[2] = {true, false};
    MPI_Cart_sub(comm, tf, &col_comm);

    if (coords[1] == 0) {
        int recvcounts[dims[0]];
        int displs[dims[0]] = {0};
        recvcounts[0] = block_decompose(n, dims[0], 0);
        for (int i = 1; i < dims[0]; i++) {
            recvcounts[i] = block_decompose(n, dims[0], i);
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
        //finish with scatterv
        MPI_Gatherv(local_vector, recvcounts[coords[0]], MPI_DOUBLE,
            output_vector, recvcounts, displs, MPI_DOUBLE, 0, col_comm);
    }
}

// https://stackoverflow.com/questions/9269399/sending-blocks-of-2d-array-in-c-using-mpi/9271753#9271753
void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm) {

    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);

    // Activate sub comm for later use.
    MPI_Comm row_comm, col_comm;
    int ft[2] = {false, true};
    int tf[2] = {true, false};
    MPI_Cart_sub(comm, ft, &row_comm);
    MPI_Cart_sub(comm, tf, &col_comm);


    double local_vector[(n / dims[0] + 1) * n]; // used by 1st col ONLY

    int sendcounts[dims[0]];
    int displs[dims[0]] = {0};

    // First scatterv to column
    if (coords[1] == 0) {
        sendcounts[0] = block_decompose(n, dims[0], 0) * n;
        for (int i = 1; i < dims[0]; i++) {
            sendcounts[i] = n * block_decompose(n, dims[0], i);
            displs[i] = displs[i - 1] + sendcounts[i - 1];
        }

        //finish with scatterv
        MPI_Scatterv(input_matrix, sendcounts, displs, MPI_DOUBLE,
            local_vector, sendcounts[coords[0]], MPI_DOUBLE, 0, col_comm);

    }

    int num_rows, num_cols;
    num_rows = block_decompose(n, dims[0], coords[0]);
    num_cols = block_decompose(n, dims[1], coords[1]);

    // re initialize sendcounts and displs
    displs[0] = 0;
    sendcounts[0] = block_decompose(n, dims[1], 0);
    for (int i = 1; i < dims[1]; i++) {
        sendcounts[i] = block_decompose(n, dims[1], i);
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }


    // OF COURSE THIS ALSO DOESNT HAVE SPACE
    (*local_matrix) = new double[num_rows * num_cols];

    // yeah fuck a constant tau
    // just send it row by row a million times!
    for (int i = 0; i < num_rows; i++) {
        MPI_Scatterv(&local_vector[i * n], sendcounts, displs, MPI_DOUBLE,
            &((*local_matrix)[i * num_cols]), num_cols, MPI_DOUBLE, 0, row_comm);
    }


    // // //////////////////////////////////////////////////////////////// // //
    // // /////////////////// To hell with smart stuff /////////////////// // //
    ////////////////////////////////////////////////////////////////////////////
    // MPI_Datatype col_wise_block;
    // MPI_Type_vector(num_rows, 1, n, MPI_DOUBLE, &col_wise_block);
    // MPI_Type_commit(&col_wise_block);

    // // re initialize sendcounts and displs
    // displs[0] = 0;
    // sendcounts[0] = block_decompose(n, dims[1], 0);
    // for (int i = 1; i < dims[1]; i++) {
    //     sendcounts[i] = block_decompose(n, dims[1], i);
    //     displs[i] = displs[i - 1] + sendcounts[i - 1];
    // }

    // double local_1d_matrix[num_rows * num_cols];
    // MPI_Scatterv(local_vector, sendcounts, displs, col_wise_block,
    //     local_1d_matrix, num_rows * sendcounts[coords[1]], MPI_DOUBLE, 0, row_comm);

    // // OF COURSE THIS ALSO DOESNT HAVE SPACE
    // (*local_matrix) = new double[num_rows * num_cols];
    // for (int i = 0; i < num_rows; i++) {
    //     for (int j = 0; j < num_cols; j++)
    //     {
    //         (*local_matrix)[i * num_cols + j] = local_1d_matrix[j * num_rows + i];
    //     }
    // }
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm) {

    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);

    MPI_Comm row_comm, col_comm;
    int tf[2] = {true, false};
    int ft[2] = {false, true};
    MPI_Cart_sub(comm, tf, &col_comm);
    MPI_Cart_sub(comm, ft, &row_comm);


    // How many gets more?
    int count = block_decompose(n, dims[0], coords[0]);
    if (coords[0] == 0 && coords[1] == 0) {
        memcpy(row_vector, col_vector, count * sizeof(double));
    } else if (coords[1] == 0) {
        // first move the first column's data to the diagonal
        MPI_Send(col_vector, count, MPI_DOUBLE, coords[0], 44, row_comm);
    } else if (coords[0] == coords[1]) {
        MPI_Recv(row_vector, count, MPI_DOUBLE, 0, 44, row_comm, MPI_STATUS_IGNORE);
    }

    count = block_decompose(n, dims[0], coords[1]);
    MPI_Bcast(row_vector, count, MPI_DOUBLE, coords[1], col_comm);
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);


    // then figure out matrix size
    int num_rows = block_decompose(n, dims[0], coords[0]);
    int num_cols = block_decompose(n, dims[1], coords[1]);

    double x[num_cols];

    transpose_bcast_vector(n, local_x, x, comm);

    double y[num_rows] = {0};
    matrix_vector_mult(num_rows, num_cols, local_A, x, y);

    MPI_Comm row_comm;
    int ft[2] = {false, true};
    MPI_Cart_sub(comm, ft, &row_comm);


    MPI_Reduce(y, local_y, num_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination) {
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);
    // then figure out matrix size
    int num_rows = block_decompose(n, dims[0], coords[0]);
    int num_cols = block_decompose(n, dims[1], coords[1]);
    // prepare sub comm
    MPI_Comm row_comm, col_comm;
    int ft[2] = {false, true};
    int tf[2] = {true, false};
    MPI_Cart_sub(comm, ft, &row_comm);
    MPI_Cart_sub(comm, tf, &col_comm);


    // Create D
    double* d, * y, * r;

    // All diagonals are along the diagonal processors
    if (coords[0] == coords[1]) {
        d = new double[num_rows];
        r = new double[num_rows * num_cols];
        memcpy(r, local_A, num_rows * num_cols * sizeof(double));

        for (int i = 0; i < num_rows; i++) {
            d[i] = local_A[i * (num_cols + 1)];
            r[i * (num_cols + 1)] = 0;
        }
        if (coords[0] != 0) {
            MPI_Send(d, num_rows, MPI_DOUBLE, 0, 77, row_comm);
            delete[] d;
        }
    } else {
        r = local_A;
    }


    if (coords[1] == 0) {
        y = new double[num_rows];
    }

    // The corresponding Recv for D
    if (coords[1] == 0 && coords[0] != 0) {
        d = new double[num_rows];
        MPI_Recv(d, num_rows, MPI_DOUBLE, coords[0], 77, row_comm, MPI_STATUS_IGNORE);
    }

    int should_continue = 1;
    memset(local_x, 0, num_rows * sizeof(double));
    for (int i = 0; i < max_iter; i++) {
        // R * x
        distributed_matrix_vector_mult(n, r, local_x, y, comm);


        // x = (b - R*x) - D
        if (coords[1] == 0) {
            for (int j = 0; j < num_rows; j++) {
                local_x[j] = (local_b[j] - y[j]) / d[j];
            }
        }

        // A * xw1
        distributed_matrix_vector_mult(n, local_A, local_x, y, comm);

        if (coords[1] == 0) {
            double temp = 0;
            for (int i = 0; i < num_rows; i++) {
                temp += (local_b[i] - y[i]) * (local_b[i] - y[i]);
            }
            double l2norm_sum;

            MPI_Allreduce(&temp, &l2norm_sum, 1, MPI_DOUBLE, MPI_SUM, col_comm);
            should_continue = sqrt(l2norm_sum) < l2_termination ? 0 : 1;
        }


        MPI_Bcast(&should_continue, 1, MPI_INTEGER, 0, row_comm);


        if (should_continue == 0) {
            if (coords[0] == coords[1]) {
                delete[] r;
            }
            if (coords[1] == 0) {
                delete[] d;
                delete[] y;
            }
            return;
        }
    }
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}

