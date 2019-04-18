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

/*
 * TODO: Implement your solutions here
 */
#include <iostream> // [DEBUG] get rid of before submit

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
    // std::cout << wraparound[0] << " " << wraparound[1] << "\n";

    if (coords[1] == 0) {
        int recvcounts[dims[0]];
        int displs[dims[0]] = {0};
        recvcounts[0] = block_decompose(n, dims[0], 0);
        for (int i = 1; i < dims[0]; i++) {
            recvcounts[i] = block_decompose(n, dims[0], i);
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        //need separate comm just for column
        MPI_Comm col_comm;
        int tf[2] = {true, false};
        MPI_Cart_sub(comm, tf, &col_comm);

        //finish with scatterv
        MPI_Gatherv(local_vector, recvcounts[coords[0]], MPI_DOUBLE,
            output_vector, recvcounts, displs, MPI_DOUBLE, 0, col_comm);
    }
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // std::cout << "dist_MATRIX entered!!\n"; // [DEBUG]
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);
    // std::cout << "  dist_MATRIX entered!! (" << coords[0] << ", " << coords[1] << ")\n"; // [DEBUG]

    // Activate sub comm for later use.
    MPI_Comm row_comm, col_comm;
    int ft[2] = {false, true};
    int tf[2] = {true, false};
    MPI_Cart_sub(comm, ft, &row_comm);
    MPI_Cart_sub(comm, tf, &col_comm);


    int sendcounts[dims[0]];
    int displs[dims[0]] = {0};


    double local_vector[(n / dims[0] + 1) * n]; // used by 1st col ONLY
    // double local_1d_matrix[(num_elements + 1) * (num_elements + 1)];

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

    // yeah fuck a constant tau
    // just send it row by row a million times!
    for (int i = 0; i < num_rows; i++) {
        MPI_Scatterv(&local_vector[i * n], sendcounts, displs, MPI_DOUBLE,
            &local_matrix[i * num_cols], num_cols, MPI_DOUBLE, 0, row_comm);
    }

    // // //////////////////////////////////////////////////////////////// // //
    // // /////////////////// To hell with smart stuff /////////////////// // //
    ////////////////////////////////////////////////////////////////////////////
    // MPI_Datatype col_wise_block;
    // // Scatter across rows
    // if (coords[0] < mod_value) {
    //     // The ones who have 1 more line each
    //     MPI_Type_vector(num_elements + 1, 1, n, MPI_DOUBLE, &col_wise_block);
    //     std::cout << "      MPI_Type_vector 'if' done\n"; // [DEBUG]
    // } else {
    //     // The ones who have num_elements number of lines
    //     MPI_Type_vector(num_elements, 1, n, MPI_DOUBLE, &col_wise_block);
    //     std::cout << "      MPI_Type_vector 'else' done\n"; // [DEBUG]
    // }
    // MPI_Type_commit(&col_wise_block);

    // // re-initialize the sendcounts, displacements
    // memset(displs, 0, dims[0]); // reset displs to 0
    // sendcounts[0] = block_decompose(n, dims[1], 0);
    // std::cout << "  sendcounts: " << sendcounts[0] << " ";
    // for (int i = 1; i < dims[1]; i++) {
    //     sendcounts[i] = block_decompose(n, dims[1], i);
    //     displs[i] = displs[i - 1] + sendcounts[i - 1];
    //     std::cout << sendcounts[i] << " ";
    // }

    // std::cout << "\n  Hi im prob the problem\n"; //[DEBUG]
    // MPI_Scatterv(local_vector, sendcounts, displs, col_wise_block,
    //         local_1d_matrix, sendcounts[coords[1]], MPI_DOUBLE, 0, row_comm);
    // std::cout << "  Maybe not (" << coords[0] << ", " << coords[1] << ")\n\n\n"; // [DEBUG]

    // // local calculations to reshape it back into 2D
    // if (coords[0] < mod_value && coords[1] < mod_value) {
    //     // largest
    //     for (int i = 0; i < num_elements + 1; i++) {
    //         for (int j = 0; j < num_elements + 1; j++) {
    //             local_matrix[i][j] = local_1d_matrix[j * (num_elements + 1) + i];
    //         }
    //     }
    // } else if (coords[0] < mod_value) {
    //     for (int i = 0; i < num_elements + 1; i++) {
    //         for (int j = 0; j < num_elements; j++) {
    //             local_matrix[i][j] = local_1d_matrix[j * (num_elements + 1) + i];
    //         }
    //     }
    // } else if (coords[1] < mod_value) {
    //     for (int i = 0; i < num_elements; i++) {
    //         for (int j = 0; j < num_elements + 1; j++) {
    //             local_matrix[i][j] = local_1d_matrix[j * num_elements + i];
    //         }
    //     }
    // } else {
    //     for (int i = 0; i < num_elements; i++) {
    //         for (int j = 0; j < num_elements; j++) {
    //             local_matrix[i][j] = local_1d_matrix[j * num_elements + i];
    //         }
    //     }
    // }
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // first figure out dimensions
    int dims[2], wraparound[2], coords[2];
    MPI_Cart_get(comm, 2, dims, wraparound, coords);
    // std::cout << "        BCAST_vector start! (" << coords[0] << ", " << coords[1] << ")\n"; // [DEBUG]

    MPI_Comm row_comm, col_comm;
    int tf[2] = {true, false};
    int ft[2] = {false, true};
    MPI_Cart_sub(comm, tf, &col_comm);
    // std::cout << "        first Cart_sub passed\n"; // [DEBUG]
    MPI_Cart_sub(comm, ft, &row_comm);
    // std::cout << "        second Cart_sub passed\n"; // [DEBUG]


    // How many gets more?
    // std::cout << "        hooooo???" << dims[0] << " " << coords[0] << "\n"; // [DEBUG]
    int count = block_decompose(n, dims[0], coords[0]);
    // std::cout << "        hooooooeh???\n\n\n"; // [DEBUG]
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
    std::cout << "    vector_MULT start! (" << coords[0] << ", " << coords[1] << ")\n"; // [DEBUG]


    // then figure out matrix size
    int num_rows = block_decompose(n, dims[0], coords[0]);
    int num_cols = block_decompose(n, dims[1], coords[1]);

    // double* x = new double[num_cols];
    double x[num_cols];
    transpose_bcast_vector(n, local_x, x, comm);
    std::cout << "    BCAST_vector success! (" << coords[0] << ", " << coords[1] << ")\n"; // [DEBUG]

    // double* y = new double[num_rows];
    double y[num_rows];
    matrix_vector_mult(num_rows, num_cols, local_A, x, y);
    std::cout << "    SEQ_VECTOR_MULT success!!\n"; // [DEBUG]

    MPI_Comm row_comm;
    int ft[2] = {false, true};
    MPI_Cart_sub(comm, ft, &row_comm);

    MPI_Reduce(y, local_y, num_rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    // delete[] x;
    // delete[] y;
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // Get the Cartesian topology information
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);

    // Compute the dimentions of local matrix
    int nrows = block_decompose(n, dims[0], coords[0]);
    int ncols = block_decompose(n, dims[1], coords[1]);

    // Create row and column communicators
    MPI_Comm comm_col;
    int remain_dims[2] = {true, false};
    MPI_Cart_sub(comm, remain_dims, &comm_col);

    MPI_Comm comm_row;
    remain_dims[0] = false; remain_dims[1] = true;
    MPI_Cart_sub(comm, remain_dims, &comm_row);

    // Compute the diagonal elements and matrix R = A - D
    double* local_D = new double[nrows];
    double* local_R = new double[nrows * ncols];
    if (coords[0] == coords[1]) {
        std::copy(local_A, local_A + nrows * ncols, local_R);
        for (int i = 0; i < nrows; i++) {
            local_D[i] = local_A[i * (ncols + 1)];
            local_R[i * (ncols + 1)] = 0;
        }
        if (coords[0] != 0) {
            int dest_rank;
            int dest_coords[] = {0};
            MPI_Cart_rank(comm_row, dest_coords, &dest_rank);
            MPI_Send(local_D, nrows, MPI_DOUBLE, dest_rank, 0, comm_row);
        }
    } else {
        std::copy(local_A, local_A + nrows * ncols, local_R);
    }

    if (coords[1] == 0 && coords[0] != 0) {
        int source_rank;
        int source_coords[] = {coords[0]};
        MPI_Cart_rank(comm_row, source_coords, &source_rank);
        MPI_Recv(local_D, nrows, MPI_DOUBLE, source_rank, 0, comm_row, MPI_STATUS_IGNORE);
    }

    // Initialize x to zero
    if (coords[1] == 0) {
        for (int i = 0; i < nrows; i++) local_x[i] = 0;
    }

    double l2_norm_total;

    int iter = 1;
    while (iter <= max_iter) {

        double* local_y = new double[nrows];

        distributed_matrix_vector_mult(n, local_R, local_x, local_y, comm);

        if (coords[1] == 0) {
            for (int i = 0; i < nrows; i++) {
                local_x[i] = (local_b[i] - local_y[i]) / local_D[i];
            }
        }

        distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

        if (coords[1] == 0) {
            double l2_norm = 0.0;
            for (int i = 0; i < nrows; i++) {
                l2_norm += (local_b[i] - local_y[i]) * (local_b[i] - local_y[i]);
            }
            MPI_Allreduce(&l2_norm, &l2_norm_total, 1, MPI_DOUBLE, MPI_SUM, comm_col);
        }

        delete [] local_y;

        l2_norm_total = sqrt(l2_norm_total);

        int root_rank;
        int root_coords[] = {0};
        MPI_Cart_rank(comm_row, root_coords, &root_rank);
        MPI_Bcast(&l2_norm_total, 1, MPI_DOUBLE, root_rank, comm_row);

        if (l2_norm_total <= l2_termination) {
            return;
        }

        iter++;
    }

    // // Free the column communicator
    // MPI_Comm_free(&comm_col);
    // MPI_Comm_free(&comm_row);

    delete [] local_D;
    delete [] local_R;
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

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}

