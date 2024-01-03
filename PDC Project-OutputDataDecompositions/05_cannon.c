#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define MATRIX_SIZE 3

int main(int argc, char *argv[]) {
    int rank, size, row = 0, column = 0, count = 0, i = 0, j = 0, k = 0;
    int *A, *B, *C, a = 0, b = 0, c = 0;
    double start_time, end_time;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(time(NULL));

    start_time = MPI_Wtime();

    if (rank == 0) {
        row = MATRIX_SIZE;
        column = MATRIX_SIZE;
        count = size;

        A = (int *)calloc(sizeof(int), row * column);
        B = (int *)calloc(sizeof(int), row * column);

        k = 0;
        printf("Matrix A:\n");
        for (i = 0; i < row; i++) {
            for (j = 0; j < column; j++) {
                A[k] = i+1;
                printf("%d\t", A[k]);
                k++;
            }
            printf("\n");
        }

        k = 0;
        printf("\nMatrix B:\n");
        for (i = 0; i < row; i++) {
            for (j = 0; j < column; j++) {
                B[k] = j+1;
                printf("%d\t", B[k]);
                k++;
            }
            printf("\n");
        }
    }

    MPI_Bcast(&row, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int periods[] = {1, 1}; // both vertical and horizontal movement;
    int dims[] = {row, row};
    int coords[2];          /* 2 Dimension topology so 2 coordinates */
    int right = 0, left = 0, down = 0, up = 0; // neighbor ranks
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Scatter(A, 1, MPI_INT, &a, 1, MPI_INT, 0, cart_comm);
    MPI_Scatter(B, 1, MPI_INT, &b, 1, MPI_INT, 0, cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    MPI_Cart_shift(cart_comm, 1, coords[0], &left, &right);
    MPI_Cart_shift(cart_comm, 0, coords[1], &up, &down);
    MPI_Sendrecv_replace(&a, 1, MPI_INT, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(&b, 1, MPI_INT, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);
    c = c + a * b;
    for (i = 1; i < row; i++) {
        MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
        MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(&a, 1, MPI_INT, left, 11, right, 11, cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(&b, 1, MPI_INT, up, 11, down, 11, cart_comm, MPI_STATUS_IGNORE);
        c = c + a * b;
    }

    C = (int *)calloc(sizeof(int), row * row);
    MPI_Gather(&c, 1, MPI_INT, C, 1, MPI_INT, 0, cart_comm);
    if (rank == 0) {
        k = 0;
        printf("\nResult A * B:\n");
        for (i = 0; i < row; i++) {
            for (j = 0; j < row; j++) {
                printf("%d\t", C[k]);
                k++;
            }
            printf("\n");
        }
    }

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Execution Time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}