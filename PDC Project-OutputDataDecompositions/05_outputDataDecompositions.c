#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define N 800

void printMatrix(int C[N][N]) {
    printf("\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d      ", C[i][j]);
        }
        printf("\n");
    }
}

void input(int A[N][N]) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10 + 1;
        }
    }
}

void multiplyOption1(int matrixA[N][N], int matrixB[N][N], int matrixC[N][N], int offset, int rowsPerProcess) {
    #pragma omp parallel for num_threads(N/2)
    for (int i = offset; i < offset + rowsPerProcess; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < N; k++) {
                matrixC[i][j] += (matrixA[i][k] * matrixB[k][j]);
            }
        }
    }
}

void multiplyOption2(int matrixA[N][N], int matrixB[N][N], int matrixC[N][N], int offset, int rowsPerProcess) {
    #pragma omp parallel for num_threads(N/2)
    for (int i = offset + rowsPerProcess - 1; i >= offset; i--) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < N; k++) {
                matrixC[i][j] += (matrixA[i][k] * matrixB[k][j]);
            }
        }
    }
}

void multiplyOption3(int matrixA[N][N], int matrixB[N][N], int matrixC[N][N], int offset, int rowsPerProcess) {
    #pragma omp parallel for num_threads(N/2)
    for (int i = offset; i < offset + rowsPerProcess; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
            for (int k = 1; k < N - 1; k++) {
                matrixC[i][j] += (matrixA[i][k] * matrixB[k][j]);
            }
        }
    }

    // Multiply with the first row
    #pragma omp parallel for num_threads(N/2)
    for (int j = 0; j < N; j++) {
        matrixC[offset][j] = 0;
        for (int k = 0; k < N; k++) {
            matrixC[offset][j] += (matrixA[offset][k] * matrixB[k][j]);
        }
    }

    // Multiply with the last row
    #pragma omp parallel for num_threads(N/2)
    for (int j = 0; j < N; j++) {
        matrixC[offset + rowsPerProcess - 1][j] = 0;
        for (int k = 0; k < N; k++) {
            matrixC[offset + rowsPerProcess - 1][j] += (matrixA[offset + rowsPerProcess - 1][k] * matrixB[k][j]);
        }
    }
}

int main(int argc, char **argv) {
    int rank, size, offset, rowsPerProcess;
    int matrixA[N][N], matrixB[N][N], matrixC[N][N];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rowsPerProcess = N / (size - 1);

    srand(time(NULL));

    int option=atoi(argv[1]);

    if (rank == 0) {
        input(matrixA);
        input(matrixB);

        printMatrix(matrixA);
        printMatrix(matrixB);

        double startmpi = MPI_Wtime();
        offset = 0;
        for (int i = 1; i < size; i++) {
            MPI_Send(&offset, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&matrixA[offset][0], rowsPerProcess * N, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&matrixB, N * N, MPI_INT, i, 3, MPI_COMM_WORLD);
            offset += rowsPerProcess;
        }

        for (int i = 1; i < size; i++) {
            MPI_Recv(&offset, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&matrixC[offset][0], rowsPerProcess * N, MPI_INT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double endmpi = MPI_Wtime();

        printMatrix(matrixC);

        printf("\nExecution time is: %lf seconds\n\n", (endmpi - startmpi));
    } else {
        // slave process
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&matrixA[offset][0], rowsPerProcess * N, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&matrixB, N * N, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process # %d, Offset: %d\n", rank, offset);

        switch (option) {
            case 1:
                multiplyOption1(matrixA, matrixB, matrixC, offset, rowsPerProcess);
                break;
            case 2:
                multiplyOption2(matrixA, matrixB, matrixC, offset, rowsPerProcess);
                break;
            case 3:
                multiplyOption3(matrixA, matrixB, matrixC, offset, rowsPerProcess);
                break;
            default:
                printf("Invalid option\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&matrixC[offset][0], rowsPerProcess * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}