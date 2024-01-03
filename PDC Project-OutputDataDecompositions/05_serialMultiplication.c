#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 800

void printMatrix(int C[N][N]){
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d      ", C[i][j]);
        }
        printf("\n");
    }
}

void input(int A[N][N]) {
    srand(time(NULL));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = rand()%10 +1;
        }
    }
}

int main()
{
    int matrixA[N][N], matrixB[N][N], matrixC[N][N];

    input(matrixA);
    input(matrixB);

    printf("\nMatrix A:\n");
    printMatrix(matrixA);

    printf("\nMatrix B:\n");
    printMatrix(matrixB);

    clock_t start_time = clock();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrixC[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                matrixC[i][j] += (matrixA[i][k] * matrixB[k][j]);
            }
        }
    }
    clock_t end_time = clock();

    printf("\nResult Matrix C:\n");
    printMatrix(matrixC);

    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("\nTotal Execution Time: %lf seconds\n", total_time);

    return 0;
}