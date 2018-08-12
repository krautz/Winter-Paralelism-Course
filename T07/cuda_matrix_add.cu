#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void matrix_add (int *device_A, int *device_B, int *device_C, int *device_n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < *device_n)
        device_C[index] = device_A[index] + device_B[index];
}

int main()
{
    int *host_A, *host_B, *host_C, *host_n;
    int *device_A, *device_B, *device_C, *device_n;
    int i, j;

    //Input
    int linhas, colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando memória na CPU
    host_A = (int *)malloc(sizeof(int)*linhas*colunas);
    host_B = (int *)malloc(sizeof(int)*linhas*colunas);
    host_C = (int *)malloc(sizeof(int)*linhas*colunas);
    host_n = (int *)malloc (sizeof(int));

    *host_n = linhas*colunas;

    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            host_A[i*colunas+j] =  host_B[i*colunas+j] = i+j;
        }
    }

    cudaMalloc ((void **) &device_A, sizeof(int)*linhas*colunas);
    cudaMalloc ((void **) &device_B, sizeof(int)*linhas*colunas);
    cudaMalloc ((void **) &device_C, sizeof(int)*linhas*colunas);
    cudaMalloc ((void **) &device_n, sizeof(int));

    cudaMemcpy (device_A, host_A, sizeof(int)*linhas*colunas, cudaMemcpyHostToDevice);
    cudaMemcpy (device_B, host_B, sizeof(int)*linhas*colunas, cudaMemcpyHostToDevice);
    cudaMemcpy (device_n, host_n, sizeof(int), cudaMemcpyHostToDevice);

    matrix_add<<<linhas,colunas>>> (device_A, device_B, device_C, device_n);

    cudaMemcpy (host_C, device_C, sizeof(int)*linhas*colunas, cudaMemcpyDeviceToHost);

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=host_C[i*colunas+j];
        }
    }

    printf("%lli\n", somador);

    free(host_A);
    free(host_B);
    free(host_C);
    free(host_n);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    cudaFree(device_n);
}
