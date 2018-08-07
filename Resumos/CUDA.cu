/* Loop DOALL -> loop em que todos termos do vetor sao usados e nao ha dependencia entre  iteracoes -> PARALELIZAVEL
 * Loop DOACROSS-> uma iteração depende de uma outra iteração -> nao paralelizavel.
 * Regra do 80/20 -> 80% da execução do programa ocorre em 20% das instrucoes do codigo (se for do all é paralelizavel).
 * Se esse loop for do all e executar a mesma intrução não ha necessidade de cada core ter a sua I-cache -> ela e grande e pode ser eliminada -> espaço para mais cores -> GPU
 * CUDA -> linguagem para programação em GPU que permite computação heterogenia (CPU + GPU)
 * Terminologia: Host -> CPU e sua memoria; Device -> GPU e sua memoria;
 * Fluxo do programa: CPU transfere dados para a memoria da GPU (esse processo é muito custoso e por isos as vezes não compensa utilizar a GPU); GPU executa o kernel em paralelo; GPU transefere novamente o dado para CPU (ou DRAM).
 * A GPU é organizada em SMX, cada SMX tem um conjunto de blocos e cada bloco tem um conjunto de threads. o work load pe despachado da SMX para os blocos em wraps q sao conjuntos de threads (normalmente 32).
 * A GPU tem uma memória local acessada por todas SMX e uma memória local compartilhada (shared memory) que é mais rápida que a global, porém restringida ao uso apenas dentro de uma SMX.
 *
 *
 * Criando um kernel:
 * keyword -> __global__ (significa q a função sera chamada pelo host e executada no device).
 * __global__ void mykernel (void) {

  }
 *
 *
 * Invocando um Kernel:
 * mykernel<<<N,M>>> () -> N é numero de blocos e M numero de threads por bloco;
 *
 *
 * Precisamos alocar memoria na GPU para executar um kernel:
 * Ponteiros da GPU apontam para memoria na GPU e pontiero da CPU para memória da CPU.
 * cudaMalloc((void **) &variavel_cuda, tamanho_variavel);
 * cudaFree(variavel_cuda);
 * cudaMemcpy(variavel_cuda, variavel_CPU, tamanho, cudaMemcpyHostToDevice);
 * cudaMemcpy (variavel_CPU, variavel_cuda, tamanho, cudaMemcpyDeviceToHost);
 *
 *
 * Um kernel é executado uma grid de blocos, que é 3D, e temos algumas palavras reservadas:
 * threadIdx.x/y/z
 * blockIdx.x/y/z
 * blockDim.x/y/z -> variavel N setada na chamada do kernel na main.
 * gridDim -> N*M;
 * Podemos combinar varios blocos e varias threads, e o calculo de indice complica um pouco (int index = threadIdx.x + blockIdx.x * blockDim.x);
 * Vale a pena usar threads pois elas podem sincronizar e se comunicar!
 *
 *
 * Como acessar a memoria compartilhada do bloco:
 * palavra reservada __shared__ (usa dentro do kernel para variaveis criadas dentro do kernel. Exemplo: __shared__ int variavel).
 * o dado não é visível para threads em outros blocos.
 *
 *
 * __syncthreads(): usado para previnir hazard the read/write after write/read:
 * todas threads de um BLOCO devem chegar ate aquele ponto para progredir.
 *
 *
 * Coordenação:
 * as chamadas de kernels sao feitas de forma asincrona!
 * A CPU precisa sincronizar antes de ler os resultados da computação no device:
 * cudaMemcpy(); -> bloqueia a CPU e so retoma quando termina de copiar (e logo termina a computação da GPU)
 * cudaMemcpyAsync(); -> asincrono, nao bloqueia a GPU
 * cudaDevideSynchronize(); ->bloqueia a CPU ate que todos processos da GPU terminem.
 *
 *
 * Device Management:
 * cudaGetDeviceCount(int *count)
 * cudaSetDevice(int device)
 * cudaGetDevice(int *device)
 * cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
 *
 *
 * Memoria:
 * uma linha inteira da DRAM é lida para o row buffer (memory core speed -> lento).
 * dessa linha um MUX é usado para selecionar qual word voce realmente quer (interface speed -> rapido)
 * DDRx -> memory core speed = 1/(2^x) interface speed.
 * para compensar essa lentidao uma linha da ram tem 2^x de bits em largura, e todos esses bits sao uma linha que vao para o row buffer, que seleciona a word desejada (isso é chamado de burst -> trazer varias palavras para o row buffer).
 * Também, aliado a isso pode-se ter varios bancos, um colados aos outros de DRAM, que operam em paralelo e aumentam o bandwidth (largura de banda -> bits/s transferidos).
 *
 *
 * Memory Coalescing:
 * O acesso a memoria é dito coalesced se todos endereços do burst foram usados e uncoalesced se nao usou todos endereços.
 * um acesso não é coalesced se uma das palavras reservadas (threadIdx, BlockIdx, etc) for multiplicada por uma constante.
 *
 *
 * Control divergence:
 * se um mesmo wrap executa um bloco com if's entao ha perda de eficiencia pois cada thread pode executar o bloco if ou o bloco else, introduzindo "nop's" ao que ele nao executar, aumentando assim o tempo de execução do programa.
 */

 #include <stdio.h>
 #include <stdlib.h>

 __global__ void add_vector (int * device_A, int * device_B, int * device_C) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    device_C[index] = device_A[index] + device_B[index];
 }

 int main () {
    int tamanho_dos_vetores, i;
    int *host_A, *host_B, *host_C;
    int *device_A, *device_B, *device_C;

    printf("Insira o tamanho dos vetores (multiplo de 4 e pelo menos 4): \n");
    scanf("%d", &tamanho_dos_vetores);

    host_A = (int *) malloc (tamanho_dos_vetores * sizeof(int));
    host_B = (int *) malloc (tamanho_dos_vetores * sizeof(int));
    host_C = (int *) malloc (tamanho_dos_vetores * sizeof(int));

    cudaMalloc ((void ** ) &device_A, tamanho_dos_vetores * sizeof(int));
    cudaMalloc ((void ** ) &device_B, tamanho_dos_vetores * sizeof(int));
    cudaMalloc ((void ** ) &device_C, tamanho_dos_vetores * sizeof(int));

    printf("Insira os %d numeros do 1o vetor: \n", tamanho_dos_vetores);
    for (i = 0; i < tamanho_dos_vetores; i++) {
      scanf("%d", &host_A[i]);
    }
    printf("Insira os %d numeros do 2o vetor: \n", tamanho_dos_vetores);
    for (i = 0; i < tamanho_dos_vetores; i++) {
      scanf("%d", &host_B[i]);
    }
    cudaMemcpy (device_A, host_A, tamanho_dos_vetores * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (device_B, host_B, tamanho_dos_vetores * sizeof(int), cudaMemcpyHostToDevice);

    add_vector<<<4, tamanho_dos_vetores/4>>>(device_A, device_B, device_C);

    cudaMemcpy (host_C, device_C, tamanho_dos_vetores * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultado da Soma: \n");
    printf("%d", host_C[0]);
    for (i = 1; i < tamanho_dos_vetores; i++) {
      printf(" %d", host_C[i]);
    }
    printf("\n");

    free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    return 0;
 }
