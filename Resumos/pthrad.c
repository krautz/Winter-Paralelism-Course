/*
 * #include <pthread.h>
 * pthread_t *thread_handle -> usado para controle pelo SO.
 * thread_handle = malloc (thread_count * sizeof(pthread_t))
 *
 *
 * CRIAR THREAD:
 * pthread_create (endereço do handle, NULL, rotina de inicio (deve ser do tipo void*), lista de    argumentos (tambem deve ser void*));
 * pthread_create (&thread_handle[thread], NULL, Hello, (void *) thread);
 *
 *
 * JOIN:
 * A funcao so retorna algo quando a thread encerrar -> usado para a main esperar a exec de todas threads!
 * pthread_join (thread_handle[thread], NULL);
 *
 *
 * MUTEX -> RESOLVER PROBLEMA DA CORRIDA:
 * Mutual Exclusion -> usado quando mais de uma thread pode acessar uma variavel global compartilhada -> pode dar problema!
 * pthread_mutex_t mutex_name; -> cria um mutex
 * pthread_mutex_init (&mutex_name, NULL) -> inicia o mutex
 * pthread_mutex_destroy (&mutex_name) -> como se fosse free(mutex_name)
 * pthread_mutex_lock (&mutex_name) -> pegar acesso a regiao critica.
 * pthread_mutex_unlock (&mutex_name) -> liberar acesso a regiao critica.
 *
 *
 * SEMAFOROS -> Resolver Ordem e Nao Problema da Corrida!
 * #include <semaphore.h>
 * semt_t semaphore_name;
 * sem_init (&semaphore_name, Shares (usar 1), valor inicial (usar 1))
 * sempre q ver um wait ele ve se tem um numero > 0 no semaforo, se tiver ele passa e decrementa esse valor, caso nao tenha ele espera alguem fazer post, que incrementa esse valor.
 * sem_destroy (&semaphore_name) -> destroi o semaforo qnd o programa acaba
 * sem_post (&semaphore_name)
 * sem_wait (&semaphore_name)
 *
 *
 * BARREIRAS:
 * usadas quando queremos que todas threads cheguem num mesmo ponto e ai sejam disparadas.
 * pthread_barrier_t  barrier_name;
 * pthread_barrier_init (&barrier_name, NULL (atributos padroes), const (numero de threads q devem chegar na barreira para ela ser liberada).
 * pthread_barrier_wait(&barrier_name);
 * pthread_barrier_destroy(&barrier_name) -> destroi barreira no fim do programa
 *
 *
 * VARIAVEIS DE CONDIÇÃO:
 * usadas para suspender threads ate que um evento ocorra
 * pthread_cond_t cond_name;
 * pthread_cond_wait (&cond_name, &mutex_name) -> ele libera o mutex no qual esta e espera um sinal disparar o cond_name
 * pthread_cond_broadcast (&cond_name) -> libera todas threads paradas em cond_wait
 *
 *
 * THREAD-SAFETY:
 * Uma thread é dita safe se seu bloco pode ser executado simultaneamente por varias threads sem causar problemas.
 */


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int thread_count;

void * Hello (void * rank) {
    long my_rank = (long) rank;

    printf("Hello from the thread %ld of %d threads\n", my_rank, thread_count);

    return NULL;
}

int main (int argc, char* argv[]) {
    long thread;
    pthread_t *thread_handle;

    thread_count = strtol(argv[1], NULL, 10);

    thread_handle = malloc (thread_count * sizeof(pthread_t));

    for (thread = 0; thread < thread_count; thread++) {
        pthread_create (&thread_handle[thread], NULL, Hello, (void *) thread);
    }

    printf ("Hello from the main Thread\n");

    for (thread = 0; thread < thread_count; thread++) {
        pthread_join (thread_handle[thread], NULL);
    }

    printf ("end\n");

    free (thread_handle);

    return 0;
}
