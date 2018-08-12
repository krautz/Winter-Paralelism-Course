#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>

int thread_count;
int counter = 0;
unsigned int numero_jogadas;
long unsigned int duracao = -1;
long long unsigned int in = 0;

pthread_t *thread_handle;
pthread_mutex_t acumular_in, maior_tempo;
sem_t counter_sem;
sem_t barrier_sem;

void * monte_carlo_pi (void * rank) {
    long my_rank = (long) rank;
    int my_init = my_rank * numero_jogadas/thread_count;
    int my_end = (my_rank + 1) * numero_jogadas/thread_count;
    int local_j, local_i, my_in = 0;
    unsigned int my_seed = time(0) + my_rank;
    double x, y, d;
    struct timeval my_start, my_finish;
    long unsigned int my_duracao;

    sem_wait(&counter_sem);
    if (counter == thread_count - 1) {
        counter = 0;
        sem_post (&counter_sem);
        for (local_j = 0; local_j < thread_count - 1; local_j++)
            sem_post (&barrier_sem);
    } else {
        counter++;
        sem_post(&counter_sem);
        sem_wait(&barrier_sem);
    }

    gettimeofday(&my_start, NULL);

     for (local_i = my_init; local_i < my_end; local_i++) {
        x = ((rand_r(&my_seed) % 1000000)/500000.0)-1;
		y = ((rand_r(&my_seed) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1)
            my_in += 1;
    }

    pthread_mutex_lock (&acumular_in);
    in += my_in;
    pthread_mutex_unlock (&acumular_in);

    gettimeofday(&my_finish, NULL);
    my_duracao = ((my_finish.tv_sec * 1000000 + my_finish.tv_usec) - (my_start.tv_sec * 1000000 + my_start.tv_usec));

    pthread_mutex_lock (&maior_tempo);
    if (duracao < my_duracao)
        duracao = my_duracao;
    pthread_mutex_unlock (&maior_tempo);

    return NULL;
}
 int main(void) {
	long i;
	double pi;
 	scanf("%d %u",&thread_count, &numero_jogadas);
     thread_handle = malloc (thread_count * sizeof(pthread_t));
     pthread_mutex_init(&acumular_in, NULL);
    pthread_mutex_init(&maior_tempo, NULL);
    sem_init (&counter_sem, 1, 1);
    sem_init (&barrier_sem, 1, 0);
     for (i = 0; i < thread_count; i++) {
	       pthread_create(&thread_handle[i], NULL, monte_carlo_pi, (void *) i);
    }
     for (i = 0; i < thread_count; i++) {
	       pthread_join(thread_handle[i], NULL);
    }
 	pi = 4*in/((double)numero_jogadas);
	printf("%lf\n%lu\n",pi,duracao);
     free (thread_handle);
    pthread_mutex_destroy (&acumular_in);
    pthread_mutex_destroy (&maior_tempo);
    sem_destroy (&counter_sem);
    sem_destroy (&barrier_sem);
 	return 0;
}
