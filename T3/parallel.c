#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>

int thread_count, numero_el_vet_dados, numero_de_bins, *qtd_dados_por_bin, counter = 0;
double *dados, *bins, interval_bin;
long unsigned int tempo = -1;
pthread_t *thread_handle;
pthread_mutex_t acumular_bin, maior_tempo;
sem_t counter_sem;
sem_t barrier_sem;

/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}

	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}

	return max;
}

void * histogram (void * rank) {
    long my_rank = (long) rank;
    int my_init = my_rank * numero_el_vet_dados/thread_count;
    int my_end = (my_rank + 1) * numero_el_vet_dados/thread_count;
    int local_j, local_i, *my_bin, count;
	double min_t, max_t;
    struct timeval start, end;
    long unsigned int duracao;

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

	gettimeofday(&start, NULL);

	my_bin = malloc (numero_de_bins * sizeof(int));

	for(local_j = 0; local_j < numero_de_bins; local_j++) {
		count = 0;
		min_t = bins[0] + local_j * interval_bin;
		max_t = bins[0] + (local_j + 1) * interval_bin;
		for(local_i = my_init; local_i < my_end; local_i++) {
			if( (dados[local_i] <= max_t && dados[local_i] > min_t) || (local_j == 0 && dados[local_i] <= min_t) ) {
				//printf ("dado: %lf, bin_esq: %lf, bin_dir: %lf, local_j = %d\n", val[i], min_t, max_t, j);
				count++;
			}
		}
		my_bin[local_j] = count;
	}

    pthread_mutex_lock (&acumular_bin);
    for (local_i = 0; local_i < numero_de_bins; local_i++) {
        qtd_dados_por_bin[local_i] += my_bin[local_i];
    }
    pthread_mutex_unlock(&acumular_bin);

    gettimeofday(&end, NULL);
	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));
    pthread_mutex_lock (&maior_tempo);
    if (duracao > tempo)
        tempo = duracao;
    pthread_mutex_unlock(&maior_tempo);
    return NULL;
}


int main () {

	long i;

    scanf("%d", &thread_count);
    scanf("%d", &numero_el_vet_dados);
    scanf("%d", &numero_de_bins);

    dados = malloc (numero_el_vet_dados * sizeof(double));
    bins = malloc ((numero_de_bins + 1) * sizeof(double));
    qtd_dados_por_bin = malloc (numero_de_bins * sizeof(int));
    thread_handle = malloc (thread_count * sizeof(pthread_t));

    for (i = 0; i < numero_el_vet_dados; i++) {
        scanf("%lf", &dados[i]);
    }

    bins[0] = floor(min_val(dados , numero_el_vet_dados));
    bins[numero_de_bins] = ceil (max_val(dados , numero_el_vet_dados));
    interval_bin = (bins[numero_de_bins] - bins[0]) / numero_de_bins;
    for (i = 1; i < numero_de_bins; i++) {
        bins[i] = bins[i-1] + interval_bin;
    }

    for (i = 0; i < numero_de_bins; i++)
        qtd_dados_por_bin[i] = 0;

    pthread_mutex_init(&acumular_bin, NULL);
    pthread_mutex_init(&maior_tempo, NULL);
    sem_init (&counter_sem, 1, 1);
    sem_init (&barrier_sem, 1, 0);

    for (i = 0; i < thread_count; i++) {
        pthread_create (&thread_handle[i], NULL, histogram, (void *) i);
    }

    for (i = 0; i < thread_count; i++) {
        pthread_join (thread_handle[i], NULL);
    }

    for (i = 0; i <= numero_de_bins; i++) {
        printf("%.2lf", bins[i]);
        if (i != numero_de_bins)
            printf(" ");
    }
    printf("\n");

    for (i = 0; i < numero_de_bins; i++) {
        printf("%d", qtd_dados_por_bin[i]);
        if (i != numero_de_bins - 1)
            printf(" ");
    }
    printf("\n");
    printf("%lu\n", tempo);

    free (dados);
    free (bins);
    free (qtd_dados_por_bin);
    free (thread_handle);
    pthread_mutex_destroy (&acumular_bin);
    pthread_mutex_destroy (&maior_tempo);
    sem_destroy (&counter_sem);
    sem_destroy (&barrier_sem);

    return 0;
}
