//Paralelisation de l'algorithme k-means avec p-threads

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define NB_POINTS 5000000
#define NB_CLUSTERS 2
#define NB_THREADS 2

int n, d = 2, k; 
double data[NB_POINTS][2];  // Tableau de points 
double centroids[NB_CLUSTERS][2];  
int clusters[NB_POINTS];
int points_per_thread[NB_THREADS];

pthread_mutex_t lock;

double euclidean_distance(double *point, double *centroid, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sqrt(sum);
}

typedef struct {
    int thread_id;
    int start;
    int end;
} thread_args_t;

void* assign_points(void* args) {
    thread_args_t* thread_data = (thread_args_t*)args;
    int start = thread_data->start;
    int end = thread_data->end;

    for (int i = start; i < end; i++) {
        double min_dist = DBL_MAX;
        int best_cluster = -1;
        for (int j = 0; j < k; j++) {
            double dist = euclidean_distance(data[i], centroids[j], d);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }
        clusters[i] = best_cluster;
    }
    return NULL;
}

void* update_centroids(void* args) {
    thread_args_t* thread_data = (thread_args_t*)args;
    int start = thread_data->start;
    int end = thread_data->end;

    double local_sums[NB_CLUSTERS][2] = {0};
    int local_counts[NB_CLUSTERS] = {0};

    for (int i = start; i < end; i++) {
        int cluster = clusters[i];
        for (int j = 0; j < d; j++) {
            local_sums[cluster][j] += data[i][j];
        }
        local_counts[cluster]++;
    }

    // Mise à jour globale des centroïdes sans verrouillage
    pthread_mutex_lock(&lock);
    for (int j = 0; j < k; j++) {
        for (int l = 0; l < d; l++) {
            centroids[j][l] += local_sums[j][l];
        }
        points_per_thread[j] += local_counts[j];
    }
    pthread_mutex_unlock(&lock);

    return NULL;
}

int check_convergence(double old_centroids[NB_CLUSTERS][2]) {
    double threshold = 1e-4;
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            if (fabs(centroids[i][j] - old_centroids[i][j]) > threshold) {
                return 0;
            }
        }
    }
    return 1;
}

void read_data_from_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Erreur d'ouverture du fichier");
        exit(EXIT_FAILURE);
    }

    // Lire les points et les stocker dans 'data'
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            fscanf(file, "%lf", &data[i][j]);
        }
    }

    fclose(file);
}

int main() {
    pthread_t threads[NB_THREADS];
    thread_args_t thread_args[NB_THREADS];
    pthread_mutex_init(&lock, NULL);

    // Saisie du nombre de points et de clusters
    printf("Entrez le nombre de points : \n");
    scanf("%d", &n);
     printf("Entrez le nombre de clusters : \n");
    scanf("%d", &k);

    char filename[256];
    printf("Entrez le nom du fichier contenant les données : \n");
    scanf("%s", filename);

    // Lecture des points à partir du fichier
    read_data_from_file(filename);

    // lecture des centroides initiaux
    printf("Entrez les coordonnées des %d centroids (chaque centroid a 2 dimensions) :\n", k);
    for (int i = 0; i < k; i++) {
        printf("Centroid %d : ", i + 1);
        for (int j = 0; j < d; j++) {
            scanf("%lf", &centroids[i][j]);
        }
    }

    int converged = 0;
    double old_centroids[NB_CLUSTERS][2];

    // Temps de départ
    clock_t start_time = clock();

    while (!converged) {
        // Sauvegarde des anciens centroides
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                old_centroids[i][j] = centroids[i][j];
            }
        }

        // Répartition du travail entre les threads
        int chunk_size = n / NB_THREADS;
        for (int t = 0; t < NB_THREADS; t++) {
            thread_args[t].thread_id = t;
            thread_args[t].start = t * chunk_size;
            thread_args[t].end = (t == NB_THREADS - 1) ? n : (t + 1) * chunk_size;
            pthread_create(&threads[t], NULL, assign_points, &thread_args[t]);
        }

        for (int t = 0; t < NB_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        // Réinitialisation des centroides et comptage des points par thread
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                centroids[i][j] = 0.0;
            }
            points_per_thread[i] = 0;
        }

        // Mise à jour des centroides
        for (int t = 0; t < NB_THREADS; t++) {
            pthread_create(&threads[t], NULL, update_centroids, &thread_args[t]);
        }

        for (int t = 0; t < NB_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }

        // Calcul des nouveaux centroides
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                centroids[i][j] /= points_per_thread[i];
            }
        }

        // Vérification de la convergence
        converged = check_convergence(old_centroids);
    }

    // Temps de fin
    clock_t end_time = clock();

    // Calcul et affichage du temps d'exécution
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Temps d'exécution : %.6f secondes\n", execution_time);

    pthread_mutex_destroy(&lock);
    return 0;
}
