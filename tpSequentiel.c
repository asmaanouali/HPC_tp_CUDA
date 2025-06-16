#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_ITERS 100
#define TOL 1e-4

typedef struct {
    double x, y;
} Point;

double euclidean_distance(Point a, Point b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

void initialize_centroids(Point *centroids, int k) {
    printf("\nInitializing centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Enter coordinates for centroid %d (x y): ", i + 1);
        scanf("%lf %lf", &centroids[i].x, &centroids[i].y);
    }
}

void assign_clusters(Point *data, Point *centroids, int *clusters, int n, int k) {
    for (int i = 0; i < n; i++) {
        double min_dist = euclidean_distance(data[i], centroids[0]);
        int cluster = 0;

        for (int j = 1; j < k; j++) {
            double dist = euclidean_distance(data[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }
        clusters[i] = cluster;
    }
}

void update_centroids(Point *data, Point *centroids, int *clusters, int n, int k) {
    Point sums[k];
    int counts[k];

    for (int i = 0; i < k; i++) {
        sums[i].x = 0.0;
        sums[i].y = 0.0;
        counts[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        int cluster = clusters[i];
        sums[cluster].x += data[i].x;
        sums[cluster].y += data[i].y;
        counts[cluster]++;
    }

    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            centroids[i].x = sums[i].x / counts[i];
            centroids[i].y = sums[i].y / counts[i];
        }
    }
}

void k_means(Point *data, Point *centroids, int *clusters, int n, int k) {
    for (int iter = 0; iter < MAX_ITERS; iter++) {
        assign_clusters(data, centroids, clusters, n, k);

        Point old_centroids[k];
        for (int i = 0; i < k; i++) {
            old_centroids[i] = centroids[i];
        }

        update_centroids(data, centroids, clusters, n, k);

        int converged = 1;
        for (int i = 0; i < k; i++) {
            if (euclidean_distance(old_centroids[i], centroids[i]) > TOL) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            printf("\nConverged after %d iterations.\n", iter + 1);
            return;
        }
    }
    printf("\nReached maximum iterations without full convergence.\n");
}

int read_data_from_file(const char *filename, Point **data) {
    //pour lire les données à partir d'un fichier
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Erreur d'ouverture du fichier %s.\n", filename);
        return -1;
    }

    int n = 0;
    while (!feof(file)) {
        double x, y;
        if (fscanf(file, "%lf %lf", &x, &y) == 2) {
            n++;
        }
    }

    rewind(file);

    *data = malloc(n * sizeof(Point));
    if (*data == NULL) {
        printf("Erreur d'allocation mémoire.\n");
        fclose(file);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        fscanf(file, "%lf %lf", &(*data)[i].x, &(*data)[i].y);
    }

    fclose(file);
    return n;
}

int main() {
    int k;
    char filename[100];
    Point *data;
     int n = read_data_from_file("data2.txt", &data);
    
    if (n == -1) {
        return 1; 
    }

    printf("\nEnter the number of clusters (k): ");
    scanf("%d", &k);

    Point *centroids = malloc(k * sizeof(Point));
    int *clusters = malloc(n * sizeof(int));

    initialize_centroids(centroids, k);

    // Enregistrer le temps de début
    clock_t start_time = clock();

    k_means(data, centroids, clusters, n, k);

    // Enregistrer le temps de fin
    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC; 

    printf("\nFinal centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: (%.2f, %.2f)\n", i + 1, centroids[i].x, centroids[i].y);
    }

    printf("\nExecution time: %.6f seconds\n", execution_time); 

    free(data);
    free(centroids);
    free(clusters);

    return 0;
}
