//Paralelisation de l'algorithme k-means avec cuda


#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX_ITERS 100
#define NB_POINT 5000000
#define threads_per_block 512
#define TOL 1e-4

typedef struct {
    float x, y;
} Point;

// Calcul de la distance euclidienne entre deux points
__device__ float euclidean_distance(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Kernel pour assigner chaque point à un cluster
__global__ void assign_clusters(Point *data, Point *centroids, int *clusters, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index global du thread
    if (idx < n) { // Vérifier si le thread est dans les limites des données
        float min_dist = euclidean_distance(data[idx], centroids[0]);
        int cluster = 0;

        // Calculer la distance pour chaque centroid
        for (int j = 1; j < k; j++) {
            float dist = euclidean_distance(data[idx], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                cluster = j;
            }
        }
        clusters[idx] = cluster; // Assigner le cluster
    }
}

// Kernel pour mettre à jour les centroids
__global__ void update_centroids(Point *data, int *clusters, int n, int k, Point *sums, int *counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int cluster = clusters[idx];
        atomicAdd(&(sums[cluster].x), data[idx].x);
        atomicAdd(&(sums[cluster].y), data[idx].y);
        atomicAdd(&(counts[cluster]), 1);
    }
}

int read_data_from_file(const char *filename, Point **data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return -1;
    }

    *data = (Point *)malloc(NB_POINT * sizeof(Point));

    for (int i = 0; i < NB_POINT; i++) {
        fscanf(file, "%f %f", &(*data)[i].x, &(*data)[i].y); 
    }

    fclose(file);
    return NB_POINT; 
}

int main() {
    int k;
    Point *data, *d_data, *centroids, *d_centroids, *sums, *d_sums;
    int *clusters, *d_clusters, *counts, *d_counts;
    char filename[256];

    // Demander le nom du fichier à l'utilisateur
    printf("Enter the name of the data file: ");
    scanf("%s", filename);

    // Lire les points depuis le fichier
    int n = read_data_from_file(filename, &data);
    if (n == -1) return 1;

    printf("\nEnter the number of clusters (k): ");
    scanf("%d", &k);

    centroids = (Point *)malloc(k * sizeof(Point));
    clusters = (int *)malloc(n * sizeof(int));
    sums = (Point *)malloc(k * sizeof(Point));
    counts = (int *)malloc(k * sizeof(int));

    // Initialisation manuelle des centroids
    printf("\nEnter the centroids (x, y) for each cluster:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: ", i + 1);
        scanf("%f %f", &centroids[i].x, &centroids[i].y);
    }

    // Allocation mémoire sur le GPU
    cudaMalloc(&d_data, n * sizeof(Point));
    cudaMalloc(&d_centroids, k * sizeof(Point));
    cudaMalloc(&d_clusters, n * sizeof(int));
    cudaMalloc(&d_sums, k * sizeof(Point));
    cudaMalloc(&d_counts, k * sizeof(int));

    // Copier les données sur le GPU
    cudaMemcpy(d_data, data, n * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, k * sizeof(Point), cudaMemcpyHostToDevice);

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int iter;
    for (iter = 0; iter < MAX_ITERS; iter++) {
        // Réinitialiser sums et counts
        cudaMemset(d_sums, 0, k * sizeof(Point));
        cudaMemset(d_counts, 0, k * sizeof(int));

        // Lancer le kernel assign_clusters
        assign_clusters<<<num_blocks, threads_per_block>>>(d_data, d_centroids, d_clusters, n, k);

        // Lancer le kernel update_centroids
        update_centroids<<<num_blocks, threads_per_block>>>(d_data, d_clusters, n, k, d_sums, d_counts);


        // Copier les résultats partiels sur le CPU
        cudaMemcpy(sums, d_sums, k * sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(counts, d_counts, k * sizeof(int), cudaMemcpyDeviceToHost);

        // Calculer les nouveaux centroids sur le CPU
        float max_change = 0.0;
        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                float new_x = sums[i].x / counts[i];
                float new_y = sums[i].y / counts[i];
                max_change = fmax(max_change, fabs(new_x - centroids[i].x));
                max_change = fmax(max_change, fabs(new_y - centroids[i].y));
                centroids[i].x = new_x;
                centroids[i].y = new_y;
            }
        }

        // Copier les nouveaux centroids sur le GPU
        cudaMemcpy(d_centroids, centroids, k * sizeof(Point), cudaMemcpyHostToDevice);

        // Vérifier la convergence
        if (max_change < TOL) {
            printf("Convergence reached at iteration %d\n", iter + 1);
            break;
        }
    }

    // Mesurer le temps d'exécution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("\nTemps d'execution: %.2f ms\n", milliseconds);
    printf("Nombre d'iterations: %d\n", iter + 1);

    cudaMemcpy(clusters, d_clusters, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Libérer la mémoire
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_sums);
    cudaFree(d_counts);

    free(data);
    free(centroids);
    free(clusters);
    free(sums);
    free(counts);

    return 0;
}
