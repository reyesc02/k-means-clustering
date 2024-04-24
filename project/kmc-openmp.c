/**
 * Parallel implementation of the K-Means Clustering algorithm with OpenMP
 * to compile and run this file, run the following command:
 *    gcc-13 $(mpicc -showme:compile) $(mpicc -showme:link) -Wall -O3 -march=native -fopenmp kmc-openmp.c -o kmc-openmp -lm
 *     mpirun -np 4 ./kmc-openmp 128 2 1000 4
 *
 * on linux:
 *    gcc -o kmc-openmp kmc-openmp.c -fopenmp -lm && ./kmc-openmp 128 2 1000 4
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// tolerance constant
#define TOLERANCE 1e-4

double calculate_euclidean_distance(double *data_point, double *k_cluster, size_t n_dimensions)
{
    double sum = 0;
    for (size_t i = 0; i < n_dimensions; i++)
    {
        sum += (data_point[i] - k_cluster[i]) * (data_point[i] - k_cluster[i]);
    }
    return sum;
}

int main(int argn, char *argv[])
{
    // declare variables
    size_t num_data_points = 10;
    size_t n_dimensions = 2;
    size_t num_k_clusters = 4;
    double grid_size = 32;

    // check for command line arguments
    switch (argn) {
        case 1:
            break;
        case 5:
            num_data_points = atoi(argv[1]);
            n_dimensions = atoi(argv[2]);
            num_k_clusters = atoi(argv[3]);
            grid_size = atof(argv[4]);
            break;
        default:
            printf("Usage: %s <num_data_points> <n_dimensions> <num_k_clusters> <grid_size>\n", argv[0]);
            return 1;
    }

    // 1. allocate memory for data_points, cluster_id, and k_clusters
    double** data_points = (double**)malloc(num_data_points * sizeof(double*));
    for (size_t i = 0; i < num_data_points; i++) {
        data_points[i] = (double*)malloc(n_dimensions * sizeof(double));
    }
    int* cluster_id = (int*)malloc(num_data_points * sizeof(int));

    double** k_clusters = (double**)malloc(num_k_clusters * sizeof(double*));
    for (size_t i = 0; i < num_k_clusters; i++) {
        k_clusters[i] = (double*)malloc(n_dimensions * sizeof(double));
    }

    // randomly initialize data_points between -GRID_SIZE and GRID_SIZE
    srand(0);
    for (size_t i = 0; i < num_data_points; i++)
    {
        for (size_t j = 0; j < n_dimensions; j++)
        {
            data_points[i][j] = (double)rand() / RAND_MAX * 2 * grid_size - grid_size;
        }
        cluster_id[i] = -1;
    }

    // 3. randomly initialize k_clusters from data_points
    for (size_t i = 0; i < num_k_clusters; i++)
    {
        size_t rand_index = rand() % num_data_points;
        for (size_t j = 0; j < n_dimensions; j++)
        {
            k_clusters[i][j] = data_points[rand_index][j];
        }
    }

    // declare conditions for convergence
    size_t max_iterations = 300;
    size_t is_k_clusters_changed;
    size_t is_k_cluster_points_changed;

    // 6. repeat steps 4 and 5 until convergence
    for (size_t iteration = 0; iteration < max_iterations; iteration++)
    {
        is_k_clusters_changed = 0;
        is_k_cluster_points_changed = 0;

        // 4. assign each data point to the nearest centroid
        #pragma omp parallel for shared(data_points, k_clusters, cluster_id, is_k_clusters_changed, num_data_points, n_dimensions, num_k_clusters) default(none)
        for (size_t i = 0; i < num_data_points; i++)
        {
            double min_distance;
            size_t nearest_cluster = cluster_id[i]; // Initialize to current cluster
            if (nearest_cluster == -1)
            {
                min_distance = INFINITY;
            }
            else
            {
                min_distance = calculate_euclidean_distance(data_points[i], k_clusters[nearest_cluster], n_dimensions);
            }
            for (size_t k = 0; k < num_k_clusters; k++)
            {
                double distance = calculate_euclidean_distance(data_points[i], k_clusters[k], n_dimensions);
                if (distance < min_distance)
                {
                    min_distance = distance;
                    nearest_cluster = k; // Update nearest cluster
                    is_k_clusters_changed = 1;
                }
            }
            cluster_id[i] = nearest_cluster; // Update cluster assignment
        }

        // check for convergence
        if (!is_k_clusters_changed)
        {
            // printf("is_k_clusters_changed: %d\n", is_k_clusters_changed);
            // printf("Converged at iteration %d\n", iteration);
            break;
        }

        // 5. recalculate the centroids of the clusters
        #pragma omp parallel for shared(data_points, k_clusters, cluster_id, is_k_cluster_points_changed, num_k_clusters, num_data_points, n_dimensions) default(none)
        for (int i = 0; i < num_k_clusters; i++)
        {
            for (int j = 0; j < n_dimensions; j++)
            {
                double sums = 0; 
                size_t num_points = 0;
                for (int k = 0; k < num_data_points; k++)
                {
                    if (cluster_id[k] == i)
                    {
                        sums += data_points[k][j];
                        num_points++;
                    }
                }
                if (num_points > 0)
                {
                    double new_value = sums / num_points;
                    if (fabs(new_value - k_clusters[i][j]) > TOLERANCE)
                    {
                        k_clusters[i][j] = new_value;
                        is_k_cluster_points_changed = 1;
                    }
                }
            }
        }

        // check for convergence
        if (!is_k_cluster_points_changed)
        {
            // printf("is_k_cluster_points_changed: %d\n", is_k_cluster_points_changed);
            // printf("Converged at iteration %d\n", iteration);
            break;
        }
    }

    // print the data_points and their x, y, and cluster_id to output.txt separated by a comma
    // filename is output-date-time.txt in the output folder
    char filename[256];
    sprintf(filename, "output/output-%ld.txt", time(NULL));
    FILE *output_file = fopen(filename, "w");
    // FILE* output_file = fopen("output.txt", "w");
    for (size_t i = 0; i < num_data_points; i++)
    {
        for (size_t j = 0; j < n_dimensions; j++)
        {
            fprintf(output_file, "%lf", data_points[i][j]);
            if (j < n_dimensions - 1)
            {
                fprintf(output_file, ",");
            }
        }
        fprintf(output_file, ",%d\n", cluster_id[i]);
    }
    fclose(output_file);

    return 0;
}