/**
 * to compile and run this file, run the following command:
 *    gcc -lm -o kmc-serial kmc-serial.c matrix.c && ./kmc-serial data/housing.csv 8
 *    gcc -lm -o kmc-serial kmc-serial.c matrix.c && ./kmc-serial 32767 2 8 1024 0
 * 
*/


#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "matrix.h"

// tolerance constant
#define TOLERANCE 1e-4

// index of latitude and longitude in the data_points
#define LATITUDE_INDEX 0
#define LONGITUDE_INDEX 1

/**
 * Calculate Euclidean Distance function
 * This function calculates the Euclidean distance between two points.
 * This function does not square root the sum of the squares of the differences
 * because the square root is not necessary for comparing distances.
 * @param data_point: the pointer to the data_points row
 * @param k_cluster: the pointer to the k_clusters row
 * @param n_dimensions: the number of dimensions
 * @return the Euclidean distance between the two points
 */
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
    unsigned long long num_data_points = 10;
    unsigned long long n_dimensions = 2;
    unsigned long long num_k_clusters = 4;
    double grid_size = 32;

    Matrix* data_points;
    int* cluster_id;
    Matrix* k_clusters;

    // parse command line arguments
    if (argn == 1) {
        data_points = matrix_random_grid(num_data_points, n_dimensions, grid_size);
        k_clusters = matrix_create_raw(num_k_clusters, n_dimensions);
    } else if (argn == 3) {
        data_points = matrix_from_csv_path(argv[1]);
        if (data_points == NULL) { perror("error reading input"); return 1; }
        n_dimensions = data_points->cols;
        num_data_points = data_points->rows;
        num_k_clusters = atoi(argv[2]);
        k_clusters = matrix_create_raw(num_k_clusters, n_dimensions);
    } else if (argn == 5 || argn == 6) {
        num_data_points = atoi(argv[1]);
        n_dimensions = atoi(argv[2]);
        num_k_clusters = atoi(argv[3]);
        grid_size = atof(argv[4]);
        srand((argn == 6) ? atoi(argv[5]) : time(NULL));
        data_points = matrix_random_grid(num_data_points, n_dimensions, grid_size);
        k_clusters = matrix_create_raw(num_k_clusters, n_dimensions);
    } else {
        printf("Usage: %s <input_file> <num_k_clusters>\n", argv[0]);
        printf("Usage: %s <num_data_points> <n_dimensions> <num_k_clusters> <grid_size> [seed]\n", argv[0]);
        return 1;
    }

    // initialize cluster_id to -1
    cluster_id = (int*)malloc(num_data_points * sizeof(int));
    memset(cluster_id, -1, num_data_points * sizeof(int));
    
    // 3. randomly initialize k_clusters from data_points
    for (size_t i = 0; i < num_k_clusters; i++) {
        size_t rand_index = rand() % num_data_points;
        for (size_t j = 0; j < n_dimensions; j++)
        {
            k_clusters->data[i * n_dimensions + j] = data_points->data[rand_index * n_dimensions + j];
        }
    }

    // declare conditions for convergence
    size_t max_iterations = 300;
    size_t is_k_clusters_changed;
    size_t is_k_cluster_points_changed;

    size_t iterations_reached = 0;
    size_t reason = 0;

    // 6. repeat steps 4 and 5 until convergence
    for (size_t iteration = 0; iteration < max_iterations; iteration++)
    {
        is_k_clusters_changed = 0;
        is_k_cluster_points_changed = 0;

        // 4. assign each data point to the nearest centroid
        for (size_t i = 0; i < num_data_points; i++)
        {
            double min_distance = INFINITY;
            int current_cluster_id = cluster_id[i];
            int i_n_dimensions = i * n_dimensions;
            if (current_cluster_id != -1) {
                min_distance = calculate_euclidean_distance(data_points->data + i_n_dimensions, k_clusters->data + current_cluster_id * n_dimensions, n_dimensions);
            }
            for (size_t k = 0; k < num_k_clusters; k++) {
                double distance = calculate_euclidean_distance(data_points->data + i_n_dimensions, k_clusters->data + k * n_dimensions, n_dimensions);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_id[i] = k;
                    is_k_clusters_changed = 1;
                }
            }
        }

        // check for convergence
        if (!is_k_clusters_changed)
        {
            iterations_reached = iteration;
            reason = 1;
            // printf("is_k_clusters_changed: %d\n", is_k_clusters_changed);
            // printf("Converged at iteration %d\n", iteration);
            break;
        }

        // 5. recalculate the centroids of the clusters
        for (int i = 0; i < num_k_clusters; i++) {
            // Loop fission for better performance
            double* sum = (double*)calloc(n_dimensions, sizeof(double));
            int count = 0;
            // for (int j = 0; j < n_dimensions; j++) {
            //     sum[j] = 0;
            // }
            for (int k = 0; k < num_data_points; k++) {
                if (cluster_id[k] == i) {
                    for (int j = 0; j < n_dimensions; j++) {
                        sum[j] += data_points->data[k * n_dimensions + j];
                    }
                    count++;
                }
            }
            for (int j = 0; j < n_dimensions; j++) {
                if (count > 0) {
                    double new_centroid = sum[j] / count;
                    if (fabs(k_clusters->data[i * n_dimensions + j] - new_centroid) > TOLERANCE) {
                        k_clusters->data[i * n_dimensions + j] = new_centroid;
                        is_k_cluster_points_changed = 1;
                    }
                }
            }
            free(sum);
        }

        // check for convergence
        if (!is_k_cluster_points_changed)
        {
            iterations_reached = iteration;
            reason = 2;
            // printf("is_k_cluster_points_changed: %d\n", is_k_cluster_points_changed);
            // printf("Converged at iteration %d\n", iteration);
            break;
        }
    }

    // write output info to output-info-[time].txt    
    char filename[256];
    sprintf(filename, "output/output-info-%ld.txt", time(NULL));
    FILE *output_file = fopen(filename, "w");
    if (output_file == NULL)
    {
        printf("Error opening file for writing.\n");
        return 1;
    }

    // Print the number of clusters and points
    fprintf(output_file, "num_clusters\t%llu\nnum_points\t%llu\n\n", num_k_clusters, num_data_points);

    // Initialize an array to count the number of points in each cluster
    int* points_in_cluster = calloc(num_k_clusters, sizeof(int));
    for (size_t i = 0; i < num_data_points; i++) {
        points_in_cluster[cluster_id[i]]++; 
    }

    // Print the number of points in each cluster
    fprintf(output_file, "cluster\tnum_points\n");
    for (size_t i = 0; i < num_k_clusters; i++) {
        fprintf(output_file, "%zu\t%d\n", i, points_in_cluster[i]);
    }
    fprintf(output_file, "\n");

    // Print the reason for convergence
    char* reason_string;
    switch(reason) {
        case 0:
            reason_string = "max iterations reached";
            break;
        case 1:
            reason_string = "no change in cluster centroids";
            break;
        case 2:
            reason_string = "no change in cluster points";
            break;
        default:
            reason_string = "unknown";
            break;
    }
    
    fprintf(output_file, "converged after %zu iterations\nconverged due to %s\n", iterations_reached, reason_string);

    // Close the output file
    fclose(output_file);

    // Print data_points and cluster_id
    sprintf(filename, "output/output-data-points-%ld.csv", time(NULL));
    output_file = fopen(filename, "w");
    for (size_t i = 0; i < num_data_points; i++) {
        for (size_t j = 0; j < n_dimensions; j++) {
            fprintf(output_file, "%f,", data_points->data[i * n_dimensions + j]);
        }
        fprintf(output_file, "%d\n", cluster_id[i]);
    }
    fclose(output_file);

    // Free memory
    matrix_free(data_points);
    free(cluster_id);
    matrix_free(k_clusters);
    free(points_in_cluster);

    return 0;
}
