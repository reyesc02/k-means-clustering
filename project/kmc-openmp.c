/**
 * @file kmc-openmp.c
 * @brief K-means clustering using OpenMP
 * @details This program implements K-means clustering using OpenMP.
 * The program reads data points from a CSV file and performs K-means clustering with threads.
 * The program outputs the cluster id of each data point to a CSV file.
 * The program also outputs the time taken for the clustering.
 * The program takes two command line arguments: the input file and the number of clusters.
 * The input file should be a CSV file with each row representing a data point.
 * The program uses the Euclidean distance to calculate the distance between data points and centroids.
 * 
 * @date 2024-04-30
 * 
 * @authors
 * Carl R.
 * Brian D.
 * Anna H.
 * 
 * To compile this file run the following command:
 * gcc-13 $(mpicc -showme:compile) $(mpicc -showme:link) -Wall -O3 -march=native -fopenmp kmc-openmp.c matrix.c -o kmc-openmp -lm
 * 
 * To run the compiled file run the following command:
 * mpirun -np 4 ./kmc-openmp <input_file> <k>
 * mpirun -np 4 ./kmc-openmp <n> <d> <k> <grid_size> <seed>
*/

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "matrix.h"

#define WRITE_OUTPUT 0

// tolerance constant
#define TOLERANCE 1e-4

// global variables
unsigned long long num_data_points;
unsigned long long n_dimensions;
unsigned long long num_k_clusters;
double grid_size;

// global pointers
Matrix* data_points;
int* cluster_id;
Matrix* k_clusters;

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
double calculate_euclidean_distance(double* data_point, double* k_cluster, size_t n_dimensions) {
    double sum = 0;
    for (size_t i = 0; i < n_dimensions; i++) {
        double difference = data_point[i] - k_cluster[i];
        double square = difference * difference;
        sum += square;
    }
    return sum;
}

/**
 * Parse Command Line Arguments function
 * This function parses the command line arguments and initializes
 * the global variables.
 * @param argn: the number of arguments
 * @param argv: the array of arguments
 * @return true if the command line arguments are valid, false otherwise
 */
bool parse_command_line_arguments(int argn, char* argv[]) {
    if (argn == 1) {
        num_data_points = 32767;
        n_dimensions = 2;
        num_k_clusters = 8;
        grid_size = 1024;
        srand(0);
        data_points = matrix_random_grid(num_data_points, n_dimensions, grid_size);
    } else if (argn == 3) {
        data_points = matrix_from_csv_path(argv[1]);
        if (data_points == NULL) { perror("error reading input"); return false; }
        n_dimensions = data_points->cols;
        num_data_points = data_points->rows;
        num_k_clusters = atoi(argv[2]);
    } else if (argn == 5 || argn == 6) {
        num_data_points = atoi(argv[1]);
        n_dimensions = atoi(argv[2]);
        num_k_clusters = atoi(argv[3]);
        grid_size = atof(argv[4]);
        srand((argn == 6) ? atoi(argv[5]) : time(NULL));
        data_points = matrix_random_grid(num_data_points, n_dimensions, grid_size);
    } else {
        printf("Usage: %s <input_file> <num_k_clusters>\n", argv[0]);
        printf("Usage: %s <num_data_points> <n_dimensions> <num_k_clusters> <grid_size> [seed]\n", argv[0]);
        return false;
    }
    
    // initialize cluster_id to -1
    cluster_id = (int*)malloc(num_data_points * sizeof(int));
    memset(cluster_id, -1, num_data_points * sizeof(int));

    // randomly initialize k_clusters from data_points
    k_clusters = matrix_create_raw(num_k_clusters, n_dimensions);
    for (size_t i = 0; i < num_k_clusters; i++) {
        size_t rand_index = rand() % num_data_points;
        size_t rand_index_n_dimensions = rand_index * n_dimensions;
        size_t i_n_dimensions = i * n_dimensions;
        for (size_t j = 0; j < n_dimensions; j++) {
            k_clusters->data[i_n_dimensions + j] = data_points->data[rand_index_n_dimensions + j];
        }
    }

    return true;
}

/**
 * Assign Data Points to Nearest Cluster function
 * This function assigns each data point to the nearest cluster.
 * @return true if the k_clusters are changed, false otherwise
 */
bool assign_data_points_to_nearest_cluster() {
    bool is_k_clusters_changed = false;
    #pragma omp parallel for shared(data_points, k_clusters, cluster_id, num_data_points, n_dimensions, num_k_clusters) reduction(+:is_k_clusters_changed) schedule(static) num_threads(4) default(none)
    for (size_t i = 0; i < num_data_points; i++) {
        double min_distance = INFINITY;
        int current_cluster_id = cluster_id[i];
        size_t current_cluster_id_n_dimensions = current_cluster_id * n_dimensions;
        int i_n_dimensions = i * n_dimensions;
        if (current_cluster_id != -1) {
            min_distance = calculate_euclidean_distance(data_points->data + i_n_dimensions, k_clusters->data + current_cluster_id_n_dimensions, n_dimensions);
        }
        for (size_t k = 0; k < num_k_clusters; k++) {
            double distance = calculate_euclidean_distance(data_points->data + i_n_dimensions, k_clusters->data + k * n_dimensions, n_dimensions);
            if (distance < min_distance) {
                min_distance = distance;
                cluster_id[i] = k;
                is_k_clusters_changed = true;
            }
        }
    }
    return is_k_clusters_changed;
}

/**
 * Recalculate Centroids function
 * This function recalculates the centroids of the clusters.
 * @return true if the k_cluster points are changed, false otherwise
 */
bool recalculate_centroids() {
    bool is_k_cluster_points_changed = false;
    #pragma omp parallel for shared(data_points, k_clusters, cluster_id, num_data_points, n_dimensions, num_k_clusters) reduction(+:is_k_cluster_points_changed) schedule(static) num_threads(4) default(none)
    for (int i = 0; i < num_k_clusters; i++) {
        // loop fission for better performance
        double* sum = (double*)calloc(n_dimensions, sizeof(double));
        int count = 0;
        for (int k = 0; k < num_data_points; k++) {
            if (cluster_id[k] == i) {
                size_t k_n_dimensions = k * n_dimensions;
                for (int j = 0; j < n_dimensions; j++) {
                    sum[j] += data_points->data[k_n_dimensions + j];
                }
                count++;
            }
        }
        size_t i_n_dimensions = i * n_dimensions;
        for (int j = 0; j < n_dimensions; j++) {
            if (count > 0) {
                double new_centroid = sum[j] / count;
                if (fabs(k_clusters->data[i_n_dimensions + j] - new_centroid) > TOLERANCE) {
                    k_clusters->data[i_n_dimensions + j] = new_centroid;
                    is_k_cluster_points_changed = true;
                }
            }
        }
        free(sum);
    }
    return is_k_cluster_points_changed;
}

/**
 * Run K Means function
 * This function runs the K Means algorithm.
 * @return the iteration reached when the algorithm converged
 */
int run_kmeans(size_t _max_iterations, int* reason_converged) {
    size_t max_iterations = _max_iterations;
    size_t iteration_reached = 0;

    for (size_t iteration = 0; iteration < max_iterations; iteration++) {

        // assign data points to nearest cluster
        bool is_k_clusters_changed = assign_data_points_to_nearest_cluster();

        // check for convergence
        if (is_k_clusters_changed == false) {
            iteration_reached = iteration;
            *reason_converged = 1;
            break;
        }

        // recalculate centroids
        bool is_k_cluster_points_changed = recalculate_centroids();

        // check for convergence
        if (is_k_cluster_points_changed == false) {
            iteration_reached = iteration;
            *reason_converged = 2;
            break;
        }
    }
    return iteration_reached;
}

/**
 * Print Results function
 * This function prints the results of the K Means algorithm.
 */
void print_results_to_file(int iteration_converged, int reason_converged, double time_taken) {
    // save timestamp to variable
    char timestamp[128];
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    sprintf(timestamp, "%d-%d-%d-%d-%d-%d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

    
    // output data_points and cluster_id to file
    char output_file[512];
    FILE* file;
    if (WRITE_OUTPUT) {
        sprintf(output_file, "output/openmp-output-%s-data.csv", timestamp);
        file = fopen(output_file, "w");
        if (file == NULL) { perror("error writing output"); return; }
        for (size_t i = 0; i < num_data_points; i++) {
            for (size_t j = 0; j < n_dimensions; j++) {
                fprintf(file, "%f,", data_points->data[i * n_dimensions + j]);
            }
            fprintf(file, "%d\n", cluster_id[i]);
        }
        fclose(file);
    }

    // output program info
    sprintf(output_file, "output/openmp-output-%s-info.txt", timestamp);
    file = fopen(output_file, "w");
    if (file == NULL) { perror("error writing output"); return; }
    fprintf(file, "Data Points: %llu\n", num_data_points);
    fprintf(file, "Dimensions: %llu\n", n_dimensions);
    fprintf(file, "K Clusters: %llu\n\n", num_k_clusters);
    fprintf(file, "Time Taken: %f seconds\n\n", time_taken);

    // print each cluster and its number of data points
    for (size_t i = 0; i < num_k_clusters; i++) {
        int count = 0;
        for (size_t j = 0; j < num_data_points; j++) {
            if (cluster_id[j] == i) {
                count++;
            }
        }
        fprintf(file, "Cluster %zu: %d\n", i, count);
    } fprintf(file, "\n");

    // print iteration converged and reason
    if (reason_converged == 1) {
        fprintf(file, "Converged after %d iterations because k_clusters did not change\n", iteration_converged);
    } else if (reason_converged == 2) {
        fprintf(file, "Converged after %d iterations because k_cluster points did not change\n", iteration_converged);
    } else {
        fprintf(file, "Did not converge after %d iterations\n", iteration_converged);
    }
    fclose(file);   
}

/**
 * Main function
 * This is the entry point of the program.
 * @param argn: the number of arguments
 * @param argv: the array of arguments
 * @return 0 if the program runs successfully, 1 otherwise
 */
int main(int argn, char* argv[]) {
    
    // parse command line arguments
    if(!parse_command_line_arguments(argn, argv)) { return 1; }

    // start timer
    clock_t start = clock();

    // run k means algorithm
    size_t max_iterations = 300;
    int reason_converged = 0;
    int iteration_converged = run_kmeans(max_iterations, &reason_converged);

    // end timer
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    // print the results
    print_results_to_file(iteration_converged, reason_converged, time_taken);

    // Free memory
    matrix_free(data_points);
    free(cluster_id);
    matrix_free(k_clusters);
    
    return 0;
}
