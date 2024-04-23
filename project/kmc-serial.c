/**
 * to compile and run this file, run the following command:
 *    gcc -lm -o kmc-serial kmc-serial.c && ./kmc-serial
 * 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// grid size constant
#define GRID_SIZE 128
// tolerance constant
#define TOLERANCE 1e-4

double calculate_euclidean_distance(double* data_point, double* k_cluster, size_t n_dimensions) {
    double sum = 0;
    for (size_t i = 0; i < n_dimensions; i++) {
        sum += (data_point[i] - k_cluster[i])*(data_point[i] - k_cluster[i]);
    }
    return sum;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <grid_size> <n_dimensions> <num_data_points> <num_k_clusters>\n", argv[0]);
        return 1;
    }

    // parse the user input
    int grid_size = atoi(argv[1]);
    int n_dimensions = atoi(argv[2]);
    int num_data_points = atoi(argv[3]);
    int num_k_clusters = atoi(argv[4]);

    // declare data_points and cluster_id arrays
    double data_points[num_data_points][n_dimensions];
    int cluster_id[num_data_points];

    // randomly initialize data_points between -GRID_SIZE and GRID_SIZE
    srand(0);
    for (size_t i = 0; i < num_data_points; i++) {
        for (size_t j = 0; j < n_dimensions; j++) {
            data_points[i][j] = (double)rand() / RAND_MAX * 2 * grid_size - grid_size;
        }
        cluster_id[i] = -1;
    }

    // 2. declare num_k_clusters and k_clusters
    double k_clusters[num_k_clusters][n_dimensions];

    // 3. randomly initialize k_clusters from data_points
    for (size_t i = 0; i < num_k_clusters; i++) {
        size_t index = rand() % num_data_points;
        for (size_t j = 0; j < n_dimensions; j++) {
            k_clusters[i][j] = data_points[index][j];
        }
    }

    // declare conditions for convergence
    size_t max_iterations = 256;
    size_t is_k_clusters_changed;
    size_t is_k_cluster_points_changed;

    // 6. repeat steps 4 and 5 until convergence
    for (size_t iteration = 0; iteration < max_iterations; iteration++) {
        is_k_clusters_changed = 0;
        is_k_cluster_points_changed = 0;
        
        // 4. assign each data point to the nearest centroid
        for (size_t i = 0; i < num_data_points; i++) {
            double min_distance;
            if (cluster_id[i] == -1) {
                min_distance = INFINITY;
            } else {
                min_distance = calculate_euclidean_distance(data_points[i], k_clusters[cluster_id[i]], n_dimensions);
            }
            for (size_t k = 0; k < num_k_clusters; k++) {
                double distance = calculate_euclidean_distance(data_points[i], k_clusters[k], n_dimensions);
                if (distance < min_distance) {
                    min_distance = distance;
                    cluster_id[i] = k;
                    is_k_clusters_changed = 1;
                }
            }
        }

        // check for convergence
        if (!is_k_clusters_changed) {
            //printf("is_k_clusters_changed: %d\n", is_k_clusters_changed);
            //printf("Converged at iteration %d\n", iteration);
            break;
        }

        // 5. recalculate the centroids of the clusters
        for (int i = 0; i < num_k_clusters; i++) {
            double* sums = (double*)calloc(n_dimensions, sizeof(double)); // initialize sums array with zeros
            size_t num_points = 0;
            for (int k = 0; k < num_data_points; k++) {
                if (cluster_id[k] == i) {
                    for (int j = 0; j < n_dimensions; j++) {
                        sums[j] += data_points[k][j];
                    }
                    num_points++;
                }
            }
            if (num_points > 0) {
                for (int j = 0; j < n_dimensions; j++) {
                    double new_value = sums[j] / num_points;
                    if (fabs(new_value - k_clusters[i][j]) > TOLERANCE) {
                        k_clusters[i][j] = new_value;
                        is_k_cluster_points_changed = 1;
                    }
                }
            }
        }

        // check for convergence
        if (!is_k_cluster_points_changed) {
            //printf("is_k_cluster_points_changed: %d\n", is_k_cluster_points_changed);
            //printf("Converged at iteration %d\n", iteration);
            break;
        }
    }

    // print the data_points and their x, y, and cluster_id to output.txt separated by a comma
    // filename is output-date-time.txt in the output folder
    char filename[256];
    sprintf(filename, "output/output-%ld.txt", time(NULL));
    FILE* output_file = fopen(filename, "w");
    //FILE* output_file = fopen("output.txt", "w");
    for (size_t i = 0; i < num_data_points; i++) {
        for (size_t j = 0; j < n_dimensions; j++) {
            fprintf(output_file, "%lf", data_points[i][j]);
            if (j < n_dimensions - 1) {
                fprintf(output_file, ",");
            }
        }
        fprintf(output_file, ",%d\n", cluster_id[i]);
    }
    fclose(output_file);

    return 0;
}