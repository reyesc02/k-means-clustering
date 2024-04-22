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

typedef struct {
    double distance;
    size_t index;
}DistanceIndex;

int compare(const void* a, const void* b) {
    double distanceA = ((DistanceIndex*)a)->distance;
    double distanceB = ((DistanceIndex*)b)->distance;
    return (distanceA > distanceB) - (distanceA < distanceB);
}

DistanceIndex* calculate_distances(double* data_point, double** k_clusters, size_t num_k_clusters, size_t n_dimensions) {
    DistanceIndex* distances = (DistanceIndex*)malloc(num_k_clusters * sizeof(DistanceIndex));
    for (size_t i = 0; i < num_k_clusters; i++) {
        double distance = 0;
        for (size_t j = 0; j < n_dimensions; j++) {
            distance += (data_point[j] - k_clusters[i][j]) * (data_point[j] - k_clusters[i][j]);
        }
        distances[i].distance = sqrt(distance);
        distances[i].index = i;
    }
    qsort(distances, num_k_clusters, sizeof(DistanceIndex), compare);
    return distances;
}

int main(int argn, char* argv[]) {

    // declare n_dimensions
    size_t n_dimensions = 2;
    
    // 1. declare num_data_points and data_points
    size_t num_data_points = 16;
    double data_points[num_data_points][n_dimensions];
    int cluster_id[num_data_points];

    // randomly initialize data_points between -GRID_SIZE and GRID_SIZE
    srand(0);
    for (size_t i = 0; i < num_data_points; i++) {
        for (size_t j = 0; j < n_dimensions; j++) {
            data_points[i][j] = (double)rand() / RAND_MAX * 2 * GRID_SIZE - GRID_SIZE;
        }
        cluster_id[i] = -1;
    }

    // 2. declare num_k_clusters and k_clusters
    size_t num_k_clusters = 4;
    double k_clusters[num_k_clusters][n_dimensions];

    // 3. randomly initialize k_clusters from data_points
    for (size_t i = 0; i < num_k_clusters; i++) {
        for (size_t j = 0; j < n_dimensions; j++) {
            k_clusters[i][j] = data_points[i][j];
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
        // for (size_t i = 0; i < num_data_points; i++) {
        //     double min_distance;
        //     if (cluster_id[i] == -1) {
        //         min_distance = INFINITY;
        //     } else {
        //         min_distance = calculate_euclidean_distance(data_points[i].x, data_points[i].y, k_clusters[data_points[i].cluster_id].x, k_clusters[data_points[i].cluster_id].y);
        //     }
        //     for (size_t j = 0; j < num_k_clusters; j++) {
        //         double distance = calculate_euclidean_distance(data_points[i].x, data_points[i].y, k_clusters[j].x, k_clusters[j].y);
        //         if (distance < min_distance) {
        //             min_distance = distance;
        //             data_points[i].cluster_id = j;
        //             is_k_clusters_changed = 1;
        //         }
        //     }
        // }
        
        // 4. assign each data point to the nearest centroid
        for (size_t i = 0; i < num_data_points; i++) {
            for (size_t j = 0; j < n_dimensions; j++) {
                double min_distance;
                if (cluster_id[i] == -1) {
                    min_distance = INFINITY;
                } else {
                    min_distance = /* INSERT COMPLICATED LOOP FOR DISTANCE */0;
                }
                for (size_t k = 0; k < num_k_clusters; k++) {
                    double distance = /* INSERT COMPLICATED LOOP FOR DISTANCE */0;
                    if (distance < min_distance) {
                        min_distance = distance;
                        cluster_id[i] = k;
                        is_k_clusters_changed = 1;
                    }
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
        // for (int i = 0; i < num_k_clusters; i++) {
        //     double sum_x = 0;
        //     double sum_y = 0;
        //     size_t num_points = 0;
        //     for (int j = 0; j < num_data_points; j++) {
        //         if (data_points[j].cluster_id == i) {
        //             sum_x += data_points[j].x;
        //             sum_y += data_points[j].y;
        //             num_points++;
        //         }
        //     }
        //     if (num_points > 0) {
        //         double new_x = sum_x / num_points;
        //         double new_y = sum_y / num_points;
        //         if (fabs(new_x - k_clusters[i].x) > TOLERANCE || fabs(new_y - k_clusters[i].y) > TOLERANCE) {
        //             k_clusters[i].x = new_x;
        //             k_clusters[i].y = new_y;
        //             is_k_cluster_points_changed = 1;
        //         }
        //     }
        // }

        // 5. recalculate the centroids of the clusters
        for (int i = 0; i < num_k_clusters; i++) {
            for (int j = 0; j < n_dimensions; j++) {
                double sums[n_dimensions];
                size_t num_points = 0;
                for (int k = 0; k < num_data_points; k++) {
                    if (cluster_id[k] == i) {
                        sums[j] += data_points[k][j];
                        num_points++;
                    }
                }
                if (num_points > 0) {
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