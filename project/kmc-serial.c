/**
 * to compile and run this file, run the following command:
 *    gcc -lm -o kmc-serial kmc-serial.c && ./kmc-serial
 * 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// constant for the grid size
// the grid size is the size of the 2D grid where the data points are located
#define GRID_SIZE 128

struct Point {
    double x, y;
    size_t cluster_id;
};

/**
 * Calculate the Euclidean distance between two points
 * @param x1 x-coordinate of the first point
 * @param y1 y-coordinate of the first point
 * @param x2 x-coordinate of the second point
 * @param y2 y-coordinate of the second point
 * @return the Euclidean distance between the two points
*/
double calculate_euclidean_distance(const double x1, const double y1, const double x2, const double y2) {
    double x = x1 - x2;
    double y = y1 - y2;
    return sqrt((x) * (x) + (y) * (y));
}

int main(int argn, char* argv[]) {
    
    // 1. declare num_data_points and data_points
    size_t num_data_points = 16;
    struct Point data_points[num_data_points];

    // randomly initialize data_points between -GRID_SIZE and GRID_SIZE
    srand(0);
    for (size_t i = 0; i < num_data_points; i++) {
        data_points[i].x = (double)rand() / RAND_MAX * 2 * GRID_SIZE - GRID_SIZE;
        data_points[i].y = (double)rand() / RAND_MAX * 2 * GRID_SIZE - GRID_SIZE;
        data_points[i].cluster_id = -1;
    }

    // 2. declare num_k_clusters and k_clusters
    size_t num_k_clusters = 4;
    struct Point k_clusters[num_k_clusters];

    // 3. randomly initialize k_clusters from data_points
    for (size_t i = 0; i < num_k_clusters; i++) {
        k_clusters[i] = data_points[rand() % num_data_points];
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
            if (data_points[i].cluster_id == -1) {
                min_distance = INFINITY;
            } else {
                min_distance = calculate_euclidean_distance(data_points[i].x, data_points[i].y, k_clusters[data_points[i].cluster_id].x, k_clusters[data_points[i].cluster_id].y);
            }
            for (size_t j = 0; j < num_k_clusters; j++) {
                double distance = sqrt(pow(data_points[i].x - k_clusters[j].x, 2) + pow(data_points[i].y - k_clusters[j].y, 2));
                if (distance < min_distance) {
                    min_distance = distance;
                    data_points[i].cluster_id = j;
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
            double sum_x = 0;
            double sum_y = 0;
            size_t num_points = 0;
            for (int j = 0; j < num_data_points; j++) {
                if (data_points[j].cluster_id == i) {
                    sum_x += data_points[j].x;
                    sum_y += data_points[j].y;
                    num_points++;
                }
            }
            if (num_points > 0) {
                double new_x = sum_x / num_points;
                double new_y = sum_y / num_points;
                if (new_x != k_clusters[i].x || new_y != k_clusters[i].y) {
                    k_clusters[i].x = new_x;
                    k_clusters[i].y = new_y;
                    is_k_cluster_points_changed = 1;
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
    FILE* output_file = fopen("output.txt", "w");
    for (size_t i = 0; i < num_data_points; i++) {
        fprintf(output_file, "%f,%f,%zu\n", data_points[i].x, data_points[i].y, data_points[i].cluster_id);
    }
    fclose(output_file);

    
    
    return 0;
}