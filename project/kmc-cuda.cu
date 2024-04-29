/** K-means cuda
 * to run: nvcc -arch=sm_86 -O3 --compiler-options -march=native kmc-cuda.cu -o kmc-cuda -lm && ./kmc-cuda
*/
#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>

#include "matrix.hpp"

// global variables
unsigned long long num_data_points;
unsigned long long num_dimensions;
unsigned long long num_clusters;

// global pointers (matrices)
Matrix<double> *data_points;
int *cluster_id;
Matrix<double> *centroids;

__global__ void assign_data_points_to_clusters(const double *data_points, int *cluster_id, const double *centroids, const unsigned long long *num_data_points, const unsigned long long *num_dimensions, const unsigned long long *num_clusters) {
	unsigned long long data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (data_point_index >= *num_data_points) {
		return;
	}

	double min_distance = 1e9;
	int min_cluster_id = -1;
	for (unsigned long long cluster_index = 0; cluster_index < *num_clusters; cluster_index++) {
		double distance = 0;
		for (unsigned long long dimension_index = 0; dimension_index < *num_dimensions; dimension_index++) {
			double diff = data_points[data_point_index * (*num_dimensions) + dimension_index] - centroids[cluster_index * (*num_dimensions) + dimension_index];
			distance += diff * diff;
		}
		if (distance < min_distance) {
			min_distance = distance;
			min_cluster_id = cluster_index;
		}
	}
	cluster_id[data_point_index] = min_cluster_id;

}

__global__ void recalculate_centroids(const double *data_points, const int *cluster_id, double *centroids, const unsigned long long *num_data_points, const unsigned long long *num_dimensions, const unsigned long long *num_clusters) {
	unsigned long long cluster_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (cluster_index >= *num_clusters) {
		return;
	}

	unsigned long long num_data_points_in_cluster = 0;
	for (unsigned long long data_point_index = 0; data_point_index < *num_data_points; data_point_index++) {
		if (cluster_id[data_point_index] == cluster_index) {
			num_data_points_in_cluster++;
		}
	}

	if (num_data_points_in_cluster == 0) {
		return;
	}

	for (unsigned long long dimension_index = 0; dimension_index < *num_dimensions; dimension_index++) {
		centroids[cluster_index * (*num_dimensions) + dimension_index] = 0;
	}

	for (unsigned long long data_point_index = 0; data_point_index < *num_data_points; data_point_index++) {
		if (cluster_id[data_point_index] == cluster_index) {
			for (unsigned long long dimension_index = 0; dimension_index < *num_dimensions; dimension_index++) {
				centroids[cluster_index * (*num_dimensions) + dimension_index] += data_points[data_point_index * (*num_dimensions) + dimension_index];
			}
		}
	}

	for (unsigned long long dimension_index = 0; dimension_index < *num_dimensions; dimension_index++) {
		centroids[cluster_index * (*num_dimensions) + dimension_index] /= num_data_points_in_cluster;
	}
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
    char output_file[128];
    sprintf(output_file, "output/serial-output-%s-data.csv", timestamp);
    FILE* file = fopen(output_file, "w");
    if (file == NULL) { perror("error writing output"); return; }
    for (size_t i = 0; i < num_data_points; i++) {
        for (size_t j = 0; j < num_dimensions; j++) {
            fprintf(file, "%f,", data_points->data[i * num_dimensions + j]);
        }
        fprintf(file, "%d\n", cluster_id[i]);
    }
    fclose(file);

    // output program info
    sprintf(output_file, "output/serial-output-%s-info.txt", timestamp);
    file = fopen(output_file, "w");
    if (file == NULL) { perror("error writing output"); return; }
    fprintf(file, "Data Points: %llu\n", num_data_points);
    fprintf(file, "Dimensions: %llu\n", num_dimensions);
    fprintf(file, "K Clusters: %llu\n\n", num_clusters);
    fprintf(file, "Time Taken: %f seconds\n\n", time_taken);

    // print each cluster and its number of data points
    for (size_t i = 0; i < num_clusters; i++) {
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

int main(int argn, const char* argv[]) {
	num_data_points = 32767;
	num_dimensions = 2;
	num_clusters = 8;

	data_points = new Matrix<double>(num_data_points, num_dimensions);
	data_points->random();

	// printf("Data points:\n");
	// for (int i = 0; i < num_data_points; i++) {
	// 	for (int j = 0; j < num_dimensions; j++) {
	// 		printf("%f ", data_points->data[i * num_dimensions + j]);
	// 	}
	// 	printf("\n");
	// }

	cluster_id = new int[num_data_points];
	for (int i = 0; i < num_data_points; i++) {
		cluster_id[i] = -1;
	}

	// printf("Cluster id:\n");
	// for (int i = 0; i < num_data_points; i++) {
	// 	printf("%d ", cluster_id[i]);
	// }
	// printf("\n");

	srand(0);
	centroids = new Matrix<double>(num_clusters, num_dimensions);
	for (int i = 0; i < num_clusters; i++) {
		int random_index = rand() % num_data_points;
		for (int j = 0; j < num_dimensions; j++) {
			centroids->data[i * num_dimensions + j] = data_points->data[random_index * num_dimensions + j];
		}
	}

	// printf("Centroids:\n");
	// for (int i = 0; i < num_clusters; i++) {
	// 	for (int j = 0; j < num_dimensions; j++) {
	// 		printf("%f ", centroids->data[i * num_dimensions + j]);
	// 	}
	// 	printf("\n");
	// }


	// copy global variables to device
	unsigned long long *d_num_data_points;
	unsigned long long *d_num_dimensions;
	unsigned long long *d_num_clusters;

	cudaMalloc(&d_num_data_points, sizeof(unsigned long long));
	cudaMemcpy(d_num_data_points, &num_data_points, sizeof(unsigned long long), cudaMemcpyHostToDevice);
	
	cudaMalloc(&d_num_dimensions, sizeof(unsigned long long));
	cudaMemcpy(d_num_dimensions, &num_dimensions, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	cudaMalloc(&d_num_clusters, sizeof(unsigned long long));
	cudaMemcpy(d_num_clusters, &num_clusters, sizeof(unsigned long long), cudaMemcpyHostToDevice);

	// copy data points to device
	double *d_data_points;
	int *d_cluster_id;
	double *d_centroids;
	cudaMalloc(&d_data_points, num_data_points * num_dimensions * sizeof(double));
	cudaMemcpy(d_data_points, data_points->data, num_data_points * num_dimensions * sizeof(double), cudaMemcpyHostToDevice);

	// copy cluster id to device
	cudaMalloc(&d_cluster_id, num_data_points * sizeof(int));
	cudaMemcpy(d_cluster_id, cluster_id, num_data_points * sizeof(int), cudaMemcpyHostToDevice);

	// copy centroids to device
	cudaMalloc(&d_centroids, num_clusters * num_dimensions * sizeof(double));
	cudaMemcpy(d_centroids, centroids->data, num_clusters * num_dimensions * sizeof(double), cudaMemcpyHostToDevice);

	int max_iterations = 300;
	int iteration;

	for (iteration = 0; iteration < max_iterations; iteration++) {
		// assign data points to clusters
		assign_data_points_to_clusters<<<(num_data_points + 255) / 256, 256>>>(d_data_points, d_cluster_id, d_centroids, d_num_data_points, d_num_dimensions, d_num_clusters);

		// copy cluster id back to host
		cudaMemcpy(cluster_id, d_cluster_id, num_data_points * sizeof(int), cudaMemcpyDeviceToHost);

		// recalculate centroids
		recalculate_centroids<<<(num_clusters + 255) / 256, 256>>>(d_data_points, d_cluster_id, d_centroids, d_num_data_points, d_num_dimensions, d_num_clusters);

		// copy centroids back to host
		cudaMemcpy(centroids->data, d_centroids, num_clusters * num_dimensions * sizeof(double), cudaMemcpyDeviceToHost);
	}

	printf("converged after %d iterations\n", iteration);

	// print results to file
	print_results_to_file(iteration, 0, 0);

	// free memory
	delete data_points;
	delete centroids;
	delete cluster_id;

	cudaFree(d_data_points);
	cudaFree(d_cluster_id);
	cudaFree(d_centroids);

	cudaFree(d_num_data_points);
	cudaFree(d_num_dimensions);
	cudaFree(d_num_clusters);



	return 0;
}