/** K-means cuda
 * to run: nvcc -arch=sm_86 -O3 --compiler-options -march=native kmc-cuda.cu -o kmc-cuda -lm && ./kmc-cuda
*/
#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>

#include "matrix.hpp"

#define WRITE_OUTPUT 0

// global variables
unsigned long long num_data_points;
unsigned long long num_dimensions;
unsigned long long num_clusters;

// global pointers (matrices)
Matrix<double> data_points;
int *cluster_id;
Matrix<double> *centroids;

__global__ void assign_data_points_to_clusters(const double *data_points, int *cluster_id, const double *centroids, const unsigned long long *num_data_points, const unsigned long long *num_dimensions, const unsigned long long *num_clusters, bool *changed) {
	unsigned long long data_point_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (data_point_index >= *num_data_points) {
		return;
	}

	double min_distance = 1e9;
	int min_cluster_id = cluster_id[data_point_index];
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
	if (cluster_id[data_point_index] != min_cluster_id) {
		*changed = true;
		cluster_id[data_point_index] = min_cluster_id;
	}
}

__global__ void recalculate_centroids(const double *data_points, const int *cluster_id, double *centroids, const unsigned long long *num_data_points, const unsigned long long *num_dimensions, const unsigned long long *num_clusters, bool *changed) {
	unsigned long long cluster_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (cluster_index >= *num_clusters) {
		return;
	}
	for (int i = 0; i < *num_clusters; i++) {
		double* sum = new double[*num_dimensions];
		for (int j = 0; j < *num_dimensions; j++) {
			sum[j] = 0;
		}
		int count = 0;
		for (int k = 0; k < *num_data_points; k++) {
			if (cluster_id[k] == i) {
				for (int j = 0; j < *num_dimensions; j++) {
					sum[j] += data_points[k * (*num_dimensions) + j];
				}
				count++;
			}
		}
		if (count > 0) {
			for (int j = 0; j < *num_dimensions; j++) {
				centroids[i * (*num_dimensions) + j] = sum[j] / count;
				*changed = true;
			}
		}
		// free memory
		delete[] sum;
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
	FILE* file;
		if (WRITE_OUTPUT) {
		sprintf(output_file, "output/cuda-output-%s-data.csv", timestamp);
		file = fopen(output_file, "w");
		if (file == NULL) { perror("error writing output"); return; }
		for (size_t i = 0; i < num_data_points; i++) {
			for (size_t j = 0; j < num_dimensions; j++) {
				fprintf(file, "%f,", data_points.data[i * num_dimensions + j]);
			}
			fprintf(file, "%d\n", cluster_id[i]);
		}
		fclose(file);
	}

    // output program info
    sprintf(output_file, "output/cuda-output-%s-info.txt", timestamp);
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
        num_dimensions = 2;
        num_clusters = 8;
        data_points = Matrix<double>(num_data_points, num_dimensions);
		data_points.random();
    } else if (argn == 3) {
        data_points = Matrix<double>::from_csv(argv[1]);
		printf("rows: %d, cols: %d\n", data_points.rows, data_points.cols);
        if (data_points.data == NULL) { perror("error reading input"); return false; }
        num_dimensions = data_points.cols;
        num_data_points = data_points.rows;
        num_clusters = atoi(argv[2]);
    } else if (argn == 4) {
        num_data_points = atoi(argv[1]);
        num_dimensions = atoi(argv[2]);
        num_clusters = atoi(argv[3]);
        data_points = Matrix<double>(num_data_points, num_dimensions);
		data_points.random();
    } else {
        printf("Usage: %s <input_file> <num_k_clusters>\n", argv[0]);
        printf("Usage: %s <num_data_points> <n_dimensions> <num_k_clusters> <grid_size> [seed]\n", argv[0]);
        return false;
    }
    
    // initialize cluster_id to -1
    cluster_id = new int[num_data_points];
	for (int i = 0; i < num_data_points; i++) {
		cluster_id[i] = -1;
	}

    // randomly initialize k_clusters from data_points
    centroids = new Matrix<double>(num_clusters, num_dimensions);
	for (int i = 0; i < num_clusters; i++) {
		int random_index = rand() % num_data_points;
		for (int j = 0; j < num_dimensions; j++) {
			centroids->data[i * num_dimensions + j] = data_points.data[random_index * num_dimensions + j];
		}
	}

    return true;
}

int main(int argn, char* argv[]) {
	// parse command line arguments
    if(!parse_command_line_arguments(argn, argv)) { return 1; };

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
	cudaMemcpy(d_data_points, data_points.data, num_data_points * num_dimensions * sizeof(double), cudaMemcpyHostToDevice);

	// copy cluster id to device
	cudaMalloc(&d_cluster_id, num_data_points * sizeof(int));
	cudaMemcpy(d_cluster_id, cluster_id, num_data_points * sizeof(int), cudaMemcpyHostToDevice);

	// copy centroids to device
	cudaMalloc(&d_centroids, num_clusters * num_dimensions * sizeof(double));
	cudaMemcpy(d_centroids, centroids->data, num_clusters * num_dimensions * sizeof(double), cudaMemcpyHostToDevice);

	// k-means algorithm
	int max_iterations = 300;
	int iteration;

	// copy changed to device
	bool changed = false;
	const bool false_value = false;
	bool *d_changed;
	cudaMalloc(&d_changed, sizeof(bool));

	// start time
	clock_t start = clock();

	for (iteration = 0; iteration < max_iterations; iteration++) {
		// assign data points to clusters
		cudaMemcpy(d_changed, &false_value, sizeof(bool), cudaMemcpyHostToDevice); // set changed to false (host)
		assign_data_points_to_clusters<<<(num_data_points + 255) / 256, 256>>>(d_data_points, d_cluster_id, d_centroids, d_num_data_points, d_num_dimensions, d_num_clusters, d_changed);
		cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

		

		if (!changed) {
			// copy cluster id to host
			cudaMemcpy(cluster_id, d_cluster_id, num_data_points * sizeof(int), cudaMemcpyDeviceToHost);
			// copy centroids to host
			cudaMemcpy(centroids->data, d_centroids, num_clusters * num_dimensions * sizeof(double), cudaMemcpyDeviceToHost);
			break;
		}

		// recalculate centroids
		cudaMemcpy(d_changed, &false_value, sizeof(bool), cudaMemcpyHostToDevice); // set changed to false (host)
		recalculate_centroids<<<(num_clusters + 255) / 256, 256>>>(d_data_points, d_cluster_id, d_centroids, d_num_data_points, d_num_dimensions, d_num_clusters, d_changed);
		cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);

		

		if (!changed) {
			// copy cluster id to host
			cudaMemcpy(cluster_id, d_cluster_id, num_data_points * sizeof(int), cudaMemcpyDeviceToHost);
			// copy centroids to host
			cudaMemcpy(centroids->data, d_centroids, num_clusters * num_dimensions * sizeof(double), cudaMemcpyDeviceToHost);
			break;
		}
	}

	// end time
	clock_t end = clock();
	double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

	printf("converged after %d iterations\n", iteration);
	printf("time taken: %f seconds\n", time_taken);

	// print results to file
	print_results_to_file(iteration, 0, time_taken);

	// free memory
	delete centroids;
	delete cluster_id;

	cudaFree(d_data_points);
	cudaFree(d_cluster_id);
	cudaFree(d_centroids);

	cudaFree(d_num_data_points);
	cudaFree(d_num_dimensions);
	cudaFree(d_num_clusters);

	cudaFree(d_changed);

	return 0;
}