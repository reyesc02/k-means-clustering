/**
 * @file kmc-cuda.cu
 * @brief K-means clustering using CUDA
 * @details This program implements K-means clustering using CUDA.
 * The program reads data points from a CSV file and performs K-means clustering on the GPU.
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
 * To run this file you must be on a machine with a CUDA compatible GPU.
 * 
 * To compile this file run the following command:
 * nvcc -arch=sm_86 -O3 --compiler-options -march=native kmc-cuda.cu -o kmc-cuda -lm
 * 
 * To run the compiled file run the following command:
 * ./kmc-cuda <input_file> <k>
 * ./kmc-cuda <n> <d> <k>
*/

#include "matrix.hpp"

#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>

/**
 * Macro for checking CUDA errors following a CUDA launch or API call.
 */
#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
   }                                                                      \
}

/**
 * CUDA kernel to assign clusters to data points.
 * @param data_points The data points.
 * @param centroids The centroids.
 * @param cluster_id The cluster id of each data point.
 * @param n The number of data points.
 * @param d The number of dimensions.
 * @param k The number of clusters.
 * @param changed Flag to indicate if the cluster assignment has changed.
*/
__global__ void assign_clusters(double *data_points, double *centroids, int *cluster_id, int n, int d, int k, int *changed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		double min_dist = 1e9;
		int min_idx = cluster_id[i];
		for (int j = 0; j < k; j++) {
			double dist = 0;
			for (int l = 0; l < d; l++) {
				dist += (data_points[i * d + l] - centroids[j * d + l]) * (data_points[i * d + l] - centroids[j * d + l]);
			}
			if (dist < min_dist) {
				min_dist = dist;
				min_idx = j;
			}
		}
		if (cluster_id[i] != min_idx) {
			*changed = 1;
		}
		cluster_id[i] = min_idx;
	}
}

/**
 * CUDA kernel to recalculate centroids.
 * @param data_points The data points.
 * @param centroids The centroids.
 * @param cluster_id The cluster id of each data point.
 * @param n The number of data points.
 * @param d The number of dimensions.
 * @param k The number of clusters.
 * @param changed Flag to indicate if the cluster assignment has changed.
*/
__global__ void recalculate_centroids(double *data_points, double *centroids, int *cluster_id, int n, int d, int k, int *changed) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < k) {
		int cnt = 0;
		double *sum = new double[d];
		for (int j = 0; j < d; j++) {
			sum[j] = 0;
		}
		for (int j = 0; j < n; j++) {
			if (cluster_id[j] == i) {
				cnt++;
				for (int l = 0; l < d; l++) {
					sum[l] += data_points[j * d + l];
				}
			}
		}
		for (int j = 0; j < d; j++) {
			// check for tolerance
			double new_centroid = sum[j] / cnt;
			if (fabs(centroids[i * d + j] - new_centroid) > 1e-6) {
				centroids[i * d + j] = new_centroid;
				*changed = 1;
			}
		}
	}
}

/**
 * Parse command line arguments.
 * @param argn The number of command line arguments.
 * @param argv The command line arguments.
 * @param n The number of data points.
 * @param d The number of dimensions.
 * @param k The number of clusters.
 * @param data_points The data points.
 * @details If the number of command line arguments is 4, the function generates random data points.
 * If the number of command line arguments is 3, the function reads data points from a CSV file.
 * If the number of command line arguments is not 3 or 4, the function prints an error message and exits.
*/
void parse_cmd_line(int argn, char** argv, int &n, int &d, int &k, Matrix<double> &data_points) {
	if (argn == 4) {
		n = atoi(argv[1]);
		d = atoi(argv[2]);
		k = atoi(argv[3]);
		data_points = Matrix<double>(n, d);
		data_points.random();
	} else if (argn == 3) {
		data_points = Matrix<double>::from_csv(argv[1]);
		n = data_points.rows;
		d = data_points.cols;
		k = atoi(argv[2]);
	} else {
		printf("Usage: %s <n> <d> <k>\n", argv[0]);
		printf("Usage: %s <input_file> <k>\n", argv[0]);
		exit(1);
	}
}

/**
 * Main function.
 * @param argn The number of command line arguments.
 * @param argv The command line arguments.
 * @return 0 if the program runs successfully.
*/
int main(int argn, char **argv) {

	Matrix<double> data_points;
	int n;
	int d;
	int k;

	parse_cmd_line(argn, argv, n, d, k, data_points);

	//Matrix<double> data_points = Matrix<double>::from_csv("data/housing_cuda.csv");
	// Matrix<double> data_points = Matrix<double>(32767, 2);
	// data_points.random();

	printf("n: %d, d: %d, k: %d\n", n, d, k);

	Matrix<double> centroids(k, d);
	for (int i = 0; i < k; i++) {
		int idx = rand() % n;
		for (int j = 0; j < d; j++) {
			centroids(i, j) = data_points(idx, j);
		}
	}
	
	int cluster_id[n];
	for (int i = 0; i < n; i++) {
		cluster_id[i] = -1;
	}

	int max_iter = 300;
	int iter;

	// moved from inside loop
	double *d_data_points, *d_centroids;
	int *d_cluster_id, *d_changed;
	CHECK( cudaMalloc(&d_data_points, n * d * sizeof(double)) );
	CHECK( cudaMalloc(&d_centroids, k * d * sizeof(double)) );
	CHECK( cudaMalloc(&d_cluster_id, n * sizeof(int)) );
	CHECK( cudaMalloc(&d_changed, sizeof(int)) );
	CHECK( cudaMemcpy(d_data_points, data_points.data, n * d * sizeof(double), cudaMemcpyHostToDevice) );
	CHECK( cudaMemcpy(d_centroids, centroids.data, k * d * sizeof(double), cudaMemcpyHostToDevice) );
	CHECK( cudaMemcpy(d_cluster_id, cluster_id, n * sizeof(int), cudaMemcpyHostToDevice) );

	// start time
	clock_t start = clock();

	for (iter = 0; iter < max_iter; iter++) {
		int changed = 0;
		// for (int i = 0; i < n; i++) {
		// 	double min_dist = 1e9;
		// 	int min_idx = -1;
		// 	for (int j = 0; j < k; j++) {
		// 		double dist = 0;
		// 		for (int l = 0; l < d; l++) {
		// 			dist += (data_points(i, l) - centroids(j, l)) * (data_points(i, l) - centroids(j, l));
		// 		}
		// 		if (dist < min_dist) {
		// 			min_dist = dist;
		// 			min_idx = j;
		// 		}
		// 	}
		// 	if (cluster_id[i] != min_idx) {
		// 		cluster_id[i] = min_idx;
		// 		changed = true;
		// 	}
		// }

		CHECK( cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice) );
		assign_clusters<<<(n + 255) / 256, 256>>>(d_data_points, d_centroids, d_cluster_id, n, d, k, d_changed);
		CHECK( cudaDeviceSynchronize() );
		CHECK( cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost) );
		//cudaMemcpy(cluster_id, d_cluster_id, n * sizeof(int), cudaMemcpyDeviceToHost);

		//getchar();
		// print cluster_id
		// for (int i = 0; i < n; i++) {
		// 	std::cout << cluster_id[i] << " ";
		// }
		// getchar();

		if (changed == 0) {
			break;
		}
		// for (int i = 0; i < k; i++) {
		// 	int cnt = 0;
		// 	double sum[d];
		// 	for (int j = 0; j < d; j++) {
		// 		sum[j] = 0;
		// 	}
		// 	for (int j = 0; j < n; j++) {
		// 		if (cluster_id[j] == i) {
		// 			cnt++;
		// 			for (int l = 0; l < d; l++) {
		// 				sum[l] += data_points(j, l);
		// 			}
		// 		}
		// 	}
		// 	for (int j = 0; j < d; j++) {
		// 		centroids(i, j) = sum[j] / cnt;
		// 	}
		// }

		CHECK( cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice) );
		recalculate_centroids<<<(k + 255) / 256, 256>>>(d_data_points, d_centroids, d_cluster_id, n, d, k, d_changed);
		CHECK( cudaDeviceSynchronize() );
		//cudaMemcpy(centroids.data, d_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
		CHECK( cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost) );

		if (changed == 0) {
			break;
		}
	}

	// end time
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "time: " << time << std::endl;

	// moved from inside loop
	CHECK( cudaMemcpy(cluster_id, d_cluster_id, n * sizeof(int), cudaMemcpyDeviceToHost) );
	CHECK( cudaMemcpy(centroids.data, d_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost) );

	CHECK( cudaFree(d_data_points) );
	CHECK( cudaFree(d_centroids) );
	CHECK( cudaFree(d_cluster_id) );
	CHECK( cudaFree(d_changed) );

	// print result
	std::cout << "iter: " << iter << std::endl;

	// output data points, cluster id to csv to file
	// FILE *fp = fopen("output/housing_cuda_result.csv", "w");
	// for (int i = 0; i < n; i++) {
	// 	for (int j = 0; j < d; j++) {
	// 		fprintf(fp, "%lf", data_points(i, j));
	// 		if (j < d - 1) {
	// 			fprintf(fp, ",");
	// 		}
	// 	}
	// 	fprintf(fp, ",%d\n", cluster_id[i]);
	// }

	return 0;
}