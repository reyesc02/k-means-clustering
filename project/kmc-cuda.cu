/** K-means cuda
 * to run: nvcc -arch=sm_86 -O3 --compiler-options -march=native kmc-cuda.cu -o kmc-cuda -lm && ./kmc-cuda
*/

#include "matrix.hpp"

#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>

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

int main() {
	//Matrix<double> data_points = Matrix<double>::from_csv("data/housing_cuda.csv");
	Matrix<double> data_points = Matrix<double>(32767, 2);
	data_points.random();

 	int n = data_points.rows;
	int d = data_points.cols;
	int k = 8;

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
	cudaMalloc(&d_data_points, n * d * sizeof(double));
	cudaMalloc(&d_centroids, k * d * sizeof(double));
	cudaMalloc(&d_cluster_id, n * sizeof(int));
	cudaMalloc(&d_changed, sizeof(int));
	cudaMemcpy(d_data_points, data_points.data, n * d * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids, centroids.data, k * d * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cluster_id, cluster_id, n * sizeof(int), cudaMemcpyHostToDevice);

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

		cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);
		assign_clusters<<<(n + 255) / 256, 256>>>(d_data_points, d_centroids, d_cluster_id, n, d, k, d_changed);
		cudaDeviceSynchronize();
		cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(cluster_id, d_cluster_id, n * sizeof(int), cudaMemcpyDeviceToHost);

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

		cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);
		recalculate_centroids<<<(k + 255) / 256, 256>>>(d_data_points, d_centroids, d_cluster_id, n, d, k, d_changed);
		cudaDeviceSynchronize();
		cudaMemcpy(centroids.data, d_centroids, k * d * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

		if (changed == 0) {
			break;
		}
	}

	// moved from inside loop

	cudaFree(d_data_points);
	cudaFree(d_centroids);
	cudaFree(d_cluster_id);
	cudaFree(d_changed);

	// print result
	std::cout << "iter: " << iter << std::endl;

	// output data points, cluster id to csv to file
	FILE *fp = fopen("output/housing_cuda_result.csv", "w");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < d; j++) {
			fprintf(fp, "%lf", data_points(i, j));
			if (j < d - 1) {
				fprintf(fp, ",");
			}
		}
		fprintf(fp, ",%d\n", cluster_id[i]);
	}


	return 0;
}