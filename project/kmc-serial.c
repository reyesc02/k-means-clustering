/**
 * to compile and run this file, run the following command:
 *    gcc -lm -o kmc-serial kmc-serial.c && ./kmc-serial
 * 
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// constant for the grid size
// the grid size is the size of the 2D grid where the data points are located
#define GRID_SIZE 100

struct Point {
    double x;
    double y;
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

void test_2d_point() {
    // for now we will just calculate the distance between two points
    // a point is represented by a pair of coordinates (x, y)
    // the distance between two points (x1, y1) and (x2, y2) is given by:
    // sqrt((x1 - x2)^2 + (y1 - y2)^2)

    // let's calculate the distance between two points
    double x1, y1;
    x1 = 1.0;
    y1 = 2.0;

    double x2, y2;
    x2 = 3.0;
    y2 = 4.0;

    double distance = calculate_euclidean_distance(x1, y1, x2, y2);

    printf("The distance between (%f, %f) and (%f, %f) is %f\n", x1, y1, x2, y2, distance);
}

int main(int argn, char* argv[]) {

    // pseudo code for kmc algorithm
    // 1. initialize the data points
    // 2. choose the number of k clusters
    // 3. select k random points as the initial centroids
    // 4. assign each data point to the nearest centroid
    // 5. recalculate the centroids of the clusters
    // 6. repeat steps 4 and 5 until until one of the following conditions is met:
    //    - the centroids do not change
    //    - the maximum number of iterations is reached
    //    - the points remain in the same cluster

    // 1. initialize the data points
    // the array of struct Point will store the data points
    size_t num_data_points = 25;
    struct Point data_points[num_data_points];

    // randomly initialize the data points between -GRID_SIZE and GRID_SIZE
    for (size_t i = 0; i < num_data_points; i++) {
        data_points[i].x = (double)rand() / RAND_MAX * 2 * GRID_SIZE - GRID_SIZE;
        data_points[i].y = (double)rand() / RAND_MAX * 2 * GRID_SIZE - GRID_SIZE;
        printf("Data point %zu: (%f, %f)\n", i, data_points[i].x, data_points[i].y);
    }

    printf("\n");

    // 2. choose the number of k clusters
    size_t k = 3;

    // 3. select k random points as the initial centroids from the data points
    struct Point centroids[k];
    for (size_t i = 0; i < k; i++) {
        centroids[i] = data_points[rand() % num_data_points];
        printf("Centroid %zu: (%f, %f)\n", i, centroids[i].x, centroids[i].y);
    }

    return 0;
}