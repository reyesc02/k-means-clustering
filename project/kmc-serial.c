/**
 * to compile and run this file, run the following command:
 *    gcc -o kmc-serial kmc-serial.c && ./kmc-serial
 * 
*/

#include <stdio.h>
#include <math.h>

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

    // psuedo code for naive k-means algorithm:
    // 1. Initialize k centroids randomly
    // 2. Assign each data point to the nearest centroid
    // 3. Compute the new centroids by averaging the data points assigned to each centroid
    // 4. Repeat steps 2 and 3 until the centroids do not change significantly

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

    return 0;
}