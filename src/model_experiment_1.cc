
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <vector>
#include <random>
#include <GridCut/GridGraph_3D_6C_MT.h>
#include <AlphaExpansion/AlphaExpansion_3D_6C_MT.h>

int main() {
    typedef AlphaExpansion_3D_6C_MT<int, float, float> AlphaExpansion;

    const int width = 10;
    const int height = 10;
    const int depth = 10;
    const int voxels = width * height * depth;
    const int num_labels = 3;
    float *cost = new float[width * height * depth * num_labels];
    AlphaExpansion::SmoothCostFn smooth_fn = [](int pix1, int pix2, int lab1, int lab2) -> float { return (lab1 != lab2) ? 1.0f : 0.0f; };
    const int num_threads = 1;
    const int block_size = 10;

    const float radius1 = 2;
    const float radius2 = 4;

    std::random_device random_device;
    std::mt19937 random_generator(random_device());
    std::normal_distribution<> normal_distr(1,1);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                float radius = pow((i - width / 2), 2) + pow((j - height / 2), 2) + pow((k - depth / 2), 2);
                int label = 0;
                if (radius < radius1 * radius1) label = 1;
                else if (radius < radius2 * radius2) label = 2;

                for (int l = 0; l < num_labels; ++l) {
                    float random = normal_distr(random_generator);

                    int pos = (i + (j + k * height) * width) * num_labels + l;
                    cost[pos] = (l == label) ? random : 1 + random * 2;
                    printf("% .1f ", cost[pos]);
                }
                printf("  ");
            }
            printf("\n");
        }
        printf("\n");
    }

    std::unique_ptr<AlphaExpansion> alpha_expansion(
        new AlphaExpansion(width, height, depth, num_labels, &cost[0], smooth_fn,
                           num_threads, block_size));
    alpha_expansion->perform();

    int correct_labels = 0;
    int incorrect_labels = 0;
    const int *labeling = alpha_expansion->get_labeling();
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                float radius = pow((i - width / 2), 2) + pow((j - height / 2), 2) + pow((k - depth / 2), 2);
                int label = 0;
                if (radius < radius1 * radius1) label = 1;
                else if (radius < radius2 * radius2) label = 2;

                int pos = i + (j + k * height) * width;
                printf("%d ", labeling[pos]);

                if (labeling[pos] == label) {
                    ++correct_labels;
                } else {
                    ++incorrect_labels;
                }
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n\ncorrect_labels = %d (%.3f)\n", correct_labels, float(correct_labels) / voxels);
    printf("incorrect_labels = %d (%.3f)\n", incorrect_labels, float(incorrect_labels) / voxels);
}
