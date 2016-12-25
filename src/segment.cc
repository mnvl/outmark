

#include <stdlib.h>
#include <memory>
#include <vector>
#include <GridCut/GridGraph_3D_6C_MT.h>
#include <AlphaExpansion/AlphaExpansion_3D_6C_MT.h>

int main() {
    typedef AlphaExpansion_3D_6C_MT<int, float, float> AlphaExpansion;

    const int width = 100;
    const int height = 100;
    const int depth = 100;
    const int num_labels = 3;
    float *cost = new float[width * height * depth * num_labels];
    AlphaExpansion::SmoothCostFn smooth_fn = [](int pix1, int pix2, int lab1, int lab2) -> float { return lab1 != lab2; };
    const int num_threads = 4;
    const int block_size = 10;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < depth; ++k) {
                for (int l = 0; l < num_labels; ++l) {
                    int pos = ((i + j * width) * depth + k) * num_labels + l;
                    cost[pos] = 1;
                }
            }
        }
    }

    std::unique_ptr<AlphaExpansion> alpha_expansion(
        new AlphaExpansion(width, height, depth, num_labels, &cost[0], smooth_fn,
                           num_threads, block_size));
    alpha_expansion->perform();
}
