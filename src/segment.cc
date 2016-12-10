

#include <GridCut/GridGraph_3D_6C_MT.h>
#include <AlphaExpansion/AlphaExpansion_3D_6C_MT.h>

int main() {
    typedef AlphaExpansion_3D_6C_MT<int,int,int> AlphaExpansion;
    AlphaExpansion *alpha_expansion = new AlphaExpansion(100, 100, 10, 0, 0);
}
