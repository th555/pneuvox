#ifndef PV_CHAMBER_H
#define PV_CHAMBER_H

#include "Voxelyze.h"
#include <vector>

/*
This class defines an object that holds a single pneumatic chamber.
 - Voxel walls are made up of quads between the voxel centers themselves.
 - The update method should be called every simulation step to apply the correct forces on the voxels

*/

class PV_Chamber
{
public:
    PV_Chamber();

    /***/
    void updateVolume();
    void preUpdate();
    void update();
    /***/

    float pressure;
    float quantity;
    float volume;
    void updatePressure(); // Pressure of this chamber in isolation (regardless of if it is linked with others), absolute.
    // If pressure relative to ambient is needed caller can subtract 1 bar.
    // In bar (1 bar = 100 000 Pa, 1 Pa = 1 N/m^2)



    float area(); // Current surface area of the chamber walls

    void initPressure(); // Initialize the quantity of air from the known volume assuming
                         // a pressure of 1 bar

    // Add a quad of voxels to the chamber wall. Inside/outside is determined by the order of the voxels (clockwise/anticlockwise).
    // Use left-hand thumb rule: if you follow the voxels in order a, b, c, d,, your thumb points to the inside of the chamber.
    void addQuad(CVX_Voxel* a, CVX_Voxel* b, CVX_Voxel* c, CVX_Voxel* d);

    #ifndef NOGUI
    void draw(void* tf);
    #endif

    int nTris(); // Return the number of quads

    std::vector<CVX_Voxel*> tris; // Contiguous array of tris, the i'th tri is at tris[i*3+0] ... tris[i*3+2]
                                  // The 2nd point of the 3 is always the right angle of a quad, the 1'st and 3rd point form the diagonal


private:
    std::vector<CVX_Voxel*> uniqueVoxels; // Contains each voxel once, used for resetting forces at the start of each timestep

    void applyPneuForce(float pressure, int tri_i);
    void applyTriForce(Vec3D<double> force, int tri_i);

    float triArea(int i);
};

#endif //PV_CHAMBER_H