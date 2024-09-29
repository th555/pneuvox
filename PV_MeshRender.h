/*******************************************************************************
Copyright (c) 2015, Jonathan Hiller
To cite academic use of Voxelyze: Jonathan Hiller and Hod Lipson "Dynamic Simulation of Soft Multimaterial 3D-Printed Objects" Soft Robotics. March 2014, 1(1): 88-101.
Available at http://online.liebertpub.com/doi/pdfplus/10.1089/soro.2013.0010

This file is part of Voxelyze.
Voxelyze is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Voxelyze is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
See <http://www.opensource.org/licenses/lgpl-3.0.html> for license details.
*******************************************************************************/

/*
Copied from VoxCAD to here because we want to use raylib for rendering, and need the private member vertices etc...
*/

#ifndef PV_MESH_H
#define PV_MESH_H

#include "Voxelyze.h"
#include <vector>
#include "raylib.h"
#include "PV_Chamber.h"
#include "PV_Pneunet.h"
#include "PV_Sensor.h"

//link direction to clockwise vertex lookup info:
extern CVX_Voxel::voxelCorner CwLookup[6][4];

//! Voxelyze mesh visualizer
/*!
A simple way to generate a deformed mesh reflecting the current state of a voxelyze object. After constructing with a pointer to the desired voxelyze object the mesh is ready. If the state of the voxelyze object has changed or a different coloring is desired, simply call updateMesh(). If voxels are added or subtracted to the voxelyze object, generateMesh() must be called to regenerate the mesh before calling updateMesh or drawGl().

The mesh can be drawn in an initialized OpenGL window by defining USE_OPEN_GL in the preprocessor and calling glDraw from within the drawing loop. An obj mesh file can also be generated at any time.
*/
class PV_MeshRender
{
public:
    //! Defines various ways of coloring the voxels in the 3D mesh
    enum viewColoring {
        MATERIAL, //!< Display the material color specified by its RGB values
        FAILURE, //!< Display the current failure status (red=failed, yellow=yielded, white=ok)
        STATE_INFO //!< Display a color coded "head map" of the specified CVoxelyze::stateInfoType (displacement, kinetic energy, etc.)
    };

    PV_MeshRender(CVoxelyze* voxelyzeInstance, PV_Pneunet* pneunet, std::string filename=""); //!< Initializes this mesh visualization with the specified voxelyze instance. This voxelyze pointer must remain valid for the duration of this object. @param[in] voxelyzeInstance The voxelyze instance to link this mesh object to.
    // If a filename is given a video will be recorded and saved with that filename
    void generateMesh(); //!< Generates (or regenerates) this mesh from the linked voxelyze object. This must be called whenever voxels are added or removed in the simulation.
    void updateMesh(viewColoring colorScheme = MATERIAL, CVoxelyze::stateInfoType stateType = CVoxelyze::DISPLACEMENT); //!< Updates the mesh according to the current state of the linked voxelyze object and the coloring scheme specified by the arguments. @param[in] colorScheme The coloring scheme. @param[in] stateType If colorScheme = STATE_INFO, this argument determines the state to color the object according to. Only kinetic energy, strain energy, displacement, and pressure are currently supported.

    void updateCameraTarget();
    void updateCameraTargetPicture();
    void renderRobot(std::vector<std::stringstream*> info, std::vector<std::string*> info2, PV_Sensor* sensor, Vec3D<double> robotCenter, Vec3D<double> trueCenter, std::string filename=""); // If a filename is given an image will be saved under that name

    void saveObj(const char* filePath); //!< Save the current deformed mesh as an obj file to the path specified. Coloring is not supported yet. @param[in] filePath File path to save the obj file as. Creates or overwrites.
    void rlDraw(); //!< Executes openGL drawing commands to draw this mesh in an Open GL window if USE_OPEN_GL is defined.

    void closePipe();

    // Transformation matrix to apply to go from voxcad to raylib coordinates,
    // otherwise we keep bumping into raylib default assumptions
    // Swap Y and Z axes, scale up, move up half a voxel so the models rest on the floor at z=0
    Matrix tf;

    Camera camera;
    Camera camera2; // for screenshots

    PV_Pneunet* pneunet; // for drawing the chambers

    int screenWidth;
    int screenHeight;

private:
    CVoxelyze* Vx;

    std::vector<float> vertices; //vx1, vy1, vz1, vx2, vy2, vz2, vx3, ...
    std::vector<CVX_Voxel*> vertexLinks; //vx1NNN, vx1NNP, [CVX_Voxel::voxelCorner enum order], ... vx2NNN, vx2NNp, ... (null if no link) 

    std::vector<int> quads; //q1v1, q1v2, q1v3, q1v4, q2v1, q2v2, ... (ccw order)
    std::vector<float> quadColors; //q1R, q1G, q1B, q2R, q2G, q2B, ... 
    std::vector<int> quadVoxIndices; //q1n, q2n, q3n, ... 
    std::vector<float> quadNormals; //q1Nx, q1Ny, q1Nz, q2Nx, q2Ny, q2Nz, ... (needs updating with mesh deformation)

    std::vector<int> lines; //l1v1, l1v2, l2v1, l2v2, ...

    float jetMapR(float val) {if (val<0.5f) return 0.0f; else if (val>0.75f) return 1.0f; else return val*4-2;}
    float jetMapG(float val) {if (val<0.25f) return val*4; else if (val>0.75f) return 4-val*4; else return 1.0f;}
    float jetMapB(float val) {if (val>0.5f) return 0.0f; else if (val<0.25f) return 1.0f; else return 2-val*4;}

    float linkMaxColorValue(CVX_Voxel* pV, CVoxelyze::stateInfoType coloring); //for link properties, the max

    FILE *pPipe = nullptr;

    Font font;
};



#endif // PV_MESH_H
