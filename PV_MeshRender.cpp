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

#include "PV_MeshRender.h"
#include "VX_Voxel.h"
#include "raylib.h"
#include "raymath.h" // For matrixrotate etc.
#include "rlgl.h"
#include "pneuvox.h"

//for file output
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iomanip>


CVX_Voxel::voxelCorner CwLookup[6][4] = {
    {CVX_Voxel::PNN, CVX_Voxel::PPN, CVX_Voxel::PPP, CVX_Voxel::PNP}, //linkDirection::X_POS
    {CVX_Voxel::NNN, CVX_Voxel::NNP, CVX_Voxel::NPP, CVX_Voxel::NPN}, //linkDirection::X_NEG
    {CVX_Voxel::NPN, CVX_Voxel::NPP, CVX_Voxel::PPP, CVX_Voxel::PPN}, //linkDirection::Y_POS
    {CVX_Voxel::NNN, CVX_Voxel::PNN, CVX_Voxel::PNP, CVX_Voxel::NNP}, //linkDirection::Y_NEG
    {CVX_Voxel::NNP, CVX_Voxel::PNP, CVX_Voxel::PPP, CVX_Voxel::NPP}, //linkDirection::Z_POS
    {CVX_Voxel::NNN, CVX_Voxel::NPN, CVX_Voxel::PPN, CVX_Voxel::PNN}  //linkDirection::Z_NEG
};


PV_MeshRender::PV_MeshRender(CVoxelyze* voxelyzeInstance, PV_Pneunet* pneunet, std::string filename)
{
    Vx = voxelyzeInstance;
    this->pneunet = pneunet;

    float s = 100.0f;
    tf = {
           s, 0.0f, 0.0f,                   0.0f,
        0.0f, 0.0f,    s, Vx->voxelSize()*s*0.5f,
        0.0f,   -s, 0.0f,                   0.0f,
        0.0f, 0.0f, 0.0f,                   1.0f
    };
    generateMesh();

    // Init Raylib stuff
    //--------------------------------------------------------------------------------------
    screenWidth = 1200;
    screenHeight = 800;


    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screenWidth, screenHeight, "Pneumatic Voxel Robots");

    font = LoadFont("fonts/DejaVuSansMono.ttf");


    // Additional camera to take screenshots head-on
    camera2 = { 0 };
    camera2.position = (Vector3){ 5.0, 20.0, 5.0 };
    camera2.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera2.fovy = 40.0f;                                // Camera field-of-view Y
    camera2.projection = CAMERA_PERSPECTIVE;             // Camera mode type
    SetCameraMode(camera2, CAMERA_FREE);

    // Define the camera to look into our 3d world
    camera = { 0 };
    camera.position = (Vector3){ 25, 5.0, 25 };  // Camera position
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera mode type
    SetCameraMode(camera, CAMERA_FREE);

    SetCameraPanControl(MOUSE_BUTTON_RIGHT);
    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second

    if(!filename.empty()){
        std::stringstream sstm;
        sstm << "ffmpeg -y -f rawvideo -vcodec rawvideo -s " << screenWidth << "x" << screenHeight  <<" -pix_fmt rgba -i - -c:v libx264 -shortest " << "videos/" << filename << ".mp4";

        if ( !(pPipe = popen(sstm.str().c_str(), "w")) ) {
            std::cout << "popen error" << std::endl;
            exit(1);
        }
    }
}

void PV_MeshRender::updateCameraTarget(){
    // Update camera to follow robot
    Vec3D<double> c = centerOfGravity(Vx);
    Vector3 cc = (Vector3){c.x, c.y, c.z};
    Vector3 ct = Vector3Transform(cc, tf);
    camera.target = (Vector3){ct.x+5, 5, ct.z};
    // SetCameraMode(camera, CAMERA_FREE);
    UpdateCamera(&camera);
}

void PV_MeshRender::updateCameraTargetPicture(){
    // Update camera to follow robot
    Vec3D<double> c = centerOfGravity(Vx);
    Vector3 cc = (Vector3){c.x, c.y, c.z};
    Vector3 ct = Vector3Transform(cc, tf);
    camera2.target = (Vector3){ct.x+5, 5, ct.z-5};
    camera2.position = (Vector3){ct.x+5, 40, ct.z-4};
    SetCameraMode(camera2, CAMERA_FREE); // Have to do these two things in the right order
    UpdateCamera(&camera2);              // to actually switch the camera
}

void PV_MeshRender::generateMesh()
{
    vertices.clear();
    vertexLinks.clear();
    quads.clear();
    quadColors.clear();
    quadVoxIndices.clear();
    quadNormals.clear();
    lines.clear();

    int minX = Vx->indexMinX();
    int sizeX = Vx->indexMaxX()-minX+1;
    int minY = Vx->indexMinY();
    int sizeY = Vx->indexMaxY()-minY+1;
    int minZ = Vx->indexMinZ();
    int sizeZ = Vx->indexMaxZ()-minZ+1;

    CArray3D<int> vIndMap; //maps
    vIndMap.setDefaultValue(-1);
    vIndMap.resize(sizeX+1, sizeY+1, sizeZ+1, minX, minY, minZ);
    int vertexCounter = 0;
    
    //for each possible voxel location: (fill in vertices)
    int vCount = Vx->voxelCount();
    for (int k=0; k<vCount; k++){
        CVX_Voxel* pV = Vx->voxel(k);
        int x=pV->indexX(), y=pV->indexY(), z=pV->indexZ();

        Index3D thisVox(x, y, z);
        for (int i=0; i<6; i++){ //for each direction that a quad face could exist
            if (pV->adjacentVoxel((CVX_Voxel::linkDirection)i)) continue;
            for (int j=0; j<4; j++){ //for each corner of the (exposed) face in this direction
                CVX_Voxel::voxelCorner thisCorner = CwLookup[i][j];
                Index3D thisVertInd3D = thisVox + Index3D(thisCorner&(1<<2)?1:0, thisCorner&(1<<1)?1:0, thisCorner&(1<<0)?1:0);
                int thisInd = vIndMap[thisVertInd3D];


                //if this vertec needs to be added, do it now!
                if (thisInd == -1){ 
                    vIndMap[thisVertInd3D] = thisInd = vertexCounter++;
                    for (int i=0; i<3; i++) vertices.push_back(0); //will be set on first updateMesh()
                }

                quads.push_back(thisInd); //add this vertices' contribution to the quad
            }
            //quadLinks.push_back(pV);
            quadVoxIndices.push_back(k);
        }
    }

    //vertex links: do here to make it the right size all at once and avoid lots of expensive allocations
    vertexLinks.resize(vertexCounter*8, NULL);
    for (int z=minZ; z<minZ+sizeZ+1; z++){ //for each in vIndMap, now.
        for (int y=minY; y<minY+sizeY+1; y++){
            for (int x=minX; x<minX+sizeX+1; x++){
                int thisInd = vIndMap[Index3D(x,y,z)];
                if (thisInd == -1) continue;

                //backwards links
                for (int i=0; i<8; i++){ //check all 8 possible voxels that could be connected...
                    CVX_Voxel* pV = Vx->voxel(x-(i&(1<<2)?1:0), y-(i&(1<<1)?1:0), z-(i&(1<<0)?1:0));
                    if (pV) vertexLinks[8*thisInd + i] = pV;
                }

                //lines
                for (int i=0; i<3; i++){ //look in positive x, y, and z directions
                    int isX = (i==0?1:0), isY = (i==1?1:0), isZ = (i==2?1:0);
                    int p2Ind = vIndMap[Index3D(x+isX, y+isY, z+isZ)];
                    if (p2Ind != -1){ //for x: voxel(x,y,z) (x,y-1,z) (x,y-1,z-1) (x,y,z-1) -- y: voxel(x,y,z) (x-1,y,z) (x-1,y,z-1) (x,y,z-1) -- z: voxel(x,y,z) (x,y-1,z) (x-1,y-1,z) (x-1,y,z)
                        if (Vx->voxel(x,            y,          z) ||
                            Vx->voxel(x-isY,        y-isX-isZ,  z) ||
                            Vx->voxel(x-isY-isZ,    y-isX-isZ,  z-isX-isY) ||
                            Vx->voxel(x-isZ,        y,          z-isX-isY)) {
                            
                            lines.push_back(thisInd); lines.push_back(p2Ind);
                        }
                    }
                }
            }
        }
    }

    //the rest... allocate space, but updateMesh will fill them in.
    int quadCount = quads.size()/4;

    quadColors.resize(quadCount*3);
    quadNormals.resize(quadCount*3);

    updateMesh();
}

//updates all the modal properties: offsets, quadColors, quadNormals.
void PV_MeshRender::updateMesh(viewColoring colorScheme, CVoxelyze::stateInfoType stateType)
{
    //location
    int vCount = vertices.size()/3;
    if (vCount == 0) return;
    for (int i=0; i<vCount; i++){ //for each vertex...
        Vec3D<float> avgPos;
        int avgCount = 0;
        for (int j=0; j<8; j++){
            CVX_Voxel* pV = vertexLinks[8*i+j];
            if (pV){
                avgPos += pV->cornerPosition((CVX_Voxel::voxelCorner)j);
                avgCount++;
            }
        }
        avgPos /= avgCount;
        vertices[3*i] = avgPos.x;
        vertices[3*i+1] = avgPos.y;
        vertices[3*i+2] = avgPos.z;
    }

    //Find a maximum if necessary:
    float minVal = 0, maxVal = 0;
    if (colorScheme == STATE_INFO){
        maxVal = Vx->stateInfo(stateType, CVoxelyze::MAX);
        minVal = Vx->stateInfo(stateType, CVoxelyze::MIN);
        if (stateType == CVoxelyze::PRESSURE){ //pressure max and min are equal pos/neg
            maxVal = maxVal>-minVal ? maxVal : -minVal;
            minVal = -maxVal;
        }

    }

    //color + normals (for now just pick three vertices, assuming it will be very close to flat...)
    int qCount = quads.size()/4;
    if (qCount == 0) return;
    for (int i=0; i<qCount; i++){
        Vec3D<float> v[4];
        for (int j=0; j<4; j++) v[j] = Vec3D<float>(vertices[3*quads[4*i+j]], vertices[3*quads[4*i+j]+1], vertices[3*quads[4*i+j]+2]);
        Vec3D<float> n = ((v[1]-v[0]).Cross(v[3]-v[0]));
        n.Normalize(); //necessary? try glEnable(GL_NORMALIZE)
        quadNormals[i*3] = n.x;
        quadNormals[i*3+1] = n.y;
        quadNormals[i*3+2] = n.z;

        float r=1.0f, g=1.0f, b=1.0f;
        float jetValue = -1.0f;
        switch (colorScheme){
            case MATERIAL:
                r = ((float)Vx->voxel(quadVoxIndices[i])->material()->red())/255.0f;
                g = ((float)Vx->voxel(quadVoxIndices[i])->material()->green())/255.0f;
                b = ((float)Vx->voxel(quadVoxIndices[i])->material()->blue())/255.0f;
                break;
            case FAILURE:
                if (Vx->voxel(quadVoxIndices[i])->isFailed()){g=0.0f; b=0.0f;}
                else if (Vx->voxel(quadVoxIndices[i])->isYielded()){b=0.0f;}
                break;
            case STATE_INFO:
                switch (stateType) {
                case CVoxelyze::KINETIC_ENERGY: jetValue = Vx->voxel(quadVoxIndices[i])->kineticEnergy()/maxVal; break;
                case CVoxelyze::STRAIN_ENERGY: case CVoxelyze::ENG_STRAIN: case CVoxelyze::ENG_STRESS: jetValue = linkMaxColorValue(Vx->voxel(quadVoxIndices[i]), stateType) / maxVal; break;
                case CVoxelyze::DISPLACEMENT: jetValue = Vx->voxel(quadVoxIndices[i])->displacementMagnitude()/maxVal; break;
                case CVoxelyze::PRESSURE: jetValue = 0.5-Vx->voxel(quadVoxIndices[i])->pressure()/(2*maxVal); break;
                default: jetValue = 0;
                }
            break;
        }

        if (jetValue != -1.0f){
            r = jetMapR(jetValue);
            g = jetMapG(jetValue);
            b = jetMapB(jetValue);
        }

        quadColors[i*3] = r;
        quadColors[i*3+1] = g;
        quadColors[i*3+2] = b;
    }
}

float PV_MeshRender::linkMaxColorValue(CVX_Voxel* pV, CVoxelyze::stateInfoType coloring)
{
    float voxMax = -FLT_MAX;
    for (int i=0; i<6; i++){
        float thisVal = -FLT_MAX;
        CVX_Link* pL = pV->link((CVX_Voxel::linkDirection)i);
        if (pL){
            switch (coloring){
                case CVoxelyze::STRAIN_ENERGY: thisVal = pL->strainEnergy(); break;
                case CVoxelyze::ENG_STRESS: thisVal = pL->axialStress(); break;
                case CVoxelyze::ENG_STRAIN: thisVal = pL->axialStrain(); break;
                default: thisVal=0;
            }
        }
        
        if(thisVal>voxMax) voxMax=thisVal;
    }
    return voxMax;
}

void PV_MeshRender::saveObj(const char* filePath)
{
    std::ofstream ofile(filePath);
    ofile << "# OBJ file generated by Voxelyze\n";
    for (int i=0; i<(int)(vertices.size()/3); i++){
        ofile << "v " << vertices[3*i] << " " << vertices[3*i+1] << " " << vertices[3*i+2] << "\n";
    }

    for (int i=0; i<(int)(quads.size()/4); i++){
        ofile << "f " << quads[4*i]+1 << " " << quads[4*i+1]+1 << " " << quads[4*i+2]+1 << " " << quads[4*i+3]+1 << "\n";
    }

    ofile.close();
}


void PV_MeshRender::rlDraw()
{
    
    //quads
    int qCount = quads.size()/4;
    for (int i=0; i<qCount; i++) {
        /*
        glNormal3d(quadNormals[i*3], quadNormals[i*3+1], quadNormals[i*3+2]);
        glColor3d(quadColors[i*3], quadColors[i*3+1], quadColors[i*3+2]);
        glLoadName(quadVoxIndices[i]); //to enable picking
        */
        Vector3 a = (Vector3){vertices[3*quads[4*i]],   vertices[3*quads[4*i]+1],   vertices[3*quads[4*i]+2]};
        Vector3 b = (Vector3){vertices[3*quads[4*i+1]], vertices[3*quads[4*i+1]+1], vertices[3*quads[4*i+1]+2]};
        Vector3 c = (Vector3){vertices[3*quads[4*i+2]], vertices[3*quads[4*i+2]+1], vertices[3*quads[4*i+2]+2]};
        Vector3 d = (Vector3){vertices[3*quads[4*i+3]], vertices[3*quads[4*i+3]+1], vertices[3*quads[4*i+3]+2]};
        a = Vector3Transform(a, tf);
        b = Vector3Transform(b, tf);
        c = Vector3Transform(c, tf);
        d = Vector3Transform(d, tf);

        // #define BEIGEALPHA      CLITERAL(Color){ 211, 176, 131, 255 }   // Beige

        int cr = quadColors[i*3]  *255;
        int cg = quadColors[i*3+1]*255;
        int cb = quadColors[i*3+2]*255;

        DrawTriangle3D(a, b, c, (Color){cr, cg, cb, 100});
        DrawTriangle3D(c, d, a, (Color){cr, cg, cb, 100});
    }
    

    // lines
    int lCount = lines.size()/2;
    for (int i=0; i<lCount; i++) {
        Vector3 a = (Vector3){vertices[3*lines[2*i]], vertices[3*lines[2*i]+1], vertices[3*lines[2*i]+2]};
        Vector3 b = (Vector3){vertices[3*lines[2*i+1]], vertices[3*lines[2*i+1]+1], vertices[3*lines[2*i+1]+2]};
        a = Vector3Transform(a, tf);
        b = Vector3Transform(b, tf);
        DrawLine3D(a, b, BLACK);
    }
}

void PV_MeshRender::renderRobot(std::vector<std::stringstream*> info, std::vector<std::string*> info2, PV_Sensor* sensor, Vec3D<double> robotCenter, Vec3D<double> trueCenter, std::string filename){

    updateMesh();
    updateCameraTarget();

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

    DrawGrid(500, 1.0f);

    for(int i=0; i<pneunet->chambers.size(); i++){
        pneunet->chambers[i]->draw(&tf);
    }
    rlDraw();

    Vec3D<double> target = sensor->target;
    Vec3D<double> targetHeading = sensor->targetHeading;
    Vec3D<double> targetOrigin = sensor->targetOrigin;

    Vec3D<double> startCenter = trueCenter - robotCenter; // bit hacky but it works

    // Draw the target line
    Vector3 a = (Vector3){targetOrigin.x+startCenter.x, targetOrigin.y+startCenter.y, 0.01};
    Vector3 b = (Vector3){target.x * 10+targetOrigin.x+startCenter.x, target.y * 10+targetOrigin.y+startCenter.y, 0.01};
    a = Vector3Transform(a, tf);
    b = Vector3Transform(b, tf);
    DrawLine3D(a, b, RED);

    // Draw the line representing the target direction sensor input
    a = (Vector3){trueCenter.x, trueCenter.y, startCenter.z};
    b = (Vector3){targetHeading.x / 3 + trueCenter.x, targetHeading.y / 3 + trueCenter.y, startCenter.z};
    a = Vector3Transform(a, tf);
    b = Vector3Transform(b, tf);
    DrawLine3D(a, b, RED);    

    EndMode3D();

    DrawFPS(10, 10);

    int y = 40;
    for(int i=0; i<info.size(); i++){
        // DrawText(info[i]->str().c_str(), 10, y, 20, BLACK);
        DrawTextEx(font, info[i]->str().c_str(), (Vector2){10, y}, 20, 0.0, BLACK);
        y += 30;
    }

    y = 10;
    for(int i=0; i<info2.size(); i++){
        // DrawText(info[i]->str().c_str(), 10, y, 20, BLACK);
        DrawTextEx(font, info2[i]->c_str(), (Vector2){screenWidth-200, y}, 20, 0.0, BLACK);
        y += 30;
    }


    /* Handle events */
    if(IsKeyPressed(KEY_LEFT)){
        sensor->steer(20, robotCenter);
    }
    if(IsKeyPressed(KEY_RIGHT)){
        sensor->steer(-20, robotCenter);
    }


    EndDrawing();

    /* Save a single picture */
    if(!filename.empty()){

        RenderTexture2D picTexture = LoadRenderTexture(1000, 1000); // square
        updateCameraTargetPicture();
        BeginTextureMode(picTexture);
        // TODO dedup
        ClearBackground(RAYWHITE);
        BeginMode3D(camera2);

        DrawGrid(500, 1.0f);

        for(int i=0; i<pneunet->chambers.size(); i++){
            pneunet->chambers[i]->draw(&tf);
        }
        rlDraw();


        Vec3D<double> target = sensor->target;
        Vec3D<double> targetHeading = sensor->targetHeading;
        Vec3D<double> targetOrigin = sensor->targetOrigin;

        Vec3D<double> startCenter = trueCenter - robotCenter; // bit hacky but it works

        // Draw the target line
        Vector3 a = (Vector3){targetOrigin.x+startCenter.x, targetOrigin.y+startCenter.y, 0.01};
        Vector3 b = (Vector3){target.x * 10+targetOrigin.x+startCenter.x, target.y * 10+targetOrigin.y+startCenter.y, 0.01};
        a = Vector3Transform(a, tf);
        b = Vector3Transform(b, tf);
        Vector3 a1 = (Vector3){a.x, a.y, a.z+1};
        Vector3 b1 = (Vector3){b.x, b.y, b.z+1};
        DrawTriangle3D(a, a1, b, RED);
        DrawTriangle3D(a1, b1, b, RED);
        // DrawLine3D(a, b, RED);

        // Draw the line representing the target direction sensor input
        a = (Vector3){trueCenter.x, trueCenter.y, startCenter.z};
        b = (Vector3){targetHeading.x / 3 + trueCenter.x, targetHeading.y / 3 + trueCenter.y, startCenter.z};
        a = Vector3Transform(a, tf);
        b = Vector3Transform(b, tf);
        DrawLine3D(a, b, RED);


        EndMode3D();
        /* no info text in picture mode */
        EndTextureMode();

        Image img = LoadImageFromTexture(picTexture.texture);
        ImageFlipVertical(&img);
        ExportImage(img, filename.c_str());
    }
    /* Pipe out as a video frame */
    if(pPipe){
        /* Trying a different way; taken from TakeScreenshot internal function, using rlgl.h */
        unsigned char *imgData = rlReadScreenPixels(screenWidth, screenHeight);
        Image img = { imgData, screenWidth, screenHeight, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };

        long lSize = screenWidth * screenHeight * 4;
        fwrite(img.data, 1, lSize, pPipe);
    }
}

void PV_MeshRender::closePipe(){
    if(pPipe){
        fflush(pPipe);
        fclose(pPipe);
    }
}


