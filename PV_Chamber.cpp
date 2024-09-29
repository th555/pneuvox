#include "PV_Chamber.h"
#include "VX_Voxel.h"
#include "PV_Conduit.h"

#include <algorithm>
#include <iostream>

#ifndef NOGUI
// Only for draw()
#include "raylib.h"
#include "raymath.h"
#endif

PV_Chamber::PV_Chamber(){
}

void PV_Chamber::initPressure(){
    updateVolume();
    quantity = volume;
}

void PV_Chamber::addQuad(CVX_Voxel* a, CVX_Voxel* b, CVX_Voxel* c, CVX_Voxel* d){
    if(!(a&&b&&c&&d)){
        std::cerr << "ERROR NULL VOXEL IN QUAD\n";
        return;
        exit(1);
    }
    tris.push_back(a);
    tris.push_back(b);
    tris.push_back(c);

    tris.push_back(c);
    tris.push_back(d);
    tris.push_back(a);    

    CVX_Voxel* quad[4] = {a, b, c, d};
    for (int i=0; i<4; i++){
        if(std::find(uniqueVoxels.begin(), uniqueVoxels.end(), quad[i]) == uniqueVoxels.end()){
            uniqueVoxels.push_back(quad[i]);
        }
    }
}

int PV_Chamber::nTris(){
    return tris.size()/3;
}

void PV_Chamber::preUpdate(){
    for (std::size_t i=0; i<uniqueVoxels.size(); i++){
        uniqueVoxels[i]->external()->clearForce();
    }
    updateVolume();
}

void PV_Chamber::update(){
    updatePressure();
    for (int i=0; i<nTris(); i++){
        applyPneuForce(pressure, i);
    }
}

float PV_Chamber::triArea(int i){
    Vec3D<double> a = tris[i*3  ]->position();
    Vec3D<double> b = tris[i*3+1]->position();
    Vec3D<double> c = tris[i*3+2]->position();
    // Using half cross product formula
    return (c - b).Cross(a - b).Length() * 0.5;
}

float PV_Chamber::area(){
    float totArea = 0;
    for (int i=0; i<nTris(); i++){
        totArea += triArea(i);
    }
    return totArea;
}

void PV_Chamber::applyTriForce(Vec3D<double> force, int tri_i){
    tris[tri_i*3+0]->external()->addForce((Vec3D<float>)force/3.0);
    tris[tri_i*3+1]->external()->addForce((Vec3D<float>)force/3.0);
    tris[tri_i*3+2]->external()->addForce((Vec3D<float>)force/3.0);
}

void PV_Chamber::applyPneuForce(float pressure, int tri_i){
    Vec3D<double> t[3];
    for (int i=0; i<3; i++){ //for each corner of the (exposed) face in this direction
        t[i] = tris[tri_i*3+i]->position();
    }

    /* Find the surface normal and area, and apply force per tri (since a quad might not be perfectly flat) */

    Vec3D<double> n = ((t[2]-t[1]).Cross(t[0]-t[1]));
    n.Normalize();
    Vec3D<double> force = -n * (pressure - 1) * triArea(tri_i) * 100000.0f;
    applyTriForce(force, tri_i);
}


void PV_Chamber::updatePressure(){
    /* apply ideal gas law:
    p = pressure
    V = volume
    n = quantity of substance
    R = ideal gas constant
    T = temperature
    
    pV = nRT

    p = nRT/V
    pV/n = constant

    instead of the usual unit we measure n as "volume it would have at 1 bar"
    This allows us to do

    (p0 * V0)/n0 = (p1 * V1)/n1
    with p0 = 1 bar, V0 = n0 = the original quantity i.e. volume
    and p1 = current pressure, V1 = current volume, n1 = current quantity
    So we rewrite and compute p1 as
    p1 = V0/V1 * n1/n0 = V0/V1 * n1/V0 = n1/V1
    */
    pressure = quantity/volume;
}


/*
Volume calculation by adding signed tetrahedron volumes as per https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
*/
void PV_Chamber::updateVolume(){
    float newVolume = 0;

    // Use a point on the mesh as the basis for volume calculations (instead of the origin) to prevent
    // numerical instability when it gets farther away from the origin.
    Vec3D<double> p0 = (Vec3D<double>)tris[0]->position();

    for (int tri_i=0; tri_i<nTris(); tri_i++){
        Vec3D<double> p1 = (Vec3D<double>)tris[tri_i*3+2]->position() - p0;
        Vec3D<double> p2 = (Vec3D<double>)tris[tri_i*3+1]->position() - p0;
        Vec3D<double> p3 = (Vec3D<double>)tris[tri_i*3]->position() - p0;
        newVolume += p1.Dot(p2.Cross(p3)) / 6.0f;
    }

    volume = newVolume > 1e-12 ? newVolume : 1e-12;
}

#ifndef NOGUI
void PV_Chamber::draw(void* tf)
{
    Matrix* tfp = (Matrix*)tf;
    for (int tri_i=0; tri_i<nTris(); tri_i++){
        Vec3D<double> p1 = (Vec3D<double>)tris[tri_i*3+2]->position();
        Vec3D<double> p2 = (Vec3D<double>)tris[tri_i*3+1]->position();
        Vec3D<double> p3 = (Vec3D<double>)tris[tri_i*3]->position();
        Vector3 a = (Vector3){p1.x, p1.y, p1.z};
        Vector3 b = (Vector3){p2.x, p2.y, p2.z};
        Vector3 c = (Vector3){p3.x, p3.y, p3.z};
        a = Vector3Transform(a, *tfp);
        b = Vector3Transform(b, *tfp);
        c = Vector3Transform(c, *tfp);

        DrawTriangle3D(a,b,c, (Color){0, 0, 0, 150});

        // Draw normals
        /*
        Vec3D<double> t[3];
        for (int i=0; i<3; i++){ //for each corner of the (exposed) face in this direction
            t[i] = tris[tri_i*3+i]->position();
        }
        Vec3D<double> n = ((t[2]-t[1]).Cross(t[0]-t[1]));
        n.Normalize();

        Vec3D<double> pn1 = p1 + n/100.0f;
        Vector3 n1 = (Vector3){pn1.x, pn1.y, pn1.z};
        n1 = Vector3Transform(n1, *tfp);
        DrawLine3D(a, n1, BLACK);
        */
    }
}
#endif

/*
public float SignedVolumeOfTriangle(Vector p1, Vector p2, Vector p3) {
    var v321 = p3.X*p2.Y*p1.Z;
    var v231 = p2.X*p3.Y*p1.Z;
    var v312 = p3.X*p1.Y*p2.Z;
    var v132 = p1.X*p3.Y*p2.Z;
    var v213 = p2.X*p1.Y*p3.Z;
    var v123 = p1.X*p2.Y*p3.Z;
    return (1.0f/6.0f)*(-v321 + v231 + v312 - v132 - v213 + v123);
}

public float VolumeOfMesh(Mesh mesh) {
    var vols = from t in mesh.Triangles
               select SignedVolumeOfTriangle(t.P1, t.P2, t.P3);
    return Math.Abs(vols.Sum());
}

"topped off at the origin" is not mandatory, you may choose any fixed point. And if the object is quite far from the origin, this will lead to numerical instability. Better choose an arbitrary point from the mesh

Met vectors kan dit ook (Schijnt, testen):
public static float SignedVolumeOfTriangle(Vector p1, Vector p2, Vector p3)
{
    return p1.Dot(p2.Cross(p3)) / 6.0f;
}

*/
