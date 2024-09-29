#include "Voxelyze.h"
#include "pneuvox.h"
#include "PV_Chamber.h"
#include "PV_Valve.h"
#include "PV_Pneunet.h"
#include "VX_Voxel.h"
#include "VX_Link.h"
#include "Vec3D.h"
#include "Quat3D.h"
#include "PV_Controller.h"
#include "PV_Sensor.h"

#ifndef NOGUI
#include "PV_MeshRender.h"
#endif

#include "json.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <cstdio>
#include <sys/time.h>
#include <random>
#include <chrono>
#include <thread>


using json = nlohmann::json;


double loadVoxelSize(json data){
    double voxelsize = data["voxelsize"];
    // std::cout << "loaded voxel size " << voxelsize << "\n";
    return voxelsize;
}

#define DBG_OFFSETX 0
#define DBG_OFFSETY 0

void loadVoxels(json data, CVoxelyze* Vx){
    std::map<int, CVX_Material*> materials;

    int resx = data["bbox"][0];
    int resy = data["bbox"][1];
    int resz = data["bbox"][2];

    // Load structural voxels
    int nvox = 0;
    for(int x=0; x<resx; x++){
        for(int y=0; y<resy; y++){
            for(int z=0; z<resz; z++){
                int mat_i = data["voxels"][x][y][z];
                json jmat = data["materials"][std::to_string(mat_i)];
                if(jmat["type"] == "structural"){
                    // Check if material exists yet, otherwise make it
                    if(materials.find(mat_i) == materials.end()){
                        CVX_Material* newmat = Vx->addMaterial((float)jmat["stiffness"], jmat["density"]);
                        newmat->setStaticFriction( 0.9);
                        newmat->setKineticFriction(0.9);
                        // newmat->setInternalDamping(1.0); // was 0.1, default is 1.0 
                        newmat->setInternalDamping(0.1); // was 0.1, default is 1.0 
                        newmat->setGlobalDamping(0.001);
                        // newmat->setGlobalDamping(0.01);
                        // newmat->setCollisionDamping(1.0);
                        newmat->setCollisionDamping(0.1);
                        newmat->setColor(jmat["color"][0], jmat["color"][1], jmat["color"][2]);
                        materials[mat_i] = newmat;
                    }
                    Vx->setVoxel(materials[mat_i], x + DBG_OFFSETX, y + DBG_OFFSETY, z);
                    nvox++;
                }
            }
        }
    }
    // std::cout << "Total number of voxels: " << nvox << "\n";
}

PV_Chamber* chamberDebugDraw;

std::vector<PV_ValveToggle*> loadPneumatics(json data, CVoxelyze* Vx, PV_Pneunet* pneunet){
    std::vector<PV_ValveToggle*> valves;

    std::map<int, PV_Chamber*> chambers;
    json jpneu = data["pneumatics"];
    for (json::iterator it = jpneu.begin(); it != jpneu.end(); ++it) {
        /* Create the chamber based on the given geometry */
        json quads = it.value()["quads"];
        PV_Chamber* chamber = new PV_Chamber();
        chamberDebugDraw = chamber;
        chambers[std::stoi(it.key())] = chamber;
        for(int i=0; i<quads.size(); i++){
            json q = quads[i];
            chamber->addQuad(
                Vx->voxel((int)(q[0][0]) + DBG_OFFSETX, (int)q[0][1] + DBG_OFFSETY, q[0][2]),
                Vx->voxel((int)(q[1][0]) + DBG_OFFSETX, (int)q[1][1] + DBG_OFFSETY, q[1][2]),
                Vx->voxel((int)(q[2][0]) + DBG_OFFSETX, (int)q[2][1] + DBG_OFFSETY, q[2][2]),
                Vx->voxel((int)(q[3][0]) + DBG_OFFSETX, (int)q[3][1] + DBG_OFFSETY, q[3][2])
            );
        }
        chamber->tris.shrink_to_fit();
        chamber->initPressure();
        pneunet->addChamber(chamber);

        // Connect to inlet and outlet via valves.
        PV_Conduit* cInlet = pneunet->connectExternal(chamber, data["crosssection"], data["pressure"]);
        PV_Conduit* cOutlet = pneunet->connectExternal(chamber, data["crosssection"], 1.0);

        // We use two separately controlled valves for each chamber, 1 inlet and 1 outlet
        PV_Conduit* conduitIn[1] = {cInlet};
        PV_ValveToggle* valveIn = new PV_ValveToggle(conduitIn);
        pneunet->addValve(valveIn);
        valves.push_back(valveIn);

        PV_Conduit* conduitOut[1] = {cOutlet};
        PV_ValveToggle* valveOut = new PV_ValveToggle(conduitOut);
        pneunet->addValve(valveOut);
        valves.push_back(valveOut);

    }
    return valves;
}


Vec3D<double> centerOfGravity(CVoxelyze* Vx){
    const std::vector<CVX_Voxel*>* voxels = Vx->voxelList();
    Vec3D<double> sum = Vec3D<double>();
    int size = 0;
    for(int i=0; i<voxels->size(); i++){
        if((*voxels)[i]->external()->isFixedAll()){
            continue;
        }
        sum += (*voxels)[i]->displacement(); // Displacement from the start position so the centroid is 0 just after initialization
        size++;
    }
    return sum / size;
}

Vec3D<double> trueCenter(CVoxelyze* Vx){
    // Actual center of gravity as opposed to the other function which measures displacement from the start
    // position of the center of gravity
    const std::vector<CVX_Voxel*>* voxels = Vx->voxelList();
    Vec3D<double> sum = Vec3D<double>();
    int size = 0;
    for(int i=0; i<voxels->size(); i++){
        if((*voxels)[i]->external()->isFixedAll()){
            continue;
        }
        sum += (*voxels)[i]->position(); // Displacement from the start position so the centroid is 0 just after initialization
        size++;
    }
    return sum / size;
}


float farthestPoint(CVoxelyze* Vx){
    const std::vector<CVX_Voxel*>* voxels = Vx->voxelList();
    float maxX = 0;
    for(int i=0; i<voxels->size(); i++){
        if((*voxels)[i]->external()->isFixedAll()){
            continue;
        }
        float newMaxX = (*voxels)[i]->position().x; // Absolute position, contrary to the centerOfGravity function
        if(newMaxX > maxX){
            maxX = newMaxX;
        }
    }
    return maxX;
}

bool flipped(CVoxelyze* Vx){
    // Detect whether the robot has flipped over
    const std::vector<CVX_Voxel*>* voxels = Vx->voxelList();
    double Yangle = 0.0; // This is the primary "rolling forward" direction
    double Xangle = 0.0; // This one is just to prevent it from flipping sideways
    int size = 0;
    for(int i=0; i<voxels->size(); i++){
        if((*voxels)[i]->external()->isFixedAll()){
            continue;
        }
        Vec3D<double> angles = (*voxels)[i]->orientation().ToRotationVector();
        Xangle += angles.x;
        Yangle += angles.y;
        size++;
    }
    Xangle /= size;
    Yangle /= size;
    // double limit = 3.14159 / 6; // 30 degrees
    double limit = 3.14159 / 3; // 60 degrees
    // std::cout << Xangle << " " << Yangle << "\n";
    return (Yangle > limit) || (Yangle < -limit) || (Xangle > limit) || (Xangle < -limit);
}

double zAngle(CVoxelyze* Vx){
    // Rotation in ground plane, for use in directed locomotion sensor
    // in radians, heading left (from robot perspective) is positive, right is negative,
    // straight forward (in x direction) is 0.
    const std::vector<CVX_Voxel*>* voxels = Vx->voxelList();
    double Zangle = 0.0;
    int size = 0;
    for(int i=0; i<voxels->size(); i++){
        if((*voxels)[i]->external()->isFixedAll()){
            continue;
        }
        Vec3D<double> angles = (*voxels)[i]->orientation().ToRotationVector();
        Zangle += angles.z;
        size++;
    }
    Zangle /= size;
    return Zangle;
}

bool exploded(CVoxelyze* Vx){
    float maxStrain = 0.0;
    for(CVX_Link* link : *(Vx->linkList())){
        if(link->axialStrain() > maxStrain){
            maxStrain = link->axialStrain();
        }
    }

    // std::cout << maxStrain << "\n";

    // if(maxStrain > 0.6){
    if(maxStrain > 2){
        return 1;
    }
    return 0;
}

bool escaped(CVoxelyze* Vx){
    // Whether the bot has gone beyond the edges of the obstacles.
    // Only useful when there is terrain, but does not hurt in other cases either.
    // Should not be used with directional experiments though!
    Vec3D<double> centroid = centerOfGravity(Vx);
    return centroid.y < -20 * Vx->voxelSize() || centroid.y > 20*Vx->voxelSize();
}

void make_terrain(json data, CVoxelyze* Vx){
    if((!data.contains("terrain_type")) || (data["terrain_type"] == "flat")){
        return;
    } else if(data["terrain_type"] == "rows"){
        CVX_Material* mat = Vx->addMaterial(10000, 1000);
        mat->setStaticFriction( 0.9);
        mat->setKineticFriction(0.9);
        mat->setInternalDamping(0.1); // was 0.1, default is 1.0 
        mat->setGlobalDamping(0.001);
        // mat->setCollisionDamping(0.1);
        mat->setCollisionDamping(5.0);
        mat->setColor(0, 255, 0);

        // std::random_device rd;
        std::uniform_int_distribution<int> ud(5,20);
        std::mt19937 mt(data["terrain_seed"]);
        int x = (int)data["bbox"][0] + 10;
        for(int i = 0; i < 10; i++){
            for(int y=-20; y<25; y++){
                Vx->setVoxel(mat, x, y, 0)->external()->setFixedAll();
            }
            x += ud(mt);
        }
    }
}

void printCentroid(CVoxelyze* Vx){
    Vec3D<double> centroid = centerOfGravity(Vx);
    std::cout << centroid.x << " " << centroid.y << " " << centroid.z << "\n";
}

int main(int argc, char** argv){
    if(argc > 1){
        std::cout << "Loading Voxbot file: " << argv[1] << "\n";
        if(argc > 2){
            std::cout << "Recording to: " << argv[2] << "\n";
        }
    } else {
        std::cout << "Did not get file to load, exiting...\n";
        return 0;
    }

    /* Load json file with robot data */
    std::ifstream f(argv[1]);
    json data = json::parse(f);
    float voxsize = loadVoxelSize(data);

    CVoxelyze* Vx = new CVoxelyze(voxsize); //init with correct voxel size
    Vx->enableCollisions(true);
    Vx->enableFloor(true);
    Vx->setGravity();

    make_terrain(data, Vx);

    /***/

    /* The order is important: 1. load voxels, 2. compute timestep based on the voxels, 3. init pneunet using this timestep */
    loadVoxels(data, Vx);
    float recommendedTimeStep = Vx->recommendedTimeStep(); // In seconds
    std::cout << "timestep " << recommendedTimeStep << "\n";
    PV_Pneunet* pneunet = new PV_Pneunet(recommendedTimeStep);

    std::vector<PV_ValveToggle*> valves = loadPneumatics(data, Vx, pneunet);

    PV_Controller controller = PV_Controller(data);

    // Initialize sensor that will be used for the directed locomotion experiments
    // We don't have to initialize the sensor with a starting position because
    // centerOfGravity already gives the displacement from initial position
    double targetAngle = 0;
    if(data.contains("evaluate_angle")){
        targetAngle = data["evaluate_angle"];
    }
    PV_Sensor sensor = PV_Sensor(Vx->voxelSize(), targetAngle);

    /***/

    int stepsPerFrame = 1.0f/60.0f/recommendedTimeStep;

    //--------------------------------------------------------------------------------------

    // run simulation for 5 seconds (or however much is specified)
    int totalSteps = 60 * (
        data.contains("eval_seconds") ? (int)data["eval_seconds"] : 5
    );


#ifndef NOGUI
    PV_MeshRender* MeshRender;
    if(data.contains("video_filename")){
        MeshRender = new PV_MeshRender(Vx, pneunet, data["video_filename"]);
    } else {
        MeshRender = new PV_MeshRender(Vx, pneunet);
    }
#endif


    struct timeval t1, t2;
    double elapsedTime;

    // start timer
    gettimeofday(&t1, NULL);

    bool kill = false; // Set to true if this robot should get a fitness of 0

    // Time it takes to reach the first row, for additional fitness component
    // float secondsToRows = -1;

    // Main loop
    for(int step=0; step<totalSteps; step++){
#ifndef NOGUI
        if(WindowShouldClose()){break;} // Detect window close button or ESC key
#endif

        std::vector<float> sensors = {};
        if(data["directed_locomotion"]){
            sensors.push_back(
                sensor.target_angle(centerOfGravity(Vx), zAngle(Vx))
            );
        }
        if(data["control_type"] == "recurrent_closed"){
            for(int i=0; i<pneunet->chambers.size(); i++){
                float pres = pneunet->chambers[i]->pressure;
                // scale from (1.0, 1.1) (roughly) to (-1, 1)
                pres -= 1.0;
                pres *= 20.0;
                pres -= 1.0;
                sensors.push_back(pres);
            }
        }
        std::vector<float> ctrl = controller.evaluate(1.0/60.0, sensors);

        for(int i=0; i<std::min(valves.size(), ctrl.size()); i++){
            valves[i]->set(ctrl[i]);
        }

        if(flipped(Vx) || exploded(Vx)){
            kill = true;
            break;
        }

        // if(secondsToRows < 0 && farthestPoint(Vx) >= 0.19){
        //     secondsToRows = (float)step / 60.0f;
        // }

        if(!(step % 60)){
            sensor.add_waypoint(centerOfGravity(Vx));
        }

        for(int i=0; i<stepsPerFrame; i++){
        // for(int i=0; i<1; i++){
            pneunet->update();
            Vx->doTimeStep(recommendedTimeStep);
        }
        // printCentroid(Vx);


        // Draw
        //----------------------------------------------------------------------------------
#ifndef NOGUI
        std::vector<std::stringstream*> info;
        std::vector<std::string*> info2;
        if(data.contains("extra_info")){
            for (std::string infoline : data["extra_info"]) {
                std::string* infostr = new std::string();
                *infostr = infoline;
                info2.push_back(infostr);
            }
        }

        std::stringstream spressure;
        spressure << "Pressure: " << pneunet->pressure() << " bar";
        info.push_back(&spressure);

        std::stringstream sctrli;
        sctrli.setf(std::ios::fixed, std::ios::floatfield);
        sctrli.precision(2);
        sctrli << "Ctrl in: ";

        int n_in_actual;
        if(data["control_type"] == "recurrent_closed"){
            n_in_actual = pneunet->chambers.size() + (controller.enable_osc ? 2 : 0);
        }else{
            n_in_actual = 2;
        }
        if(data["directed_locomotion"]){
            n_in_actual++;
        }

        for(int i=0; i < n_in_actual; i++){
            if(controller.inputs[i] >= 0){
                sctrli << " ";
            }
            sctrli << controller.inputs[i] << " ";
        }

        sctrli << "\n";
        info.push_back(&sctrli);

        std::stringstream sctrlo;
        sctrlo.setf(std::ios::fixed, std::ios::floatfield);
        sctrlo.precision(2);
        sctrlo << "Ctrl out: ";
        for(int i=0; i<std::min(valves.size(), ctrl.size()); i+=2){
            sctrlo << "(" << ctrl[i] << ", " << ctrl[i+1] << ") ";
        }
        sctrlo << "\n";
        info.push_back(&sctrlo);

        std::stringstream simtime;
        simtime << "Simtime: " << step / 60 << "s";
        info.push_back(&simtime);
        if(data.contains("pictures_filename") && ! (step % 40)){
            // Timelapse configuration:
            // A 'step' is actually a frame of which there are 60 in a second.
            // 15 pictures in 1 sec
            if(step >= 600){
                break;
            }
            std::stringstream name;
            name << "screenshots/" << data["pictures_filename"].get<std::string>() << "_frame" << std::setfill('0') << std::setw(4) << step << std::setw(0) << ".png";
            MeshRender->renderRobot(info, info2, &sensor, centerOfGravity(Vx), trueCenter(Vx), name.str());
        } else {
            MeshRender->renderRobot(info, info2, &sensor, centerOfGravity(Vx), trueCenter(Vx));
        }
#endif
        //----------------------------------------------------------------------------------
    }
    // Another one for the final position
    sensor.add_waypoint(centerOfGravity(Vx));

    // stop timer
    gettimeofday(&t2, NULL);

    // compute and print the elapsed time in millisec
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    std::cout << "Elapsed time: " << elapsedTime << " ms\n";

    // De-Initialization
    //--------------------------------------------------------------------------------------
#ifndef NOGUI
    CloseWindow();        // Close window and OpenGL context
    MeshRender->closePipe();
#endif
    //--------------------------------------------------------------------------------------
    if(kill){
        if(data["directed_locomotion"]){
            std::cout << "Distance: 0.0" << std::endl; // distance traveled in cm
        }
        std::cout << "Fitness: 0.0" << std::endl; // distance traveled in cm
        // std::cout << "SecondsToRows: -1.0" << std::endl; // additional fitness goal if obstacles are used
        std::cout << "Killed: 1" << std::endl; // It was killed from explosion or tipping over
    }else{
        Vec3D<double> centroid = centerOfGravity(Vx);
        if(data["directed_locomotion"]){
            sensor.print_fitness();
        }else{
            std::cout << "Fitness: " << centroid.x * 100 << std::endl; // distance traveled in cm
        }
        // std::cout << "SecondsToRows: " << secondsToRows << std::endl; // additional fitness goal if obstacles are used
        std::cout << "Killed: 0" << std::endl;
        // Use std::endl because it flushes!
    }

    return 0;
}
