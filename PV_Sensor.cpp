#include <vector>
#include <iostream>
#include <math.h>

#include "PV_Sensor.h"
#include "Vec3D.h"


void prtvec(Vec3D<double> vec){
    std::cout << vec.x << ", " << vec.y << ", " << vec.z << "\n";
}


/* The classes in this file will take care of computing the fitness function
based on the trajectory, as well as csalculating the sensor input which should
point to a point projected on the trajectory some distance away from the robot. */

PV_Sensor::PV_Sensor(double voxelSize, double targetAngle){
    this->voxelSize = voxelSize;
    currentTargetAngle = targetAngle;
    double radians = currentTargetAngle * (M_PI / 180);
    target = Vec3D<double>(cos(radians), sin(radians), 0);
    targetOrigin = Vec3D<double>(0, 0, 0);
    target.Normalize(); // Ensure it is a unit vector!!
}

void PV_Sensor::steer(double steerAngle, Vec3D<double> newOrigin){
    currentTargetAngle += steerAngle;
    double radians = currentTargetAngle * (M_PI / 180);
    target = Vec3D<double>(cos(radians), sin(radians), 0);
    targetOrigin = newOrigin;
    targetOrigin.z = 0;
}

void PV_Sensor::add_waypoint(Vec3D<double> point){
    point.x *= 100; // Conversion to cm
    point.y *= 100;
    point.z = 0;
    waypoints.push_back(point);
}

double PV_Sensor::print_fitness(){
    /* Fitness calculation from lan2021learning (eq 9):
    F = abs(D(p, p0)) / (L+e) * (D(p,p0) / (d(B0,B1) + 1) - P(p,p1))
    
    p is the projection of the end point on the target line
    p0 is the starting position
    D(p, p0) is the distance between these two points (signed so that it is negative when
        it moves in the wrong direction)
    L is the length of the trajectory formed by the waypoints
    e is an infinitesimal constant
    d(B0,B1) is the (smallest) intersection angle between the target line and the
        actually traversed line
    P(p,p1) is the sum of the magnitude of the vector rejections of each waypoint on the
        target line, times a penalty factor of 0.01
    */
    /* WARNING: assumes targetOrigin = 0,0,0 (since we don't need fitness in interactive mode)*/
    
    Vec3D<double> p0 = waypoints[0];
    Vec3D<double> p1 = waypoints[waypoints.size()-1];

    /* Distance traveled in target direction (Dpp0) */
    double Dpp0 = target.Dot(p1);

    /* Length of the trajectory (L)*/
    double L = 0;
    for(int i=0; i<waypoints.size()-1; i++){
        L += (waypoints[i+1] - waypoints[i]).Length();
    }
    double e = 0.00001;

    /* Deviation from target direction (dB0B1)*/
    Vec3D<double> B1 = (p1 - p0).Normalized();
    Vec3D<double> rotax = Vec3D<double>();
    double dB0B1 = B1.AlignWith(target, rotax);

    /* Penalty: deviation of trajectory at waypoints (Ppp1) */
    double Ppp1 = 0;
    for(int i=0; i<waypoints.size(); i++){
        Vec3D<double> wp = waypoints[i];
        Vec3D<double> proj = target.Dot(wp) * target;
        Vec3D<double> rej = wp - proj;
        Ppp1 += rej.Length();
    }
    Ppp1 *= 0.1;

    double fitness = abs(Dpp0) / (L + e) * (Dpp0 / (dB0B1 + 1) - Ppp1);
    std::cout << "Distance: " << target.Dot(p1) << std::endl;
    std::cout << "Fitness: " << fitness << std::endl;
    std::cout << "var_Dpp0: " << Dpp0 << std::endl;
    std::cout << "var_L: " << L << std::endl;
    std::cout << "var_dB0B1: " << dB0B1 << std::endl;
    std::cout << "var_Ppp1: " << Ppp1 << std::endl;
    return fitness;
}

double PV_Sensor::target_angle(Vec3D<double> position, double heading){
    // Angle to the target position, relative to the current heading of the robot
    // Target position is the robot position projected onto the target trajectory line,
    // from there 2 body lengths further along the target trajectory.
    position -= targetOrigin;
    Vec3D<double> projTarget = (target.Dot(position) + 20 * voxelSize) * target;
    Vec3D<double> robotHeading = Vec3D<double>(1, 0, 0).Rot(Vec3D<double>(0, 0, 1), heading);
    /* Unit vector pointing from robot to target */
    targetHeading = projTarget - position;
    targetHeading.z = 0; // force in xy plane
    targetHeading.Normalize();

    Vec3D<double> rotax = Vec3D<double>();
    double targetAngle = robotHeading.AlignWith(targetHeading, rotax);
    targetAngle *= rotax.z;

    /*
    std::cout << "Position: ";
    prtvec(position);
    std::cout << "Projected position: ";
    prtvec(proj);
    std::cout << "Target position: ";
    prtvec(projTarget);
    std::cout << "Robot heading: ";
    prtvec(robotHeading);
    std::cout << "Target heading: ";
    prtvec(targetHeading);
    std::cout << "heading angle: " << heading << "\n";
    std::cout << "target angle?: " << targetAngle << "\n";
    std::cout << "\n";
    */
    return targetAngle;
}
