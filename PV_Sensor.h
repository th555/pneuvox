#ifndef PV_SENSOR_H
#define PV_SENSOR_H

#include "Vec3D.h"

class PV_Sensor
{
public:
	PV_Sensor(double voxelSize, double targetAngle); // targetAngle in degrees
	void add_waypoint(Vec3D<double> point);
	double print_fitness();
	double target_angle(Vec3D<double> position, double heading);
	void steer(double steerAngle, Vec3D<double> newOrigin); // for interactive steering

	double voxelSize;
	double currentTargetAngle; // in degrees
	Vec3D<double> target; // Normalized vector pointing in the target direction
	Vec3D<double> targetOrigin; // Always 0,0,0 in evolution experiments, can be other values in interactive mode
	std::vector<Vec3D<double>> waypoints; // Storage for trajectory points
	Vec3D<double> targetHeading; // Only for drawing
};



#endif //PV_SENSOR_H