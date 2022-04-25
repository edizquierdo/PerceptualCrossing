#pragma once

#include "CTRNN.h"

// The PerceptualCrosser class declaration

class PerceptualCrosser {
	public:
		// The constructor
		PerceptualCrosser(double direction, int networksize, int sensorydelay, int maxsteps)
		{
			Set(direction, networksize, sensorydelay, maxsteps);
		};
		// The destructor
		~PerceptualCrosser() {};

		// Accessors
		double Position(void) {return pos;};
		void SetPosition(double newpos) {pos = newpos;};
		void SetDirection(double newdir) {dir = newdir;};
		void SetSensorWeight(int to, double value) {sensorweights[to] = value;};

		// Control
        void Set(double direction, int networksize, int sensorydelay, int maxsteps);
		void Reset(double initpos);
		void Sense(double other, double shadow, double fixed);
		void Step(double StepSize);

		int size, delay, step;
		double pos, dir, gain, sensor, pastpos;
		TVector<double> sensorweights;
		TVector<int> sensorhist;
		CTRNN NervousSystem;
};
