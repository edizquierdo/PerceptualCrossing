#include "PerceptualCrosser.h"
#include "random.h"

// Constants
const double SpaceSize = 600;
const double HalfSpace = 300;
const double SenseRange = 2.0; //1.0; //0.5; //2.0

// *******
// Control
// *******

// Init the agent
void PerceptualCrosser::Set(double direction, int networksize, int sensorydelay, int maxsteps)
{
  step = 1 + sensorydelay;
	size = networksize;
	gain = 2.0; //4.0; //2.0; //1.0;
	sensorweights.SetBounds(1, size);
	sensorweights.FillContents(0.0);
	delay = sensorydelay;
	sensorhist.SetBounds(1, maxsteps);
	sensorhist.FillContents(0.0);
	pos = 0.0;
	pastpos = 0.0;
	dir = direction;
	sensor = 0.0;
}

// Reset the state of the agent
void PerceptualCrosser::Reset(double initpos)
{
  step = 1 + delay;
	pos = initpos;
	pastpos = initpos;
	sensor = 0.0;
	sensorhist.FillContents(0.0);
	NervousSystem.RandomizeCircuitState(0.0,0.0);
}

// Sense
void PerceptualCrosser::Sense(double Other, double Shadow, double Fixed)
{
	// Sense
	sensor = 0.0;
	double dist = 0.0;

	// Sense other agent
	dist = fabs(Other - pos);
	if (dist > HalfSpace)
		dist =  SpaceSize - dist;
	if (dist < SenseRange)
		sensor = 1.0;

	// Sense other agent
	dist = fabs(Shadow - pos);
	if (dist > HalfSpace)
		dist =  SpaceSize - dist;
	if (dist < SenseRange)
		sensor = 1.0;

	// Sense other agent
	dist = fabs(Fixed - pos);
	if (dist > HalfSpace)
		dist =  SpaceSize - dist;
	if (dist < SenseRange)
		sensor = 1.0;

	sensorhist(step) = sensor;
}

// Step
void PerceptualCrosser::Step(double StepSize)
{
    // Remember past position
    pastpos = pos;

	// Set sensor to external input
	for (int i = 1; i <= size; i++)
		NervousSystem.SetNeuronExternalInput(i, sensorhist(step-delay)*sensorweights[i]);

	// Update the nervous system
	NervousSystem.EulerStep(StepSize);

	// Update the body position
	pos += StepSize * dir * gain *  (NervousSystem.NeuronOutput(2) - NervousSystem.NeuronOutput(1));

	// Wrap-around Environment
	if (pos >= SpaceSize)
		pos = pos - SpaceSize;
	if (pos < 0.0)
		pos = SpaceSize + pos;

	// Increase step
	step += 1;
}
