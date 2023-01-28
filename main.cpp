#include "TSearch.h"
#include "PerceptualCrosser.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double StepSize = 0.01; 
const double RunDuration = 800.0; 
const double TransDuration = 400.0; 

const double RunDurationMap = 1600.0; 
const double TransDurationMap = 1500.0;

const double Fixed1 = 150.0;
const double Fixed2 = 450.0;
const double Shadow = 48.0; 
const double SpaceSize = 600.0;
const double HalfSpace = 300;
const double SenseRange = 2.0; 
const double CloseEnoughRange = 2.0; 
const int STEPPOS1 = 50;
// const int STEPPOS2 = 25;

// EA params
const int POPSIZE = 96;
const int GENS = 1000;
const double MUTVAR = 0.05;			// ~ 1/VectSize for N=3
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 3; 
const double WR = 8.0;
const double SR = 8.0;
const double BR = 8.0;
const double TMIN = 1.0;
const double TMAX = 10.0;

int	VectSize = N*N + 2*N + N;

// ================================================
// A. FUNCTIONS FOR EVOLVING A SUCCESFUL CIRCUIT
// ================================================

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -SR, SR);
		k++;
	}
}

// ------------------------------------
// Fitness function 1: 50, 0.1 
// ------------------------------------
double FitnessFunction1(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
	double totaltrials = 0, totaltime = 0.0;
	double shadow1, shadow2;

	for (int fixedFlag = 0; fixedFlag <= 1; fixedFlag += 1){
		for (int shadowFlag = 0; shadowFlag <= 1; shadowFlag += 1){
			if ((fixedFlag != 0) or (shadowFlag != 0)){
				for (double x1 = 0.0; x1 < SpaceSize; x1 += STEPPOS1){
					for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += STEPPOS1) {

						// Set agents positions
						Agent1.Reset(x1);
						Agent2.Reset(x2);

						// Run the sim
						totaldist = 0.0;
						totaltime = 0;

						for (double time = 0; time < RunDuration; time += StepSize)
						{
							// Update shadow positions
							shadow1 = Agent1.pos + Shadow;
							if (shadow1 >= SpaceSize)
								shadow1 = shadow1 - SpaceSize;
							if (shadow1 < 0.0)
								shadow1 = SpaceSize + shadow1;

							// Notice the other shadow is a reflection (not a rotation, i.e., shadow2 = Agent2.pos - Shadow;).
							shadow2 = Agent2.pos + Shadow;
							if (shadow2 >= SpaceSize)
								shadow2 = shadow2 - SpaceSize;
							if (shadow2 < 0.0)
								shadow2 = SpaceSize + shadow2;

							// Sense
							if (shadowFlag == 0)
							{
								if (fixedFlag == 0){
									cout << "Errorrr" << endl;
								}
								else{
									Agent1.Sense(Agent2.pos, 999999999, Fixed2);
									Agent2.Sense(Agent1.pos, 999999999, Fixed1);
								}
							}
							else 
							{
								if (fixedFlag == 0){
									Agent1.Sense(Agent2.pos, shadow2, 999999999);
									Agent2.Sense(Agent1.pos, shadow1, 999999999);
								}
								else{
									Agent1.Sense(Agent2.pos, shadow2, Fixed2);
									Agent2.Sense(Agent1.pos, shadow1, Fixed1);
								}
							}

							// Move
							Agent1.Step(StepSize);
							Agent2.Step(StepSize);

							// Measure distance between them (after transients)
							if (time > TransDuration)
							{
								dist = fabs(Agent2.pos - Agent1.pos);
								if (dist > HalfSpace)
									dist =  SpaceSize - dist;
								if (dist < CloseEnoughRange)
									dist = CloseEnoughRange;
								totaldist += dist;
								totaltime += 1;
							}
						}
						totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
						totaltrials += 1;
					}
				}
			}
		}
	}
	return totalfit/totaltrials;
}

// ------------------------------------
// Fitness function 3: 25, 0.1 Cross
// ------------------------------------
double FitnessFunction2(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
	double totaltrials = 0, totaltime = 0.0;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;

	for (int fixedFlag = 0; fixedFlag <= 1; fixedFlag += 1){
		for (int shadowFlag = 0; shadowFlag <= 1; shadowFlag += 1){
			if ((fixedFlag != 0) or (shadowFlag != 0)){
				for (double x1 = 0.0; x1 < SpaceSize; x1 += STEPPOS1){
					for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += STEPPOS1) {

						// Set agents positions
						Agent1.Reset(x1);
						Agent2.Reset(x2);

						// Run the sim
						totaldist = 0.0;
						totaltime = 0;
						crosscounter = 0;

						for (double time = 0; time < RunDuration; time += StepSize)
						{
							// Update shadow positions
							shadow1 = Agent1.pos + Shadow;
							if (shadow1 >= SpaceSize)
								shadow1 = shadow1 - SpaceSize;
							if (shadow1 < 0.0)
								shadow1 = SpaceSize + shadow1;

							// shadow2 = Agent2.pos - Shadow;
							shadow2 = Agent2.pos + Shadow;
							if (shadow2 >= SpaceSize)
								shadow2 = shadow2 - SpaceSize;
							if (shadow2 < 0.0)
								shadow2 = SpaceSize + shadow2;

							// Sense
							if (shadowFlag == 0)
							{
								if (fixedFlag == 0){
									cout << "Errorrr" << endl;
								}
								else{
									Agent1.Sense(Agent2.pos, 999999999, Fixed2);
									Agent2.Sense(Agent1.pos, 999999999, Fixed1);
								}
							}
							else 
							{
								if (fixedFlag == 0){
									Agent1.Sense(Agent2.pos, shadow2, 999999999);
									Agent2.Sense(Agent1.pos, shadow1, 999999999);
								}
								else{
									Agent1.Sense(Agent2.pos, shadow2, Fixed2);
									Agent2.Sense(Agent1.pos, shadow1, Fixed1);
								}
							}

							// Move
							Agent1.Step(StepSize);
							Agent2.Step(StepSize);

							// Measure distance between them (after transients)
							if (time > TransDuration)
							{
								dist = fabs(Agent2.pos - Agent1.pos);
								if (dist > HalfSpace)
									dist =  SpaceSize - dist;
								if (dist < CloseEnoughRange)
									dist = CloseEnoughRange;
								totaldist += dist;
								totaltime += 1;

								// Measure number of times the agents cross paths
								if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
								{
									crosscounter += 1;
								}
							}
						}

						totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
						totaltrials += 1;
						totalcross += crosscounter/TransDuration;
					}
				}	
			}
		}
	}
	if (totalfit/totaltrials > 0.99){
		return 1.0 + (totalcross/totaltrials);
	}
	else{
		return totalfit/totaltrials;
	}
}

// ================================================
// B. FUNCTIONS FOR ANALYZING A SUCCESFUL CIRCUIT
// ================================================

// ------------------------------------
// 0. Behavioral Traces
// ------------------------------------
void BehavioralTracesRegular(TVector<double> &genotype)
{
	// Start output file
	ofstream a1file("a1_pos_regular.dat");
	ofstream a2file("a2_pos_regular.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}
	cout << Agent1.sensorweights << endl;

	double shadow1, shadow2;

	for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			for (double time = 0; time < RunDuration; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Save
				a1file << Agent1.pos << " ";
				a2file << Agent2.pos << " ";
			}
			a1file << endl;
			a2file << endl;
		}
	}
	a1file.close();
	a2file.close();
}
void BehavioralTracesNoShadow(TVector<double> &genotype)
{
	// Start output file
	ofstream a1file("a1_pos_noshadow.dat");
	ofstream a2file("a2_pos_noshadow.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}
	cout << Agent1.sensorweights << endl;

	double shadow1, shadow2;

	for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			for (double time = 0; time < RunDuration; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, 999999999, Fixed2);
				Agent2.Sense(Agent1.pos, 999999999, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Save
				a1file << Agent1.pos << " ";
				a2file << Agent2.pos << " ";
			}
			a1file << endl;
			a2file << endl;
		}
	}
	a1file.close();
	a2file.close();
}
void BehavioralTracesNoFixed(TVector<double> &genotype)
{
	// Start output file
	ofstream a1file("a1_pos_nofixed.dat");
	ofstream a2file("a2_pos_nofixed.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}
	cout << Agent1.sensorweights << endl;

	double shadow1, shadow2;

	for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			for (double time = 0; time < RunDuration; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, 999999999);
				Agent2.Sense(Agent1.pos, shadow1, 999999999);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Save
				a1file << Agent1.pos << " ";
				a2file << Agent2.pos << " ";
			}
			a1file << endl;
			a2file << endl;
		}
	}
	a1file.close();
	a2file.close();
}

// ------------------------------------
// 1. Neural traces
// ------------------------------------
void NeuralTracesRegular(TVector<double> &genotype)
{
	// Start output file
	ofstream afile("nt_a_onlyother.dat");
	ofstream sfile("nt_s_onlyother.dat");
	ofstream n1file("nt_x1_onlyother.dat");
	ofstream n2file("nt_x2_onlyother.dat");

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;
	double x1 = 0.0; //50.0;
	double x2 = 200.0; //200.0; //350.0;
	double pos = 100.28;

	// Set agents positions
	Agent1.Reset(x1);
	Agent2.Reset(x2);

	// Run the sim
	for (double time = 0; time < RunDuration; time += StepSize)
	{
		pos = 100.28 + 0.3 * sin((0.28*time)+1.3);

		// Update shadow positions
		shadow1 = Agent1.pos + Shadow;
		if (shadow1 >= SpaceSize)
			shadow1 = shadow1 - SpaceSize;
		if (shadow1 < 0.0)
			shadow1 = SpaceSize + shadow1;

		// XXX shadow2 = Agent2.pos - Shadow;
		shadow2 = Agent2.pos + Shadow;
		if (shadow2 >= SpaceSize)
			shadow2 = shadow2 - SpaceSize;
		if (shadow2 < 0.0)
			shadow2 = SpaceSize + shadow2;

		// Sense
		Agent1.Sense(Agent2.pos, 999999999, 999999999); 
		Agent2.Sense(Agent1.pos, 999999999, 999999999); 			
		// Agent1.Sense(999999999, 999999999, Fixed2); //shadow2, Fixed2);
		// Agent2.Sense(999999999, 999999999, Fixed1); //shadow1, Fixed1);		
		// Agent1.Sense(Agent2.pos, 999999999, 999999999); //shadow2, Fixed2);
		// Agent2.Sense(Agent1.pos, 999999999, 999999999); //shadow1, Fixed1);

		// Move
		Agent1.Step(StepSize);
		Agent2.Step(StepSize);

		// Save
		afile << Agent1.pos << " " << Agent2.pos << " " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << " " << pos << endl;
		sfile << Agent1.sensor << " " << Agent2.sensor << " " << endl;

		n1file << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
		n2file << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.sensor << endl;

		// n1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
		// n2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
	}
	cout << Agent1.sensorweights << endl;
	afile.close();
	sfile.close();
	n1file.close();
	n2file.close();
}
void NeuralTracesNoShadow(TVector<double> &genotype)
{
	// Start output file
	ofstream afile("nt_a_noshadow.dat");
	ofstream sfile("nt_s_noshadow.dat");
	ofstream n1file("nt_n1_noshadow.dat");
	ofstream n2file("nt_n2_noshadow.dat");

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;
	double x1 = 0.0; //50.0;
	double x2 = 200.0; //350.0;

	// Set agents positions
	Agent1.Reset(x1);
	Agent2.Reset(x2);

	// Run the sim
	for (double time = 0; time < RunDuration; time += StepSize)
	{
		// Update shadow positions
		shadow1 = Agent1.pos + Shadow;
		if (shadow1 >= SpaceSize)
			shadow1 = shadow1 - SpaceSize;
		if (shadow1 < 0.0)
			shadow1 = SpaceSize + shadow1;

		// XXX shadow2 = Agent2.pos - Shadow;
		shadow2 = Agent2.pos + Shadow;
		if (shadow2 >= SpaceSize)
			shadow2 = shadow2 - SpaceSize;
		if (shadow2 < 0.0)
			shadow2 = SpaceSize + shadow2;

		// Sense
		Agent1.Sense(Agent2.pos, 999999999, Fixed2);
		Agent2.Sense(Agent1.pos, 999999999, Fixed1);

		// Move
		Agent1.Step(StepSize);
		Agent2.Step(StepSize);

		// Save
		afile << Agent1.pos << " " << Agent2.pos << " " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << endl;
		sfile << Agent1.sensor << " " << Agent2.sensor << " " << endl;
		n1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
		n2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
	}
	cout << Agent1.sensorweights << endl;
	afile.close();
	sfile.close();
	n1file.close();
	n2file.close();
}
void NeuralTracesNoFixed(TVector<double> &genotype)
{
	// Start output file
	ofstream afile("nt_a_nofixed.dat");
	ofstream sfile("nt_s_nofixed.dat");
	ofstream n1file("nt_n1_nofixed.dat");
	ofstream n2file("nt_n2_nofixed.dat");

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;
	double x1 = 0.0; //50.0;
	double x2 = 200.0; //350.0;

	// Set agents positions
	Agent1.Reset(x1);
	Agent2.Reset(x2);

	// Run the sim
	for (double time = 0; time < RunDuration; time += StepSize)
	{
		// Update shadow positions
		shadow1 = Agent1.pos + Shadow;
		if (shadow1 >= SpaceSize)
			shadow1 = shadow1 - SpaceSize;
		if (shadow1 < 0.0)
			shadow1 = SpaceSize + shadow1;

		// XXX shadow2 = Agent2.pos - Shadow;
		shadow2 = Agent2.pos + Shadow;
		if (shadow2 >= SpaceSize)
			shadow2 = shadow2 - SpaceSize;
		if (shadow2 < 0.0)
			shadow2 = SpaceSize + shadow2;

		// Sense
		Agent1.Sense(Agent2.pos, shadow2, 999999999);
		Agent2.Sense(Agent1.pos, shadow1, 999999999);

		// Move
		Agent1.Step(StepSize);
		Agent2.Step(StepSize);

		// Save
		afile << Agent1.pos << " " << Agent2.pos << " " << shadow1 << " " << shadow2 << " " << Fixed1 << " " << Fixed2 << endl;
		sfile << Agent1.sensor << " " << Agent2.sensor << " " << endl;
		n1file << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
		n2file << Agent2.NervousSystem.NeuronOutput(1) << " " << Agent2.NervousSystem.NeuronOutput(2) << " " << Agent2.NervousSystem.NeuronOutput(3) << endl; //" " << Agent1.NervousSystem.NeuronOutput(4) << " " << endl;
	}
	cout << Agent1.sensorweights << endl;
	afile.close();
	sfile.close();
	n1file.close();
	n2file.close();
}


// ------------------------------------
//  XX 
// ------------------------------------
void PerformanceCheck(TVector<double> &genotype)
{
	// Start output file
	ofstream perfile("perfcheck.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltimetrans;
	double shadow1, shadow2;
	int crosscounter; 
	double crossfit = 0.0;
	int totalshadows = 0;

	for (double x1 = 0; x1 < SpaceSize; x1 += 1.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 1.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
					{
						crosscounter += 1;
					}					
				}

			}

			crossfit += crosscounter;
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}

	perfile << totalfittrans/totaltrials << " " << crossfit/totaltrials << endl;
	perfile.close();
}

// ------------------------------------
// 2. Robustness analysis
// ------------------------------------
void RobustnessMapNoFix(TVector<double> &genotype)
{
	// Start output file
	ofstream fitfile("rm_fit_nofixed.dat");
	ofstream fittransfile("rm_fittran_nofixed.dat");
	ofstream distfile("rm_dist_nofixed.dat");
	ofstream crossfile("rm_cross_nofixed.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totalfittrans = 0.0, totaldist = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltime = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;
	double avgtotaldist = 0.0, avgtotaldisttrans = 0.0, avgcrosses = 0.0;
	int totalshadows = 0;

	for (double x1 = 0; x1 < SpaceSize; x1 += 1.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 1.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldist = 0.0;
			totaltime = 0;
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, 999999999);
				Agent2.Sense(Agent1.pos, shadow1, 999999999);				

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;
				totaldist += dist;
				totaltime += 1;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
					{
						crosscounter += 1;
					}					
				}

			}

			// Save the results
			fitfile << x1 << " " << x2 << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			fittransfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			distfile <<  x1 << " " << x2 << " " << Agent1.pos << endl;
			crossfile << x1 << " " << x2 << " " << crosscounter << endl;
			totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	fitfile.close();
	fittransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}
void RobustnessMapReg(TVector<double> &genotype)
{
	// Start output file
	ofstream fitfile("rm_fit_reg.dat");
	ofstream fittransfile("rm_fittran_reg.dat");
	ofstream distfile("rm_dist_reg.dat");
	ofstream crossfile("rm_cross_reg.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totalfittrans = 0.0, totaldist = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltime = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;
	double avgtotaldist = 0.0, avgtotaldisttrans = 0.0, avgcrosses = 0.0;
	int totalshadows = 0;

	for (double x1 = 0; x1 < SpaceSize; x1 += 1.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 1.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldist = 0.0;
			totaltime = 0;
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;
				totaldist += dist;
				totaltime += 1;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
					{
						crosscounter += 1;
					}					
				}

			}

			// Save the results
			fitfile << x1 << " " << x2 << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			fittransfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			distfile <<  x1 << " " << x2 << " " << Agent1.pos << endl;
			crossfile << x1 << " " << x2 << " " << crosscounter << endl;
			totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	fitfile.close();
	fittransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}
void RobustnessMapNoSha(TVector<double> &genotype)
{
	// Start output file
	ofstream fitfile("rm_fit_noshadow.dat");
	ofstream fittransfile("rm_fittran_noshadow.dat");
	ofstream distfile("rm_dist_noshadow.dat");
	ofstream crossfile("rm_cross_noshadow.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totalfittrans = 0.0, totaldist = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltime = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;
	double avgtotaldist = 0.0, avgtotaldisttrans = 0.0, avgcrosses = 0.0;
	int totalshadows = 0;

	for (double x1 = 0; x1 < SpaceSize; x1 += 1.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 1.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldist = 0.0;
			totaltime = 0;
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, 999999999, Fixed2);
				Agent2.Sense(Agent1.pos, 999999999, Fixed1);			

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;
				totaldist += dist;
				totaltime += 1;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
					{
						crosscounter += 1;
					}					
				}

			}

			// Save the results
			fitfile << x1 << " " << x2 << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			fittransfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			distfile <<  x1 << " " << x2 << " " << Agent1.pos << endl;
			crossfile << x1 << " " << x2 << " " << crosscounter << endl;
			totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	fitfile.close();
	fittransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}
void ShadowOffset(TVector<double> &genotype)
{
	// Start output file
	ofstream fitfile("shadowoffset.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	// 
	for (double shadowoffset1 = -300; shadowoffset1 <= 300; shadowoffset1 += 5.0) {
		for (double shadowoffset2 = -300; shadowoffset2 <= 300; shadowoffset2 += 5.0) {

			double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
			double totaltrials = 0, totaltime = 0.0;
			double shadow1, shadow2;

			for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
				for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

					// Set agents positions
					Agent1.Reset(x1);
					Agent2.Reset(x2);

					// Run the sim
					totaldist = 0.0;
					totaltime = 0;

					for (double time = 0; time < RunDurationMap; time += StepSize)
					{
						// Update shadow positions
						shadow1 = Agent1.pos + shadowoffset1;
						if (shadow1 >= SpaceSize)
							shadow1 = shadow1 - SpaceSize;
						if (shadow1 < 0.0)
							shadow1 = SpaceSize + shadow1;

						// Notice the other shadow is a reflection (not a rotation, i.e., shadow2 = Agent2.pos - Shadow;).
						shadow2 = Agent2.pos + shadowoffset2;
						if (shadow2 >= SpaceSize)
							shadow2 = shadow2 - SpaceSize;
						if (shadow2 < 0.0)
							shadow2 = SpaceSize + shadow2;

						// Sense
						Agent1.Sense(Agent2.pos, shadow2, Fixed2);
						Agent2.Sense(Agent1.pos, shadow1, Fixed1);

						// Move
						Agent1.Step(StepSize);
						Agent2.Step(StepSize);

						// Measure distance between them (after transients)
						if (time > TransDurationMap)
						{
							dist = fabs(Agent2.pos - Agent1.pos);
							if (dist > HalfSpace)
								dist =  SpaceSize - dist;
							if (dist < CloseEnoughRange)
								dist = CloseEnoughRange;
							totaldist += dist;
							totaltime += 1;
						}
					}
					totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					totaltrials += 1;
				}
			}
			
			// Save the results
			fitfile << shadowoffset1 << " " << shadowoffset2 << " " << totalfit/totaltrials << endl;
		}
	}
	fitfile.close();
}
void FixedPosition(TVector<double> &genotype)
{
	// Start output file
	ofstream fitfile("fixedpos.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	// 
	for (double fixpos1 = 0; fixpos1 <= 600; fixpos1 += 5.0) {
		for (double fixpos2 = 0; fixpos2 <= 600; fixpos2 += 5.0) {

			double totalfit = 0.0, totaldist = 0.0, dist = 0.0;
			double totaltrials = 0, totaltime = 0.0;
			double shadow1, shadow2;

			for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
				for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

					// Set agents positions
					Agent1.Reset(x1);
					Agent2.Reset(x2);

					// Run the sim
					totaldist = 0.0;
					totaltime = 0;

					for (double time = 0; time < RunDurationMap; time += StepSize)
					{
						// Update shadow positions
						shadow1 = Agent1.pos + Shadow;
						if (shadow1 >= SpaceSize)
							shadow1 = shadow1 - SpaceSize;
						if (shadow1 < 0.0)
							shadow1 = SpaceSize + shadow1;

						// Notice the other shadow is a reflection (not a rotation, i.e., shadow2 = Agent2.pos - Shadow;).
						shadow2 = Agent2.pos + Shadow;
						if (shadow2 >= SpaceSize)
							shadow2 = shadow2 - SpaceSize;
						if (shadow2 < 0.0)
							shadow2 = SpaceSize + shadow2;

						// Sense
						Agent1.Sense(Agent2.pos, shadow2, fixpos2);
						Agent2.Sense(Agent1.pos, shadow1, fixpos1);

						// Move
						Agent1.Step(StepSize);
						Agent2.Step(StepSize);

						// Measure distance between them (after transients)
						if (time > TransDurationMap)
						{
							dist = fabs(Agent2.pos - Agent1.pos);
							if (dist > HalfSpace)
								dist =  SpaceSize - dist;
							if (dist < CloseEnoughRange)
								dist = CloseEnoughRange;
							totaldist += dist;
							totaltime += 1;
						}
					}
					totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					totaltrials += 1;
				}
			}
			
			// Save the results
			fitfile << fixpos1 << " " << fixpos2 << " " << totalfit/totaltrials << endl;
		}
	}
	fitfile.close();
}

// ------------------------------------
// 2. Robustness analysis
// ------------------------------------
double PairTest(TVector<double> &genotype1, TVector<double> &genotype2)
{
	// Map genootype to phenotype
	TVector<double> phenotype1;
	phenotype1.SetBounds(1, VectSize);
	GenPhenMapping(genotype1, phenotype1);

	// Map genotype to phenotype
	TVector<double> phenotype2;
	phenotype2.SetBounds(1, VectSize);
	GenPhenMapping(genotype2, phenotype2);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype2(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype1(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype2(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype1(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype2(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype1(k));
		Agent2.SetSensorWeight(i,phenotype2(k));
		k++;
	}

	ofstream outfile;
	outfile.open("out2.dat");

	double totaltrials = 0,totalfittrans = 0.0,totaldisttrans = 0.0, dist = 0.0;
	int totaltimetrans;
	double shadow1, shadow2;

	for (double x1 = 0; x1 < SpaceSize; x1 += 25.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 25.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Measure distance between them
				dist = fabs(Agent2.pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;
				}

			}
			outfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	outfile.close();
	return totalfittrans/totaltrials;
}
// ------------------------------------
// 3. Decoy analysis
// ------------------------------------
void DecoyMap(TVector<double> &genotype, double frequency)
{
	// Start output file
	ofstream fitfile("dm_fit_freq_"+to_string(frequency)+".dat");
	ofstream fittransfile("dm_fittran_freq_"+to_string(frequency)+".dat");
	ofstream distfile("dm_dist_freq_"+to_string(frequency)+".dat");
	ofstream crossfile("dm_cross_freq_"+to_string(frequency)+".dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	double pos, pastpos;

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfit = 0.0, totalfittrans = 0.0, totaldist = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltime = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;
	double avgtotaldist = 0.0, avgtotaldisttrans = 0.0, avgcrosses = 0.0;
	int totalshadows = 0;

	for (double amplitude = 0; amplitude <= 4.0; amplitude += 0.01) {
		for (double velocity = -2; velocity <= 2.0; velocity += 0.01) {

			// Set agents positions
			Agent1.Reset(0.0);
			pos = 300.0;

			// Run the sim
			totaldist = 0.0;
			totaltime = 0;
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDuration; time += StepSize)
			{
				// Move decoy
				pastpos = pos;
				pos += StepSize * (velocity + (amplitude * sin(frequency*time)));
				// Wrap-around Environment
				if (pos >= SpaceSize)
					pos = pos - SpaceSize;
				if (pos < 0.0)
					pos = SpaceSize + pos;

				// Sense
				Agent1.Sense(pos, 999999999, 999999999);

				// Move
				Agent1.Step(StepSize);

				// Measure number of times the agents cross paths
				if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos)))
				{
					crosscounter += 1;
				}

				// Measure distance between them
				dist = fabs(pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;
				totaldist += dist;
				totaltime += 1;

				// Measure distance also for the fitness calc
				if (time > TransDuration)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;
				}

			}

			// Save the results
			fitfile << amplitude << " " << velocity << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			fittransfile << amplitude << " " << velocity << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
			distfile <<  amplitude << " " << velocity << " " << Agent1.pos << endl;
			crossfile << amplitude << " " << velocity << " " << crosscounter << endl;
			totalfit += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totalfittrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			totaltrials += 1;
		}
	}
	fitfile.close();
	fittransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}

// ------------------------------------
// 3. Decoy analysis
// ------------------------------------
void DecoyMapFixedVel(TVector<double> &genotype)
{
	// Start output file
	ofstream fittransfile("dm_fittran_vel0.dat");
	ofstream crossfile("dm_cross_vel0.dat");

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);


	// Create the agents
	PerceptualCrosser Agent1(1,N);
	double pos, pastpos;

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double totalfittrans = 0.0, totaldisttrans = 0.0, dist = 0.0;
	int totaltrials = 0, totaltimetrans;
	double shadow1, shadow2;
	double totalcross = 0;
	int crosscounter;

	for (double frequency = 0.0; frequency <= 1.0; frequency += 0.005) {
		for (double amplitude = 0.0; amplitude <= 1.0; amplitude += 0.005) {

			// Set agents positions
			Agent1.Reset(0.0);
			pos = 100.0;

			// Run the sim
			totaldisttrans = 0.0;
			totaltimetrans = 0;
			crosscounter = 0;

			for (double time = 0; time < RunDurationMap; time += StepSize)
			{
				// Move decoy
				pastpos = pos;
				pos = 100 + amplitude * sin(frequency*time);
				// pos = 300;
				// // Wrap-around Environment
				// if (pos >= SpaceSize)
				// 	pos = pos - SpaceSize;
				// if (pos < 0.0)
				// 	pos = SpaceSize + pos;

				// Sense
				Agent1.Sense(pos, 999999999, 999999999);

				// Move
				Agent1.Step(StepSize);

				// Measure distance between them
				dist = fabs(pos - Agent1.pos);
				if (dist > HalfSpace)
					dist =  SpaceSize - dist;

				// Measure distance also for the fitness calc
				if (time > TransDurationMap)
				{
					if (dist < CloseEnoughRange)
						dist = CloseEnoughRange;
					totaldisttrans += dist;
					totaltimetrans += 1;

					// Measure number of times the agents cross paths
					if (((Agent1.pastpos < pastpos) && (Agent1.pos >= pos)) || ((Agent1.pastpos > pastpos) && (Agent1.pos <= pos)))
					{
						crosscounter += 1;
					}					
				}

			}
			double fit;
			// Save the results
			fit = 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
			if (fit < 0.97)
				fit = 0.5;
			fittransfile << amplitude << " " << frequency << " " << fit << endl;
			crossfile << amplitude << " " << frequency << " " << crosscounter << endl;

		}
	}
	fittransfile.close();
	crossfile.close();
}


void LimitSet(TVector<double> &genotype, double sensorstate)
{
	ofstream limitsetfile("limitset_"+to_string(sensorstate)+".dat");
	//ofstream limitsetfile2("limitset2_"+to_string(sensorstate)+".dat");

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent(1,N);

	// Instantiate the nervous systems
	Agent.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent.SetSensorWeight(i,phenotype(k));
		k++;
	}

	// Repetitions for random starting conditions
	for (double r = 0; r < 10; r++)
	{
		// Set agents positions
		Agent.Reset(300.0);
		Agent.NervousSystem.RandomizeCircuitOutput(0.01,0.99);
		Agent.SetSensorState(sensorstate);

		// Run the sim for transient duration without recording
		for (double time = 0; time < 200; time += StepSize)
		{
			Agent.Step(StepSize);
		}
		// Run the sim for transient duration without recording
		for (double time = 0; time < 20; time += StepSize)
		{
			Agent.Step(StepSize);
			limitsetfile << Agent.NervousSystem.NeuronOutput(2) << " " << Agent.NervousSystem.NeuronOutput(1) << endl;
			// limitsetfile << Agent.NervousSystem.NeuronOutput(2) - Agent.NervousSystem.NeuronOutput(1) << " " << Agent.NervousSystem.NeuronOutput(3) << " " << endl;
			//limitsetfile2 << Agent.NervousSystem.NeuronOutput(2) - Agent.NervousSystem.NeuronOutput(1) << " " << Agent.NervousSystem.NeuronOutput(4) << " " << endl;
		}
	}
	limitsetfile.close();
	//limitsetfile2.close();
}

void InteractionNeuralLimitSet(TVector<double> &genotype)
{
	// Start output file
	ofstream inlfile("int_neural_limit_13S.dat");

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);
	PerceptualCrosser Agent2(-1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	Agent2.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		Agent2.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			Agent2.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		Agent2.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;

	for (double x1 = 0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0; x2 < SpaceSize - x1; x2 += 50.0) {

			// Set agents positions
			Agent1.Reset(x1);
			Agent2.Reset(x2);

			// Run the sim
			for (double time = 0; time < TransDurationMap; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, shadow2, Fixed2); //shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, shadow1, Fixed1); //shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

			}

			for (double time = 0; time < 40.0; time += StepSize)
			{
				// Update shadow positions
				shadow1 = Agent1.pos + Shadow;
				if (shadow1 >= SpaceSize)
					shadow1 = shadow1 - SpaceSize;
				if (shadow1 < 0.0)
					shadow1 = SpaceSize + shadow1;

				// XXX shadow2 = Agent2.pos - Shadow;
				shadow2 = Agent2.pos + Shadow;
				if (shadow2 >= SpaceSize)
					shadow2 = shadow2 - SpaceSize;
				if (shadow2 < 0.0)
					shadow2 = SpaceSize + shadow2;

				// Sense
				Agent1.Sense(Agent2.pos, 999999999, 999999999); //shadow2, Fixed2);
				Agent2.Sense(Agent1.pos, 999999999, 999999999); //shadow1, Fixed1);

				// Move
				Agent1.Step(StepSize);
				Agent2.Step(StepSize);

				// Save
				inlfile << Agent1.sensor << " " << Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << endl;
				//inlfile << Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1) << " " << Agent1.NervousSystem.NeuronOutput(3) << " " << Agent1.sensor << endl;
			}
		}
	}
	inlfile.close();
}


double Handedness(TVector<double> &genotype)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N);

	// Instantiate the nervous systems
	Agent1.NervousSystem.SetCircuitSize(N);
	int k = 1;

	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Biases
	for (int i = 1; i <= N; i++) {
		Agent1.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent1.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent1.SetSensorWeight(i,phenotype(k));
		k++;
	}

	double shadow1, shadow2;
	double x1 = 300.0; 

	// Set agents positions
	Agent1.Reset(x1);

	// Run the sim
	for (double time = 0; time < 200; time += StepSize)
	{
		Agent1.Step(StepSize);
	}
	cout << Agent1.NervousSystem.NeuronOutput(2) << " " << Agent1.NervousSystem.NeuronOutput(1) << endl;
	if ((Agent1.NervousSystem.NeuronOutput(2) - Agent1.NervousSystem.NeuronOutput(1)) > 0){
		return 1.0;
	}
	else{
		return -1.0;
	}
}
// ================================================
// C. ADDITIONAL EVOLUTIONARY FUNCTIONS
// ================================================
int TerminationFunction(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	if (BestPerf > 0.99) return 1;
	else return 0;
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Also show the best individual in the Circuit Model form
	BestIndividualFile.open("best.ns.dat");
	GenPhenMapping(bestVector, phenotype);
	PerceptualCrosser Agent(1,N);

	// Instantiate the nervous system
	Agent.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Agent.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Agent.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
		// Sensor Weights
	for (int i = 1; i <= N; i++) {
		Agent.SetSensorWeight(i,phenotype(k));
		k++;
	}
	BestIndividualFile << Agent.NervousSystem << endl;
	BestIndividualFile << Agent.sensorweights << "\n" << endl;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
	// long randomseed = static_cast<long>(time(NULL));
	// if (argc == 2)
	// 	randomseed += atoi(argv[1]);

	// TSearch s(VectSize);
	
	// #ifdef PRINTOFILE

	// ofstream file;
	// file.open("evol.dat");
	// cout.rdbuf(file.rdbuf());
	
	// // save the seed to a file
	// ofstream seedfile;
	// seedfile.open ("seed.dat");
	// seedfile << randomseed << endl;
	// seedfile.close();
	
	// #endif
	
	// // Configure the search
	// s.SetRandomSeed(randomseed);
	// s.SetSearchResultsDisplayFunction(ResultsDisplay);
	// s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	// s.SetSelectionMode(RANK_BASED);
	// s.SetReproductionMode(GENETIC_ALGORITHM);
	// s.SetPopulationSize(POPSIZE);
	// s.SetMaxGenerations(GENS);
	// s.SetCrossoverProbability(CROSSPROB);
	// s.SetCrossoverMode(UNIFORM);
	// s.SetMutationVariance(MUTVAR);
	// s.SetMaxExpectedOffspring(EXPECTED);
	// s.SetElitistFraction(ELITISM);
	// s.SetSearchConstraint(1);
	
	// /* Stage 1 */
	// s.SetSearchTerminationFunction(TerminationFunction);
	// s.SetEvaluationFunction(FitnessFunction1); 
	// s.ExecuteSearch();
	// /* Stage 2 */
	// s.SetSearchTerminationFunction(NULL);
	// s.SetEvaluationFunction(FitnessFunction2);
	// s.ExecuteSearch();	

// ================================================
// B. MAIN FOR ANALYZING A SUCCESFUL CIRCUIT
// ================================================
	// ifstream genefile;
	// genefile.open("best.gen.dat");
	// TVector<double> genotype(1, VectSize);
	// genefile >> genotype;

	// PerformanceCheck(genotype);
	// for (double sensorstate = 0.0; sensorstate <= 1.0; sensorstate += 0.1)
	// {
	// 	LimitSet(genotype,sensorstate);
	// }
	// InteractionNeuralLimitSet(genotype);
	// NeuralTracesRegular(genotype);
	// DecoyMapFixedVel(genotype);

	// // 0. Behavioral traces
	// BehavioralTracesRegular(genotype);
	// BehavioralTracesNoShadow(genotype);
	// BehavioralTracesNoFixed(genotype);

	// 1. Neural Traces
	// NeuralTracesRegular(genotype);
	// NeuralTracesNoShadow(genotype);
	// NeuralTracesNoFixed(genotype);

	// // 2. Robustness maps
	// RobustnessMapReg(genotype);
	// RobustnessMapNoFix(genotype);
	// RobustnessMapNoSha(genotype);	
	// ShadowOffset(genotype);
	// FixedPosition(genotype);

	// // 3. Decoy map
	// DecoyMap(genotype, 0.1);
	// DecoyMap(genotype, 0.5);
	// DecoyMap(genotype, 1.0);
	// DecoyMap(genotype, 2.0);
	// DecoyMapFixedVel(genotype);

// ================================================
// C. MAIN FOR ANALYZING A SUCCESFUL CIRCUIT
// ================================================
	string g1 = argv[1];
	string g2 = argv[2];	
	string r1 = argv[3];
	string r2 = argv[4];

	ifstream genefile1;
	genefile1.open(g1+"/best.gen.dat");
	TVector<double> genotype1(1, VectSize);
	genefile1 >> genotype1;

	ifstream genefile2;
	genefile2.open(g2+"/best.gen.dat");
	TVector<double> genotype2(1, VectSize);
	genefile2 >> genotype2;

	double result = PairTest(genotype1, genotype2);
	ofstream otherfile;
	otherfile.open("othersmatrix_new.dat", std::ios_base::app);
	otherfile << r1 << " " << r2 << " " << result << endl;
	otherfile.close();

	// double result1 = Handedness(genotype1);
	// double result2 = Handedness(genotype2);
	// ofstream otherfile;
	// otherfile.open("handednessmatrix.dat", std::ios_base::app);
	// otherfile << r1 << " " << r2 << " " << result1*result2 << endl;
	// otherfile.close();

	return 0;
}
