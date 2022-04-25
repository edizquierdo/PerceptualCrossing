#include "TSearch.h"
#include "PerceptualCrosser.h"
#include "CTRNN.h"
#include "random.h"

#define PRINTOFILE

// Task params
const double StepSize = 0.1; //0.05; //0.1;
const double RunDuration = 800.0; //1600.0; //800.0;
const int Delay = 0; //25
const int MaxSteps = 8000 + 0 + 1; //8000 + 250; //250

const double TransDuration = 400.0; //400.0;
const double Fixed1 = 150.0;
const double Fixed2 = 450.0;
const double Shadow = 48.0; // YYY Maybe this should be tested to random between 48 and 52
const double SpaceSize = 600.0;
const double HalfSpace = 300;
const double SenseRange = 2.0; //1.0; //0.5; //2.0;
const double CloseEnoughRange = 2.0; // 2.0;

// EA params
const int POPSIZE = 96;
const int GENS = 500;
const double MUTVAR = 0.05;			// ~ 1/VectSize for N=3
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.02;

// Nervous system params
const int N = 3; //3;
const double WR = 8.0;
const double SR = 8.0;
const double BR = 8.0;
const double TMIN = 1.0;
const double TMAX = 10.0;

int	VectSize = N*N + 2*N + N;

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
// Fitness function
// ------------------------------------
double FitnessFunctionP(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N,Delay,MaxSteps);
	PerceptualCrosser Agent2(-1,N,Delay,MaxSteps);

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

	for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

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
	return totalfit/totaltrials;
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionPC(TVector<double> &genotype, RandomState &rs)
{
	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N,Delay,MaxSteps);
	PerceptualCrosser Agent2(-1,N,Delay,MaxSteps);

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

	for (double x1 = 0.0; x1 < SpaceSize; x1 += 50.0) {
		for (double x2 = 0.0; x2 < SpaceSize - x1; x2 += 50.0) {

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
	if (totalfit/totaltrials > 0.99){
		return 1.0 + (totalcross/totaltrials);
	}
	else{
		return totalfit/totaltrials;
	}
}

// ------------------------------------
// Performance Map
// ------------------------------------
void RobustnessMap(TVector<double> &genotype, std::string fFileName,  std::string ftFileName, std::string oFileName,  std::string otFileName, std::string dFileName, std::string cFileName)
{
	// Start output file
	ofstream fitfile(fFileName);
	ofstream fittransfile(ftFileName);
	ofstream outputfile(oFileName);
	ofstream outputtransfile(otFileName);
	ofstream distfile(dFileName);
	ofstream crossfile(cFileName);

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N,Delay,MaxSteps);
	PerceptualCrosser Agent2(-1,N,Delay,MaxSteps);

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

			avgtotaldist = 0.0;
			avgtotaldisttrans = 0.0;
			avgcrosses = 0.0;
			totalshadows = 0;

			// Repeat for shadows at different lengths, systematically.
			for (double shadowoffset1 = 48.0; shadowoffset1 <= 52.0; shadowoffset1 += 2.0) {
				for (double shadowoffset2 = 48.0; shadowoffset2 <= 52.0; shadowoffset2 += 2.0) {

					// Set agents positions
					Agent1.Reset(x1);
					Agent2.Reset(x2);

					// Run the sim
					totaldist = 0.0;
					totaltime = 0;
					totaldisttrans = 0.0;
					totaltimetrans = 0;
					crosscounter = 0;

					for (double time = 0; time < RunDuration; time += StepSize)
					{
						// Update shadow positions
						shadow1 = Agent1.pos + shadowoffset1;
						if (shadow1 >= SpaceSize)
							shadow1 = shadow1 - SpaceSize;
						if (shadow1 < 0.0)
							shadow1 = SpaceSize + shadow1;

						// XXX shadow2 = Agent2.pos - Shadow;
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

						// Measure number of times the agents cross paths
						if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
						{
								crosscounter += 1;
						}

						// Measure distance between them
						dist = fabs(Agent2.pos - Agent1.pos);
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

					if ((shadowoffset1==48.0) && (shadowoffset2==48.0)){
						fitfile << x1 << " " << x2 << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
						fittransfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
						distfile <<  x1 << " " << x2 << " " << Agent1.pos << endl;
					}

					avgtotaldist += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					avgtotaldisttrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					avgcrosses += crosscounter;
					totalshadows += 1;

				}
			}

			// Save the results
			outputfile << x1 << " " << x2 << " " << avgtotaldist/totalshadows << endl;
			outputtransfile << x1 << " " << x2 << " " << avgtotaldisttrans/totalshadows << endl;
			crossfile << x1 << " " << x2 << " " << avgcrosses/totalshadows << endl;
			totalfit += avgtotaldist/totalshadows;
			totalfittrans += avgtotaldisttrans/totalshadows;
			totaltrials += 1;

		}
	}
	fitfile.close();
	fittransfile.close();
	outputfile.close();
	outputtransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}

// ------------------------------------
// Sensory Inversion Map
// ------------------------------------
void SensoryInversionMap(TVector<double> &genotype, std::string fFileName,  std::string ftFileName, std::string oFileName,  std::string otFileName, std::string dFileName, std::string cFileName)
{
	// Start output file
	ofstream fitfile(fFileName);
	ofstream fittransfile(ftFileName);
	ofstream outputfile(oFileName);
	ofstream outputtransfile(otFileName);
	ofstream distfile(dFileName);
	ofstream crossfile(cFileName);

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N,Delay,MaxSteps);
	PerceptualCrosser Agent2(-1,N,Delay,MaxSteps);

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

	for (double x1 = 0; x1 < SpaceSize; x1 += 10.0) {
		for (double x2 = 0; x2 < SpaceSize; x2 += 10.0) {

			avgtotaldist = 0.0;
			avgtotaldisttrans = 0.0;
			avgcrosses = 0.0;
			totalshadows = 0;

			// Repeat for shadows at different lengths, systematically.
			for (double shadowoffset1 = 48.0; shadowoffset1 <= 48.0; shadowoffset1 += 2.0) {
				for (double shadowoffset2 = 48.0; shadowoffset2 <= 48.0; shadowoffset2 += 2.0) {

					// Set agents positions
					Agent1.Reset(x1);
					Agent2.Reset(x2);

					// Run the sim
					totaldist = 0.0;
					totaltime = 0;
					totaldisttrans = 0.0;
					totaltimetrans = 0;
					crosscounter = 0;

					for (double time = 0; time < RunDuration; time += StepSize)
					{
						// Update shadow positions
						shadow1 = Agent1.pos + shadowoffset1;
						if (shadow1 >= SpaceSize)
							shadow1 = shadow1 - SpaceSize;
						if (shadow1 < 0.0)
							shadow1 = SpaceSize + shadow1;

						// XXX shadow2 = Agent2.pos - Shadow;
						shadow2 = Agent2.pos + shadowoffset2;
						if (shadow2 >= SpaceSize)
							shadow2 = shadow2 - SpaceSize;
						if (shadow2 < 0.0)
							shadow2 = SpaceSize + shadow2;

						// Sense
						Agent1.Sense(Agent2.pos, shadow2, Fixed2);
						Agent2.Sense(Agent1.pos, shadow1, Fixed1);

						// Invert sensory informat
						double a1s = Agent1.sensorhist(Agent1.step);
						double a2s = Agent2.sensorhist(Agent2.step);
						Agent1.sensorhist(Agent1.step) = a2s;
						Agent2.sensorhist(Agent2.step) = a1s;

						// Move
						Agent1.Step(StepSize);
						Agent2.Step(StepSize);

						// Measure number of times the agents cross paths
						if (((Agent1.pastpos < Agent2.pastpos) && (Agent1.pos >= Agent2.pos)) || ((Agent1.pastpos > Agent2.pastpos) && (Agent1.pos <= Agent2.pos)))
						{
								crosscounter += 1;
						}

						// Measure distance between them
						dist = fabs(Agent2.pos - Agent1.pos);
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

					if ((shadowoffset1==48.0) && (shadowoffset2==48.0)){
						fitfile << x1 << " " << x2 << " " << 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
						fittransfile << x1 << " " << x2 << " " << 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange)) << endl;
						distfile <<  x1 << " " << x2 << " " << Agent1.pos << endl;
					}

					avgtotaldist += 1 - (((totaldist / totaltime) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					avgtotaldisttrans += 1 - (((totaldisttrans / totaltimetrans) - CloseEnoughRange) / (HalfSpace - CloseEnoughRange));
					avgcrosses += crosscounter;
					totalshadows += 1;

				}
			}

			// Save the results
			outputfile << x1 << " " << x2 << " " << avgtotaldist/totalshadows << endl;
			outputtransfile << x1 << " " << x2 << " " << avgtotaldisttrans/totalshadows << endl;
			crossfile << x1 << " " << x2 << " " << avgcrosses/totalshadows << endl;
			totalfit += avgtotaldist/totalshadows;
			totalfittrans += avgtotaldisttrans/totalshadows;
			totaltrials += 1;

		}
	}
	fitfile.close();
	fittransfile.close();
	outputfile.close();
	outputtransfile.close();
	distfile.close();
	crossfile.close();
	cout << "Robust performance: " << totalfit/totaltrials << " " << totalfittrans/totaltrials << endl;
}

// ------------------------------------
// Behavioral Trace
// ------------------------------------
void BehavioralTraces(TVector<double> &genotype, std::string a1FileName, std::string a2FileName)
{
	// Start output file
	ofstream a1file(a1FileName);
	ofstream a2file(a2FileName);

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agents
	PerceptualCrosser Agent1(1,N,Delay,MaxSteps);
	PerceptualCrosser Agent2(-1,N,Delay,MaxSteps);

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
				// // Measure number of times the agents cross paths
				// if (((Agent1.pastpos < 10) && (Agent1.pos > 590)) || ((Agent1.pastpos > 590) && (Agent1.pos < 10)))
				// {
				// 		a1file << endl;
				// }
				// if (((Agent2.pastpos < 10) && (Agent2.pos > 590)) || ((Agent2.pastpos > 590) && (Agent2.pos < 10)))
				// {
				// 		a2file << endl;
				// }
				a1file << time << " " << Agent1.pos << endl;
				a2file << time << " " << Agent2.pos << endl;
			}
		}
	}
	a1file.close();
	a2file.close();
}

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
	PerceptualCrosser Agent(1,N,Delay,MaxSteps);

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
	BestIndividualFile << Agent.NervousSystem;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) {

	// long IDUM=-time(0);
	// TSearch s(VectSize);
	//
	// #ifdef PRINTOFILE
	// ofstream file;
	// file.open("evol.dat");
	// cout.rdbuf(file.rdbuf());
	// #endif
	//
	// // Configure the search
	// s.SetRandomSeed(IDUM);
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
	//
	// // /* Stage 1 */
	// // s.SetSearchTerminationFunction(TerminationFunction);
	// // s.SetEvaluationFunction(FitnessFunctionP);
	// // s.ExecuteSearch();
	// // /* Stage 2 */
	// // s.SetSearchTerminationFunction(NULL);
	// // s.SetEvaluationFunction(FitnessFunctionPC);
	// // s.ExecuteSearch();
	//
	// /* Stage X */
	// s.SetEvaluationFunction(FitnessFunctionP);
	// s.ExecuteSearch();

	 ifstream genefile;
	 std::string const & gFileName = "best.gen.dat"; //argv[1];
	 // std::string const & fFileName = "f_si.dat"; //argv[3];
   // std::string const & ftFileName = "ft_si.dat"; //argv[4];
	 // std::string const & oFileName = "pf_si.dat"; //argv[2];
	 // std::string const & otFileName = "pft_si.dat"; //argv[2];
	 // std::string const & dFileName = "di_si.dat"; //argv[3];
   // std::string const & cFileName = "cr_si.dat"; //argv[4];
	 std::string const & a1FileName = "a1pos.dat"; //argv[5];
	 std::string const & a2FileName = "a2pos.dat"; //argv[6];
	 std::string const & rdFileName = "rdpos.dat"; //argv[6];
	 genefile.open(gFileName);
	 TVector<double> genotype(1, VectSize);
	 genefile >> genotype;
	 //RobustnessMap(genotype,fFileName,ftFileName,oFileName,otFileName,dFileName,cFileName);
	 //SensoryInversionMap(genotype,fFileName,ftFileName,oFileName,otFileName,dFileName,cFileName);
	 BehavioralTraces(genotype,a1FileName,a2FileName,rdFileName);
	 // genefile.close();

	return 0;
}
