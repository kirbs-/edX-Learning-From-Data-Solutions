// -------------------------------------------------------
// Visual Studio 2013 C++ solution for HW5, Logistic Regression, for CS1156x Learning From Data (introductory Machine Learning course)
//
// Author:	Victor Vernescu (Victoras)
// Notes:	The implemetation uses C++11. To compile using C++98, just replace the auto variables with the proper iterator types 
//			and rewrite the random function.
// -------------------------------------------------------

#include "stdafx.h"

// Structure can contain either a set of weights or a data point
struct DataPoint
{
	DataPoint() { w[0] = w[1] = w[2] = 0.0; }
	long double w[3];
	char cat;

	inline long double& operator[](const int x)
	{
		return w[x];
	}

	inline long double operator[](const int x) const
	{
		return w[x];
	}
};

#pragma region Operators and Functions 
DataPoint operator*(const long double& d, const DataPoint& f)
{
	DataPoint res = f;
	
	res[0] *= d;
	res[1] *= d;
	res[2] *= d;

	return res;
}

DataPoint operator/(const DataPoint& f, const long double d)
{
	DataPoint res = f;

	res[0] /= d;
	res[1] /= d;
	res[2] /= d;

	return res;
}

DataPoint operator-(const DataPoint& f1, const DataPoint& f2)
{
	DataPoint res = f1;

	res[0] -= f2[0];
	res[1] -= f2[1];
	res[2] -= f2[2];

	return res;
}

long double operator*(const DataPoint& x, const DataPoint& y)
{
	return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

long double norm(const DataPoint& x)
{
	return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

#pragma endregion Operators and other functions (norm)

#pragma region Data Generation
template<class Func>

// Generate a data point (1,x1,x2) with x1/x2 in [-1.0, 1.0]
DataPoint generateDataPoint(Func randomCoord)
{
	DataPoint p;

	p[0] = 1.0;
	p[1] = randomCoord();
	p[2] = randomCoord();

	return p;
}

// Generate the target function by getting 2 points and calculating the 
// coefficients (r0 + r1*x1 + r2*x2 = 0) of the line that passes through both
// See my post in the thread below about how to derive the coefficients:
// https://courses.edx.org/courses/CaltechX/CS1156x/Fall2013/discussion/forum/undefined/threads/5250371eecddb3395a000007
template<class Func> 
DataPoint generateTargetFunction(Func randomCoord)
{
	DataPoint res;
	DataPoint p1, p2;

	p1 = generateDataPoint(randomCoord);
	p2 = generateDataPoint(randomCoord);

	res[1] = p2[2] - p1[2];
	res[2] = p1[1] - p2[1];
	res[0] = p1[1] * (p1[2] - p2[2]) + p1[2] * (p2[1] - p1[1]);

	return res;
}

// Generate D data points and store them in the data vector. Use f to categorize them.
template<class Func>
void generateData(const int& D, const DataPoint& f, vector<DataPoint> &data, Func randomCoord)
{
	data.clear();
	DataPoint p;

	for (int i = 0; i < D; i++)
	{
		p = generateDataPoint(randomCoord);
		p.cat = applyFunction(p, f);
		data.push_back(p);
	}
}

#pragma endregion Helper functions used to generate the data

// Apply the my hypothesis to a certain point
char applyFunction(const DataPoint& point, const DataPoint& hypo)
{
	long double det = hypo[0] * point[0] + hypo[1] * point[1] + hypo[2] * point[2];

	return (det < 0) ? -1 : 1;
}

// Run logistic regression on the first N points in the data vector
// Returns number of epochs and the solution in g and 
long runLogisticRegression(const int& N, vector<DataPoint>& data, DataPoint& g)
{
	const long double eta = 0.01;
	const long double eps = 0.01;

	DataPoint w, wep;
	int epoch = 0;

	do
	{
		// Save the weights at the beggining of the epoch
		wep = w;

		// Get a random permutation of our points
		random_shuffle(data.begin(), data.begin() + N);

		// For each point in the data set, calculate gradient and then update the weights
		for (auto it = data.cbegin(); it != data.cbegin() + N; it++)
		{
			const DataPoint x = *it;
			DataPoint grad = (-x.cat * x) / (1 + expl(x.cat * (w * x)));
			w = w - eta * grad;	
		}

		epoch++;
		// Break when the norm is smaller than eps
	} while (norm(w - wep) > eps);

	g = w;

	return epoch;
}

int main(int argc, char* argv[])
{
	const int N = 100; /// Number of in-sample points
	const int D = 50000; // Number of total points (D - N will be used to calculate E_out)
	const int runs = 1000; // Number of times to run the algorithm
	long double eIn = 0;
	long double eOut = 0;
	long miscalc = 0;
	long epoch = 0;

	vector<DataPoint> data;
	DataPoint f; // Target function
	DataPoint g; // Solution

	// Initialize RNG and bind the generator to a function (C++11 magic :D).
	default_random_engine generator;
	generator.seed((unsigned long)chrono::system_clock::now().time_since_epoch().count());

	uniform_real_distribution<double> faceDistribution(-1.0, 1.0);
	auto randomCoord = bind(faceDistribution, ref(generator));

	for (int i = 0; i < runs; i++)
	{
		// Generate target function and data set
		// The first N point will be the training set and the rest will be used to calculate E_out
		f = generateTargetFunction(randomCoord);
		generateData(D, f, data, randomCoord);

		// Run logistic regression
		epoch += runLogisticRegression(N, data, g);

		// Calculate E_in
		for (auto it = data.cbegin(); it != data.cbegin() + N; it++)
		{
			eIn += logl(1 + expl(-(*it).cat * (g * (*it))));
		}

		// Calculate E_out
		for (auto it = data.cbegin(); it != data.cend(); it++)
		{
			DataPoint x = *it;
			eOut += logl(1.0 + expl(-x.cat * (g * x)));

			// See how many points are actually missclasified, not in the HW, just to satisfy my curiosity
			char cat = applyFunction(*it, g);
			if (cat != (*it).cat)
			{
					miscalc++;
			}
		}

		// Display some kind of progress status every 25 runs
		if (i % 25 == 0)
		{
			cout << "Run " << i << " completed!" << endl;
		}
	}

	cout << fixed << "For " << runs << " runs on N = " << N << " we have E_in = " << eIn / runs / N;
	cout << " and E_out = " << eOut / runs / (D - N) << " and Epochs = " << epoch / runs;
	cout << " and Misclassified% = " << (long double)miscalc / runs / (D - N) << endl;

	return 0;
}