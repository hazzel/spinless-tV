#pragma once
#include <vector>
#include <algorithm>
#include "measurements.h"
#include "fast_update.h"
#include "lattice.h"
#include "configuration.h"
#include "Random.h"

// k! / (k+n)!
template<typename T>
double factorial_ratio(T k, T n)
{
	if (k <= T(0) || n <= T(0))
		return 1.0;
	double result = 1.0;
	for (int i = 1; i <= n; ++i)
		result /= static_cast<double>(k + i);
	return result;
}

struct move_flip
{
	configuration* config;
	Random& rng;
	bool save_acc;

	double attempt()
	{
		return 1.0;
	}

	double accept()
	{
		if (save_acc)
			config->measure.add("flip field", 1.0);
		return 1.0;
	}

	void reject()
	{
		if (save_acc)
			config->measure.add("flip field", 0.0);
	}
};
