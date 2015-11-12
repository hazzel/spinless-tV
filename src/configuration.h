#pragma once
#include <vector>
#include <map>
#include <algorithm>
#include "measurements.h"
#include "dump.h"
#include "fast_update.h"
#include "lattice.h"
#include "Random.h"

// Argument type
struct arg_t
{
	int tau;
	std::map<std::pair<int, int>, double> sigma;

	double operator()(int i, int j) const
	{
		return sigma.at({std::min(i, j), std::max(i, j)});
	}

	void serialize(odump& out)
	{
		out.write(tau);
	}

	void serialize(idump& in)
	{
		double t; in.read(t); tau = t;
	}
};

struct parameters
{	
	double beta, n_tau_slices, dtau, V, lambda;
};

// The function that appears in the calculation of the determinant
struct h_entry
{
	const lattice& l;
	const parameters& params;

	double operator()(const arg_t& x, int i, int j) const
	{
		if (l.distance(i, j) == 1)
			return params.dtau + params.lambda * x(i, j);
		else
			return 0.;
	}
};

// The Monte Carlo configuration
struct configuration
{
	const lattice& l;
	fast_update<h_entry, arg_t> M;
	const parameters& params;
	measurements& measure;
	std::vector<int> shellsize;

	configuration(const lattice& l_, const parameters& params_,
		measurements& measure_)
		: l(l_), M{h_entry{l_, params_}, l_}, params(params_),
		measure(measure_)
	{
		shellsize.resize(l.max_distance() + 1, 0);
		for (int d = 0; d <= l.max_distance(); ++d)
		{
			int site = 0; //PBC used here
			for (int j = 0; j < l.n_sites(); ++j)
				if (l.distance(site, j) == d)
					shellsize[d] += 1;
		}
	}

	void serialize(odump& out)
	{
		M.serialize(out);
	}

	void serialize(idump& in)
	{
		M.serialize(in);
	}
};
