#pragma once
#include <vector>
#include <map>
#include <algorithm>
#include "measurements.h"
#include "dump.h"
#include "fast_update.h"
#include "lattice.h"
#include "Random.h"
#include "parameters.h"

// Argument type
struct arg_t
{
	int tau;
	std::map<std::pair<int, int>, double> sigma;

	double operator()(int i, int j) const
	{
		return sigma.at({std::min(i, j), std::max(i, j)});
	}

	double& operator()(int i, int j)
	{
		return sigma.at({std::min(i, j), std::max(i, j)});
	}

	void serialize(odump& out)
	{
		out.write(tau);
		int size = sigma.size();
		out.write(size);
		for (auto& s : sigma)
		{
			out.write(s.first.first);
			out.write(s.first.second);
			out.write(s.second);
		}
	}

	void serialize(idump& in)
	{
		double t; in.read(t); tau = t;
		int n; in.read(n);
		for (int k = 0; k < n; ++k)
		{
			int i, j; in.read(i); in.read(j);
			double x; in.read(x);
			sigma[{std::min(i, j), std::max(i, j)}] = x;
		}
	}
};

// The Monte Carlo configuration
struct configuration
{
	lattice l;
	parameters param;
	measurements& measure;
	fast_update<arg_t> M;
	std::vector<int> shellsize;

	configuration(measurements& measure_)
		: l(), param(), measure(measure_), M(fast_update<arg_t>(l, param, measure))
			
	{}
	
	void initialize()
	{
		shellsize.resize(l.max_distance() + 1, 0);
		for (int d = 0; d <= l.max_distance(); ++d)
		{
			int site = 0; //PBC used here
			for (int j = 0; j < l.n_sites(); ++j)
				if (l.distance(site, j) == d)
					shellsize[d] += 1;
		}
		M.initialize();
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
