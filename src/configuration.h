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

#include <chrono>

/*
// Argument type
struct arg_t
{
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
		int n; in.read(n);
		sigma.resize(n);
		for (int k = 0; k < n; ++k)
		{
			int i, j; in.read(i); in.read(j);
			double x; in.read(x);
			sigma.insert({{std::min(i, j), std::max(i, j)}, x});
		}
	}
};
*/

// Argument type
struct arg_t
{
	std::vector<int> sigma;
	int int_size = 8 * sizeof(int);
	
	arg_t(int size = 0)
		: sigma(size, 0)
	{}
	
	double get(int i) const
	{
		int n = i / int_size;
		return test_bit(sigma[n], i % int_size) ? 1. : -1;
	}

	void set(int i)
	{
		int n = i / int_size;
		sigma[n] = set_bit(sigma[n], i % int_size);
	}
	
	void flip(int i)
	{
		int n = i / int_size;
		sigma[n] = invert_bit(sigma[n], i % int_size);
	}
	
	void serialize(odump& out)
	{
		int size = sigma.size();
		out.write(size);
		for (auto& s : sigma)
			out.write(s);
	}

	void serialize(idump& in)
	{
		int n; in.read(n);
		sigma.resize(n);
		for (int k = 0; k < n; ++k)
		{
			int i; in.read(i);
			sigma[k] = i;
		}
	}
	
private:
	int set_bit(int integer, int offset) const
	{
		return integer | (1 << offset);
	}
	int clear_bit(int integer, int offset) const
	{
		return integer & (~(1 << offset));
	}
	int invert_bit(int integer, int offset) const
	{
		return integer ^ (1 << offset);
	}
	int test_bit(int integer, int offset) const
	{
		return (integer & (1 << offset)) >> offset;
	}
};

// The Monte Carlo configuration
struct configuration
{
	Random& rng;
	lattice l;
	parameters param;
	measurements& measure;
	fast_update<arg_t> M;
	std::vector<int> shellsize;
	std::complex<double> sign_phase=1.;

	configuration(Random& rng_, measurements& measure_)
		: rng(rng_), l(), param(), measure(measure_), M(fast_update<arg_t>(rng, l, param, measure))
			
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
