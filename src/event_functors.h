#pragma once
#include <map>
#include <vector>
#include "measurements.h"
#include "configuration.h"

struct event_build
{
	configuration& config;
	Random& rng;

	void trigger()
	{
		boost::multi_array<arg_t, 2> initial_vertices(
			boost::extents[2][config.param.n_tau_slices]);
		for (int i = 0; i < 2; ++i)
			for (int t = 1; t <= config.param.n_tau_slices; ++t)
			{
				std::map<std::pair<int, int>, double> sigma;
				for (int j = 0; j < config.l.n_sites(); ++j)
					for (int k = j; k < config.l.n_sites(); ++k)
						if (config.l.distance(j, k) == 1)
							sigma[{j, k}] = static_cast<int>(rng()*2.)*2-1;
				initial_vertices[i][t-1] = {t, sigma};
			}
		config.M.build(initial_vertices);
	}
};

struct event_flip_all
{
	configuration& config;
	Random& rng;

	void trigger()
	{
		int m = config.l.n_sites();
		std::vector<std::pair<int, int>> sites(m);
		for (int s = 0; s < 2; ++s)
			for (int n = 0; n < config.l.n_sites(); ++n)
			{
				int i = rng() * config.l.n_sites();
				auto& neighbors = config.l.neighbors(i, "nearest neighbors");
				int j = neighbors[rng() * neighbors.size()];
				double p = config.M.try_ising_flip(s, i, j);
				if (rng() < p)
				{
					config.M.update_equal_time_gf_after_flip(s);
					config.measure.add("flip field", 1.0);
				}
				else
				{
					//config.M.undo_ising_flip(s, i, j);
					config.measure.add("flip field", 0.0);
				}
			}
	}
};
