#pragma once
#include <map>
#include <vector>
#include "measurements.h"
#include "configuration.h"

struct event_rebuild
{
	configuration* config;
	measurements& measure;

	void trigger()
	{
		config->M.rebuild();
	}
};

struct event_build
{
	configuration* config;
	Random& rng;

	void trigger()
	{
		std::vector<arg_t> initial_vertices;
		for (int i = 1; i <= config->params.n_tau_slices; ++i)
		{
			std::map<std::pair<int, int>, double> sigma;
			for (int j = 0; j < config->l.n_sites(); ++j)
				for (int k = j; k < config->l.n_sites(); ++k)
					if (config->l.distance(j, k) == 1)
						sigma[{j, k}] = static_cast<int>(rng()*2.)*2-1;
			initial_vertices.push_back({i, sigma});
		}
		config->M.build(initial_vertices);
	}
};

struct event_flip_all
{
	configuration* config;
	Random& rng;

	void trigger()
	{
		int m = config->l.n_sites();
		std::vector<std::pair<int, int>> sites(m);
		for (int n = 0; n < config->l.n_sites(); ++n)
		{
			for (int k = 0; k < m; ++k)
			{
				int i = rng() * config->l.n_sites();
				int j = config->l.neighbors(i, "nearest neighbors")[rng() * 
				config->l.neighbors(i, "nearest neighbors").size()];
				sites[k] = {i, j};
			}
			double p = config->M.try_ising_flip(sites);
			if (rng() < p)
			{
				config->M.update_equal_time_gf_after_flip();
				config->measure.add("flip field", 1.0);
			}
			else
			{
				config->M.undo_ising_flip(sites);
				config->measure.add("flip field", 0.0);
			}
		}
	}
};
