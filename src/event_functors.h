#pragma once
#include <map>
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
		for (int i = 0; i < config->params.n_tau_slices; ++i)
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
