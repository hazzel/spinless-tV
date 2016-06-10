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

	void flip_cb_outer(int pv, int pv_min, int pv_max)
	{
		for (auto& b : config.M.get_cb_bonds(pv % 3))
		{
//			int s = rng() * 2;
			int s = 0;
			double p1 = config.M.try_ising_flip(s, b.first, b.second);
			if (rng() < std::abs(p1))
			{
				config.M.buffer_equal_time_gf();
				config.M.update_equal_time_gf_after_flip(s);
				if (config.M.get_partial_vertex(s) == pv_min)
					config.M.partial_advance(s, pv_max);
				else
					config.M.partial_advance(s, pv_min);
				double p2 = config.M.try_ising_flip(s, b.first, b.second);
				if (rng() < std::abs(p2))
				{
					config.M.partial_advance(0, config.M.get_partial_vertex(s));
					config.M.update_equal_time_gf_after_flip(0);
					config.M.flip_spin(0, b);
				}
				else
					config.M.reset_equal_time_gf_to_buffer();
			}
		}
	}

	void flip_cb_inner(int pv)
	{
		for (auto& b : config.M.get_cb_bonds(pv))
		{
//			int s = rng() * 2;
			int s = 0;
			double p = config.M.try_ising_flip(s, b.first, b.second);
			if (rng() < std::abs(p))
			{
				config.M.update_equal_time_gf_after_flip(0);
				config.M.flip_spin(0, b);
			}
		}
	}

	void trigger()
	{
		int m = config.l.n_sites();
		std::vector<std::pair<int, int>> sites(m);
		for (int s = 0; s < 2; ++s)
		{
			config.M.partial_advance(0, 0);
			config.M.partial_advance(1, 0);
			flip_cb_outer(0, 0, 4);
			
			config.M.partial_advance(0, 1);
			config.M.partial_advance(1, 1);
			flip_cb_outer(1, 1, 3);

			config.M.partial_advance(0, 2);
			config.M.partial_advance(1, 2);
			flip_cb_inner(2);

			config.M.partial_advance(0, 0);
			config.M.partial_advance(1, 0);
		}
	}
};
