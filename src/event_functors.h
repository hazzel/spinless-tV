#pragma once
#include <map>
#include <vector>
#include "measurements.h"
#include "configuration.h"
#include "wick_base.h"
#include "wick_functors.h"

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
		int bond_type = (pv < 3) ? pv : 4-pv;
		for (auto& b : config.M.get_cb_bonds(bond_type))
		{
			if (b.first > b.second) break;
//			int s = rng() * 2;
			int s = 0;
			double p_0 = config.M.try_ising_flip(s, b.first, b.second);
			if (rng() < std::abs(p_0))
			{
				config.M.buffer_equal_time_gf();
				config.M.update_equal_time_gf_after_flip(s);
				if (config.M.get_partial_vertex(s) == pv_min)
				{
					// Perform partial advance with flipped spin
					config.M.flip_spin(s, b);
					config.M.partial_advance(s, pv_max);
					// Flip back
					config.M.flip_spin(s, b);
				}
				else
					config.M.partial_advance(s, pv_min);
				p_0 = config.M.try_ising_flip(s, b.first, b.second);
				if (rng() < std::abs(p_0))
				{
					config.M.update_equal_time_gf_after_flip(s);
					config.M.flip_spin(s, b);
				}
				else
					config.M.reset_equal_time_gf_to_buffer();
			}
		}
	}

	void flip_cb_inner(int pv)
	{
		int bond_type = (pv < 3) ? pv : 4-pv;
		for (auto& b : config.M.get_cb_bonds(bond_type))
		{
			if (b.first > b.second) break;
//			int s = rng() * 2;
			int s = 0;
			double p_0 = config.M.try_ising_flip(s, b.first, b.second);
			if (rng() < std::abs(p_0))
			{
				config.M.update_equal_time_gf_after_flip(s);
				config.M.flip_spin(s, b);
			}
		}
	}

	void trigger()
	{
		int m = config.l.n_sites();
		std::vector<std::pair<int, int>> sites(m);
	
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
};

struct event_dynamic_measurement
{
	typedef fast_update<qr_stabilizer>::dmatrix_t matrix_t;
	typedef std::function<double(const matrix_t&, const matrix_t&, Random&,
		const lattice&, const parameters&)> function_t;

	configuration& config;
	Random& rng;
	std::vector<double> time_grid;
	std::vector<std::vector<double>> dyn_tau;
	std::vector<double> dyn_tau_avg;
	std::vector<wick_base<matrix_t>> obs;
	std::vector<std::string> names;

	event_dynamic_measurement(configuration& config_, Random& rng_,
		int n_prebin, const std::vector<std::string>& observables)
		: config(config_), rng(rng_)
	{
		time_grid.resize(2 * config.param.n_discrete_tau + 1);
		for (int t = 0; t < time_grid.size(); ++t)
			time_grid[t] = static_cast<double>(t) / static_cast<double>(2*config.
				param.n_discrete_tau) * config.param.beta;
		for (int i = 0; i < observables.size(); ++i)
		{
			dyn_tau.push_back(std::vector<double>(2 * config.param.n_discrete_tau
				+ 1, 0.));
			
			if (observables[i] == "M2")
				add_wick(wick_M2{config, rng});
			else if (observables[i] == "kekule")
				add_wick(wick_kekule{config, rng});
			else if (observables[i] == "epsilon")
				add_wick(wick_epsilon{config, rng});
			else if (observables[i] == "chern")
				add_wick(wick_chern{config, rng});
			else if (observables[i] == "sp")
				add_wick(wick_sp{config, rng});
			else if (observables[i] == "tp")
				add_wick(wick_tp{config, rng});
			
			names.push_back("dyn_"+observables[i]);
			if (config.param.n_discrete_tau > 0)
				config.measure.add_vectorobservable("dyn_"+observables[i]+"_tau",
					2*config.param.n_discrete_tau + 1, n_prebin);
		}
		dyn_tau_avg.resize(config.param.n_discrete_tau + 1);
	}
	
	template<typename T>
	void add_wick(T&& functor)
	{
		obs.push_back(wick_base<matrix_t>(std::forward<T>(functor)));
	}

	void trigger()
	{
		for (int i = 0; i < dyn_tau.size(); ++i)
			std::fill(dyn_tau[i].begin(), dyn_tau[i].end(), 0.);
		//config.M.measure_dynamical_observable(config.param.n_matsubara, time_grid,
		//	dyn_mat, dyn_tau, obs);
		//config.M.hirsch_fye_measurement(config.param.n_matsubara, time_grid,
		//	dyn_mat, dyn_tau, obs);

		for (int i = 0; i < dyn_tau.size(); ++i)
			if (config.param.n_discrete_tau > 0)
				config.measure.add(names[i]+"_tau", dyn_tau[i]);
	}
};
