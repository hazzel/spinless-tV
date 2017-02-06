#pragma once
#include <map>
#include <vector>
#include "measurements.h"
#include "configuration.h"
#include "fast_update.h"
#include "wick_base.h"
#include "wick_functors.h"
#include "wick_static_base.h"
#include "wick_static_functors.h"

struct event_build
{
	configuration& config;
	Random& rng;

	void trigger()
	{
		std::vector<arg_t> initial_vertices(
			config.param.n_tau_slices, arg_t(config.l.n_bonds() / sizeof(int) / 8 + 1));
		for (int t = 0; t < config.param.n_tau_slices; ++t)
		{
			for (int j = 0; j < config.l.n_sites(); ++j)
				for (int k = j; k < config.l.n_sites(); ++k)
					if (config.l.distance(j, k) == 1)
						if (rng() < 0.5)
							initial_vertices[t].set(config.M.bond_index(j, k));
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
			if (b.first > b.second) continue;
			std::complex<double> p_0 = config.M.try_ising_flip(b.first, b.second);
			if (rng() < std::abs(p_0))
			{
				config.M.buffer_equal_time_gf();
				config.M.update_equal_time_gf_after_flip();
				if (config.M.get_partial_vertex() == pv_min)
				{
					// Perform partial advance with flipped spin
					config.M.flip_spin(b);
					config.M.partial_advance(pv_max);
					// Flip back
					config.M.flip_spin(b);
				}
				else
					config.M.partial_advance(pv_min);
				p_0 = config.M.try_ising_flip(b.first, b.second);
				if (rng() < std::abs(p_0))
				{
					config.M.update_equal_time_gf_after_flip();
					config.M.flip_spin(b);
				}
				else
					config.M.reset_equal_time_gf_to_buffer();
			}
		}
	}

	void flip_cb_inner(int pv)
	{
		int bond_type = (pv < 3) ? pv : 4-pv;
		int cnt = 0;
		for (auto& b : config.M.get_cb_bonds(bond_type))
		{
			if (b.first > b.second) continue;
			int s = 0;
			std::complex<double> p_0 = config.M.try_ising_flip(b.first, b.second);
			if (config.param.mu != 0)
				config.sign_phase *= std::exp(std::complex<double>(0, std::arg(p_0)));
			if (rng() < std::abs(p_0))
			{
				config.M.update_equal_time_gf_after_flip();
				config.M.flip_spin(b);
			}
		}
	}

	void trigger()
	{
		if (config.param.V > 0.)
		{
			/*
			config.M.prepare_flip();
			config.M.partial_advance(0);
			flip_cb_outer(0, 0, 4);
				
			config.M.partial_advance(1);
			flip_cb_outer(1, 1, 3);

			config.M.partial_advance(2);
			flip_cb_inner(2);

			config.M.partial_advance(0);
			config.M.prepare_measurement();
			*/
			
			if (config.param.tprime > 0. || config.param.tprime < 0.)
				config.M.prepare_flip();
			
			for (int bt = 0; bt < config.M.n_cb_bonds(); ++bt)
			{
				config.M.partial_advance(bt);
				flip_cb_inner(bt);
			}

			config.M.partial_advance(0);
			if (config.param.tprime > 0. || config.param.tprime < 0.)
				config.M.prepare_measurement();
		}
	}
};

struct event_static_measurement
{
	typedef fast_update<qr_stabilizer>::dmatrix_t matrix_t;

	configuration& config;
	Random& rng;
	std::vector<double> values;
	std::vector<wick_static_base<matrix_t>> obs;
	std::vector<std::string> names;

	event_static_measurement(configuration& config_, Random& rng_,
		int n_prebin, const std::vector<std::string>& observables)
		: config(config_), rng(rng_)
	{
		obs.reserve(10);
		for (int i = 0; i < observables.size(); ++i)
		{
			values.push_back(0.);
			
			if (observables[i] == "energy")
				add_wick(wick_static_energy{config, rng});
			else if (observables[i] == "M2")
				add_wick(wick_static_M2{config, rng});
			else if (observables[i] == "M4")
				add_wick(wick_static_M4{config, rng});
			else if (observables[i] == "epsilon")
				add_wick(wick_static_epsilon{config, rng});
			else if (observables[i] == "chern")
				add_wick(wick_static_chern{config, rng});
			else if (observables[i] == "chern2")
				add_wick(wick_static_chern2{config, rng});
			else if (observables[i] == "chern4")
				add_wick(wick_static_chern4{config, rng});
			
			names.push_back(observables[i]);
		}
	}
	
	template<typename T>
	void add_wick(T&& functor)
	{
		obs.push_back(wick_static_base<matrix_t>(std::forward<T>(functor)));
	}

	void trigger()
	{
		std::fill(values.begin(), values.end(), 0.);
		config.M.measure_static_observable(values, obs);

		for (int i = 0; i < values.size(); ++i)
			config.measure.add(names[i], values[i]);
	}
};

struct event_dynamic_measurement
{
	typedef fast_update<qr_stabilizer>::dmatrix_t matrix_t;

	configuration& config;
	Random& rng;
	std::vector<std::vector<double>> dyn_tau;
	std::vector<wick_base<matrix_t>> obs;
	std::vector<std::string> names;

	event_dynamic_measurement(configuration& config_, Random& rng_,
		int n_prebin, const std::vector<std::string>& observables)
		: config(config_), rng(rng_)
	{
		obs.reserve(10);
		for (int i = 0; i < observables.size(); ++i)
		{
			dyn_tau.push_back(std::vector<double>(config.param.n_discrete_tau+1,
				0.));
			
			if (observables[i] == "M2")
				add_wick(wick_M2{config, rng});
			else if (observables[i] == "kekule")
				add_wick(wick_kekule{config, rng});
			else if (observables[i] == "epsilon")
				add_wick(wick_epsilon{config, rng});
			else if (observables[i] == "chern")
				add_wick(wick_chern{config, rng});
			else if (observables[i] == "gamma_mod")
				add_wick(wick_gamma_mod{config, rng});
			else if (observables[i] == "sp")
				add_wick(wick_sp{config, rng});
			else if (observables[i] == "tp")
				add_wick(wick_tp{config, rng});
			
			names.push_back("dyn_"+observables[i]);
		}
	}
	
	template<typename T>
	void add_wick(T&& functor)
	{
		obs.push_back(wick_base<matrix_t>(std::forward<T>(functor)));
	}

	void trigger()
	{
		if (config.param.n_discrete_tau == 0)
			return;
		for (int i = 0; i < dyn_tau.size(); ++i)
			std::fill(dyn_tau[i].begin(), dyn_tau[i].end(), 0.);
		config.M.measure_dynamical_observable(dyn_tau, obs);

		for (int i = 0; i < dyn_tau.size(); ++i)
			if (config.param.n_discrete_tau > 0)
				config.measure.add(names[i]+"_tau", dyn_tau[i]);
	}
};
