#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>
#include <boost/multi_array.hpp>
#include "measurements.h"
#include "configuration.h"

typedef fast_update<arg_t>::dmatrix_t matrix_t;

// M2(tau) = sum_ij <(n_i(tau) - 1/2)(n_j - 1/2)>
struct wick_M2
{
	configuration& config;
	Random& rng;

	wick_M2(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		double M2 = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				/*
				M2 += config.l.parity(i) * config.l.parity(j)
					* std::real((1. - et_gf_t(i, i)) * (1. - et_gf_0(j, j))
					+ config.l.parity(i) * config.l.parity(j) * td_gf(i, j) * td_gf(i, j)
					- (et_gf_t(i, i) + et_gf_0(j, j))/2. + 1./4.);
				*/
				M2 += std::real(td_gf(j, i)) * std::real(td_gf(j, i));
			}
		return M2 / std::pow(config.l.n_sites(), 2.);
	}
};

// kekule(tau) = sum_{kekule} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_kekule
{
	configuration& config;
	Random& rng;

	wick_kekule(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		double kek = 0.;
		std::array<const std::vector<std::pair<int, int>>*, 3> kek_bonds =
			{&config.l.bonds("kekule"), &config.l.bonds("kekule_2"),
			&config.l.bonds("kekule_3")};
		std::array<double, 3> factors = {-1., -1., 2.};
		for (int i = 0; i < kek_bonds.size(); ++i)
			for (int m = 0; m < kek_bonds.size(); ++m)
				for (int j = 0; j < kek_bonds[i]->size(); ++j)
					for (int n = 0; n < kek_bonds[m]->size(); ++n)
					{
						auto& a = (*kek_bonds[i])[j];
						auto& b = (*kek_bonds[m])[n];
						
						kek += factors[i] * factors[m]
								* (std::real(et_gf_t(a.second, a.first)) * std::real(et_gf_0(b.first, b.second))
								+ config.l.parity(a.first) * config.l.parity(b.first) * std::real(td_gf(b.first, a.first)) * std::real(td_gf(b.second, a.second)));
					}
		return kek / std::pow(config.l.n_bonds(), 2.);
	}
};

// ep(tau) = sum_{<ij>,<mn>} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_epsilon
{
	configuration& config;
	Random& rng;

	wick_epsilon(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		double ep = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			for (auto& b : config.l.bonds("nearest neighbors"))
			{
				ep += std::real(et_gf_t(a.second, a.first)) * std::real(et_gf_0(b.first, b.second))
					+ config.l.parity(a.first) * config.l.parity(b.first) * std::real(td_gf(b.first, a.first)) * std::real(td_gf(b.second, a.second));
			}
		return ep / std::pow(config.l.n_bonds(), 2.);
	}
};

struct wick_epsilon_as
{
	configuration& config;
	Random& rng;

	wick_epsilon_as(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> ep = 0.;
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		
		for (int i = 0; i < bonds.size(); ++i)
			for (int j = 0; j < bonds[i]->size(); ++j)
				for (int m = 0; m < bonds.size(); ++m)
					for (int n = 0; n < bonds[m]->size(); ++n)
					{
						auto& a = (*bonds[i])[j];
						auto& b = (*bonds[m])[n];
						
						ep += et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(b.first, a.first) * td_gf(b.second, a.second);
						
						ep -= et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(b.first, a.second) * td_gf(b.second, a.first);
						
						ep -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(b.second, a.first) * td_gf(b.first, a.second);
						
						ep += et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(b.second, a.second) * td_gf(b.first, a.first);
					}
		return std::real(ep) / std::pow(config.l.n_bonds(), 2.);
	}
};

struct wick_cdw_s
{
	configuration& config;
	Random& rng;

	wick_cdw_s(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> ch = 0.;
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.first) * td_gf(b.first, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.second) * td_gf(b.first, a.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.first) * td_gf(b.first, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.second) * td_gf(b.first, a.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.first) * td_gf(b.first, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.second) * td_gf(b.first, a.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.first) * td_gf(b.first, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.second) * td_gf(b.first, a.first)
					+ et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
	}
};

// chern(tau) = sum_{chern} <c_i^dag(tau) c_j(tau) c_n^dag c_m>
struct wick_chern
{
	configuration& config;
	Random& rng;

	wick_chern(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> ch = 0.;
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.first) * td_gf(b.first, a.second)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.second) * td_gf(b.first, a.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.first) * td_gf(b.first, a.second)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.second) * td_gf(b.first, a.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.first) * td_gf(b.first, a.second)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.second) * td_gf(b.first, a.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch -= et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(b.second, a.first) * td_gf(b.first, a.second)
					- et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
					- td_gf(b.second, a.second) * td_gf(b.first, a.first)
					- et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					- td_gf(b.first, a.first) * td_gf(b.second, a.second)
					+ et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.second) * td_gf(b.second, a.first);
			}
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
	}
};

struct wick_gamma_mod
{
	configuration& config;
	Random& rng;

	wick_gamma_mod(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> gm = 0.;
		double pi = 4. * std::atan(1.);
		
		std::vector<const std::vector<std::pair<int, int>>*> bonds =
			{&config.l.bonds("nn_bond_1"), &config.l.bonds("nn_bond_2"),
			&config.l.bonds("nn_bond_3")};
		std::vector<double> phases = {2.*std::sin(0. * pi), 2.*std::sin(2./3. * pi), 2.*std::sin(4./3. * pi)};
		
		for (int i = 0; i < bonds.size(); ++i)
			for (int j = 0; j < bonds[i]->size(); ++j)
				for (int m = 0; m < bonds.size(); ++m)
					for (int n = 0; n < bonds[m]->size(); ++n)
					{
						auto& a = (*bonds[i])[j];
						auto& b = (*bonds[m])[n];
						
						gm += phases[i] * phases[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.first) * config.l.parity(b.first) * td_gf(b.first, a.first) * td_gf(b.second, a.second));
						
						gm -= phases[i] * phases[m]
							* (et_gf_t(a.first, a.second) * et_gf_0(b.first, b.second)
							+ config.l.parity(a.second) * config.l.parity(b.first) * td_gf(b.first, a.second) * td_gf(b.second, a.first));
						
						gm -= phases[i] * phases[m]
							* (et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.first) * config.l.parity(b.second) * td_gf(b.second, a.first) * td_gf(b.first, a.second));
						
						gm += phases[i] * phases[m]
							* (et_gf_t(a.first, a.second) * et_gf_0(b.second, b.first)
							+ config.l.parity(a.second) * config.l.parity(b.second) * td_gf(b.second, a.second) * td_gf(b.first, a.first));
					}
		return std::real(gm) / std::pow(config.l.n_bonds(), 2.);
	}
};

// sp(tau) = sum_ij e^{-i K (r_i - r_j)} <c_i(tau) c_j^dag>
struct wick_sp
{
	configuration& config;
	Random& rng;

	wick_sp(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		double sp = 0.;
		auto& K = config.l.symmetry_point("K");
		std::complex<double> im = {0., 1.};
		for (int i = 0; i < config.l.n_sites(); ++i)
		{
			auto& r_i = config.l.real_space_coord(i);
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				auto& r_j = config.l.real_space_coord(j);
				double kdot = K.dot(r_i - r_j);
			
				sp += std::cos(kdot) * std::real(td_gf(i, j));
			}
		}
		return sp;
	}
};

// tp(tau) = sum_ijmn e^{-i K (r_i - r_j + r_m - r_n)}
			//		<c_i(tau) c_j(tau) c_n^dag c_m^dag>
struct wick_tp
{
	configuration& config;
	Random& rng;
	boost::multi_array<std::complex<double>, 3> expKdot;

	wick_tp(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{
		int N = config.l.n_sites();
		expKdot.resize(boost::extents[N][N][N]);
		auto& K = config.l.symmetry_point("K");
		std::complex<double> im = {0., 1.};
		int i = 0;
			for (int j = 0; j < config.l.n_sites(); ++j)
				for (int m = 0; m < config.l.n_sites(); ++m)
					for (int n = 0; n < config.l.n_sites(); ++n)
					{
						int sl_i = config.l.sublattice(i);
						int sl_j = config.l.sublattice(j);
						int sl_m = config.l.sublattice(m);
						int sl_n = config.l.sublattice(n);
						std::complex<double> p1 = 1., p2 = 1., p3 = 1., p4 = 1.;
						if (sl_i == 0)
							p1 = -im * config.l.parity(i);
						if (sl_j == 0)
							p2 = -im * config.l.parity(j);
						if (sl_m == 0)
							p3 = im * config.l.parity(m);
						if (sl_n == 0)
							p4 = im * config.l.parity(n);
						auto& r_i = config.l.real_space_coord(i);
						auto& r_j = config.l.real_space_coord(j);
						auto& r_m = config.l.real_space_coord(m);
						auto& r_n = config.l.real_space_coord(n);
						double kdot = K.dot(r_i - r_j - r_m + r_n);
						expKdot[j][m][n] = p1 * p2 * p3 * p4 * std::exp(im * kdot);
					}
	}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> tp = 0.;
		std::complex<double> im = {0., 1.};
		//for (int i = 0; i < config.l.n_sites(); ++i)
		int i = 0;
			for (int j = 0; j < config.l.n_sites(); ++j)
				for (int m = 0; m < config.l.n_sites(); ++m)
					for (int n = 0; n < config.l.n_sites(); ++n)
						tp += expKdot[j][m][n] * (td_gf(m, i) * td_gf(n, j)
							- td_gf(n, i) * td_gf(m, j));
		return std::real(tp) * config.l.n_sites();
	}
};
