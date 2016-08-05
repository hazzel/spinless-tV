#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>
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
		std::complex<double> M2 = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
				M2 += config.l.parity(i) * config.l.parity(j) * td_gf(i, j)
					* td_gf(i, j);
		return std::real(M2) / std::pow(config.l.n_sites(), 2.);
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
		std::complex<double> kek = 0.;
		for (auto& a : config.l.bonds("kekule"))
			for (auto& b : config.l.bonds("kekule"))
			{
				kek -= config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second));
			}
		return std::real(kek) / std::pow(config.l.n_bonds(), 2.);
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
		std::complex<double> ep = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			for (auto& b : config.l.bonds("nearest neighbors"))
			{
				ep -= config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second));
			}
		return std::real(ep) / std::pow(config.l.n_bonds(), 2.);
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
				ch += (et_gf_t(a.second, a.first) - et_gf_t(a.first, a.second))
					* (et_gf_0(b.second, b.first) - et_gf_0(b.first, b.second))
					+ config.l.parity(a.first) * config.l.parity(b.second)
					* td_gf(a.first, b.second) * td_gf(a.second, b.first)
					- config.l.parity(a.first) * config.l.parity(b.first)
					* td_gf(a.first, b.first) * td_gf(a.second, b.second)
					- config.l.parity(a.second) * config.l.parity(b.second)
					* td_gf(a.second, b.second) * td_gf(a.first, b.first)
					+ config.l.parity(a.second) * config.l.parity(b.first)
					* td_gf(a.second, b.first) * td_gf(a.first, b.first);
			}
		return -std::real(ch);
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
		std::complex<double> sp = 0.;
		auto& K = config.l.symmetry_point("K");
		double pi = 4.*std::atan(1.);
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				auto& r_i = config.l.real_space_coord(i);
				auto& r_j = config.l.real_space_coord(j);
				sp +=	std::cos(K.dot(r_i - r_j)) * td_gf(i, j);
			}
		return std::real(sp);
	}
};

// tp(tau) = sum_ijmn e^{-i K (r_i - r_j + r_m - r_n)}
			//		<c_i(tau) c_j(tau) c_n^dag c_m^dag>
struct wick_tp
{
	configuration& config;
	Random& rng;

	wick_tp(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
		const matrix_t& td_gf)
	{
		std::complex<double> tp = 0.;
		double pi = 4.*std::atan(1.);
		Eigen::Vector2d K(2.*pi/9., 2.*pi/9.*(2.-1./std::sqrt(3.)));
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
				for (int m = 0; m < config.l.n_sites(); ++m)
					for (int n = 0; n < config.l.n_sites(); ++n)
					{
						auto& r_i = config.l.real_space_coord(i);
						auto& r_j = config.l.real_space_coord(j);
						auto& r_m = config.l.real_space_coord(m);
						auto& r_n = config.l.real_space_coord(n);
						//tp += config.trig_spline.cos(K.dot(r_j - r_i + r_m - r_n))
						tp += std::cos(K.dot(r_i - r_j - r_m + r_n))
							* (td_gf(i, m) * td_gf(j, n)
							- td_gf(i, n) * td_gf(j, m));
					}
		return std::real(tp);
	}
};
