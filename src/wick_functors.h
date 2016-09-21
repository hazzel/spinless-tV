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
				kek += config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second));
			}
		for (auto& a : config.l.bonds("kekule"))
			for (auto& b : config.l.bonds("kekule_2"))
			{
				kek -= 2.*(config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second)));
			}
		for (auto& a : config.l.bonds("kekule_2"))
			for (auto& b : config.l.bonds("kekule_2"))
			{
				kek += config.l.parity(a.first) * config.l.parity(b.first)
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
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first);
			}
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch -= 2.*(et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first));
			}
		for (auto& a : config.l.bonds("chern_2"))
			for (auto& b : config.l.bonds("chern_2"))
			{
				ch += et_gf_t(a.second, a.first) * et_gf_0(b.second, b.first)
					+ td_gf(a.first, b.second) * td_gf(a.second, b.first);
			}
			
		return std::real(ch) / std::pow(config.l.n_bonds(), 2.);
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
		std::complex<double> im = {0., 1.};
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				auto& r_i = config.l.real_space_coord(i);
				auto& r_j = config.l.real_space_coord(j);
				double kdot = K.dot(r_i - r_j);
				if (config.l.sublattice(i) == config.l.sublattice(j))
					sp += std::cos(kdot) * td_gf(i, j)
						+ im * std::sin(kdot) * td_gf(i, j);
				else
					sp += config.l.parity(i) * (im * std::cos(kdot) * td_gf(i, j)
						- std::sin(kdot) * td_gf(i, j));
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
		std::complex<double> im = {0., 1.};
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
						double kdot = K.dot(r_i - r_j - r_m + r_n);
						int sl_i = config.l.sublattice(i);
						int sl_j = config.l.sublattice(j);
						int sl_m = config.l.sublattice(m);
						int sl_n = config.l.sublattice(n);
						std::complex<double> p1, p2;
						if (sl_i == 0 && sl_j == 0)
							p1 = -1.;
						else if (sl_i == 1 && sl_j == 1)
							p1 = 1.;
						else if (sl_i != sl_j)
							p1 = -im;
						if (sl_m == 0 && sl_n == 0)
							p2 = -1.;
						else if (sl_m == 1 && sl_n == 1)
							p2 = 1.;
						else if (sl_m != sl_n)
							p2 = im;
						tp += p1 * p2 * std::exp(im * kdot) * (td_gf(i, m) * td_gf(j, n)
							- td_gf(i, n) * td_gf(j, m));
					}
		return std::real(tp);
	}
};
