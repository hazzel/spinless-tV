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
		std::complex<double> M2 = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
				//M2 += config.l.parity(i) * config.l.parity(j) * td_gf(i, j)
				//	* td_gf(i, j);
				M2 += config.l.parity(i) * config.l.parity(j) * td_gf(j, i)
					* td_gf(j, i);
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
		/*
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
		*/
		
		std::complex<double> kek = 0.;
		std::complex<double> im = {0., 1.};
		auto& K = config.l.symmetry_point("K");
		for (auto& a : config.l.bonds("kekule"))
			for (auto& b : config.l.bonds("kekule"))
			{
				auto& r_i = config.l.real_space_coord(a.first);
				auto& r_j = config.l.real_space_coord(a.second);
				auto& r_m = config.l.real_space_coord(b.first);
				auto& r_n = config.l.real_space_coord(b.second);
				std::complex<double> kdot = std::exp(im * K.dot(r_i - r_j + r_m - r_n));
				
				kek += kdot * config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second));
			}
		for (auto& a : config.l.bonds("kekule"))
			for (auto& b : config.l.bonds("kekule_2"))
			{
				auto& r_i = config.l.real_space_coord(a.first);
				auto& r_j = config.l.real_space_coord(a.second);
				auto& r_m = config.l.real_space_coord(b.first);
				auto& r_n = config.l.real_space_coord(b.second);
				std::complex<double> kdot = std::exp(im * K.dot(r_i - r_j + r_m - r_n));
				
				kek -= kdot * 2.*(config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second)));
			}
		for (auto& a : config.l.bonds("kekule_2"))
			for (auto& b : config.l.bonds("kekule_2"))
			{
				auto& r_i = config.l.real_space_coord(a.first);
				auto& r_j = config.l.real_space_coord(a.second);
				auto& r_m = config.l.real_space_coord(b.first);
				auto& r_n = config.l.real_space_coord(b.second);
				std::complex<double> kdot = std::exp(im * K.dot(r_i - r_j + r_m - r_n));
				
				kek += kdot * config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second));
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
		//for (auto& a : config.l.bonds("kekule"))
		//	for (auto& b : config.l.bonds("kekule"))
			{
				/*
				ep += config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(a.first, b.first) * td_gf(a.second, b.second));
				*/
				ep += config.l.parity(a.first) * config.l.parity(b.first)
					* (et_gf_t(a.second, a.first) * et_gf_0(b.first, b.second)
					+ td_gf(b.first, a.first) * td_gf(b.second, a.second));
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
		std::complex<double> im = {0., 1.};
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				auto& r_i = config.l.real_space_coord(i);
				auto& r_j = config.l.real_space_coord(j);
				double kdot = K.dot(r_i - r_j);
				
				if (config.l.sublattice(i) == config.l.sublattice(j))
					sp += std::cos(kdot) * td_gf(j, i) + im * std::sin(kdot) * td_gf(j, i);
				else
					sp += config.l.parity(i) * (-im * std::cos(kdot) * td_gf(j, i)
						+ std::sin(kdot) * td_gf(j, i));
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
