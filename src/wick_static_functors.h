#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>
#include <boost/multi_array.hpp>
#include <Eigen/Dense>
#include "measurements.h"
#include "configuration.h"

typedef fast_update<arg_t>::dmatrix_t matrix_t;

struct wick_static_energy
{
	configuration& config;
	Random& rng;

	wick_static_energy(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> energy = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			energy += config.l.parity(a.first) * config.param.t * std::imag(et_gf(a.second, a.first))
				+ config.param.V * std::real(et_gf(a.second, a.first) * et_gf(a.second, a.first)) / 2.;
		for (auto& a : config.l.bonds("d3_bonds"))
			energy += config.l.parity(a.first) * config.param.tprime * std::imag(et_gf(a.second, a.first));
		return std::real(energy);
	}
};

struct wick_static_epsilon
{
	configuration& config;
	Random& rng;

	wick_static_epsilon(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> epsilon = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			epsilon += config.l.parity(a.first) * et_gf(a.second, a.first);
		return std::imag(epsilon) / config.l.n_bonds();
	}
};

struct wick_static_chern
{
	configuration& config;
	Random& rng;

	wick_static_chern(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> chern = 0.;
		for (auto& a : config.l.bonds("chern"))
			chern += et_gf(a.second, a.first) - et_gf(a.first, a.second);
		return std::imag(chern) / config.l.n_bonds();
	}
};

struct wick_static_chern2
{
	configuration& config;
	Random& rng;

	wick_static_chern2(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> chern2 = 0.;
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
			{
				chern2 -= et_gf(a.second, a.first) * et_gf(b.second, b.first)
					+ et_gf(b.second, a.first) * et_gf(b.first, a.second)
					- et_gf(a.first, a.second) * et_gf(b.second, b.first)
					- et_gf(b.second, a.second) * et_gf(b.first, a.first)
					- et_gf(a.second, a.first) * et_gf(b.first, b.second)
					- et_gf(b.first, a.first) * et_gf(b.second, a.second)
					+ et_gf(a.first, a.second) * et_gf(b.first, b.second)
					+ et_gf(b.first, a.second) * et_gf(b.second, a.first);
			}
		return std::real(chern2) / std::pow(config.l.n_bonds(), 2);
	}
};

struct wick_static_chern4
{
	configuration& config;
	Random& rng;
	typedef std::pair<int, int> bond_t;

	wick_static_chern4(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	std::complex<double> calculate_wick_det(const matrix_t& et_gf, Eigen::Matrix4cd& mat44,
		const bond_t& a, const bond_t& b, const bond_t& c, const bond_t& d)
	{
		mat44(0, 0) = et_gf(a.second, a.first);
		mat44(1, 1) = et_gf(b.second, b.first);
		mat44(2, 2) = et_gf(c.second, c.first);
		mat44(3, 3) = et_gf(d.second, d.first);
		
		mat44(0, 1) = et_gf(b.second, a.first);
		mat44(0, 2) = et_gf(c.second, a.first);
		mat44(0, 3) = et_gf(d.second, a.first);
		mat44(1, 2) = et_gf(c.second, b.first);
		mat44(1, 3) = et_gf(d.second, b.first);
		mat44(2, 3) = et_gf(d.second, c.first);
		
		mat44(1, 0) = et_gf(a.second, b.first) - ((a.second==b.first) ? 1.0 : 0.0);
		mat44(2, 0) = et_gf(a.second, c.first) - ((a.second==c.first) ? 1.0 : 0.0);
		mat44(2, 1) = et_gf(b.second, c.first) - ((b.second==c.first) ? 1.0 : 0.0);
		mat44(3, 0) = et_gf(a.second, d.first) - ((a.second==d.first) ? 1.0 : 0.0);
		mat44(3, 1) = et_gf(b.second, d.first) - ((b.second==d.first) ? 1.0 : 0.0);
		mat44(3, 2) = et_gf(c.second, d.first) - ((c.second==d.first) ? 1.0 : 0.0);
		
		return mat44.determinant();
	}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> chern4 = 0.;
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		for (auto& a : config.l.bonds("chern"))
			for (auto& b : config.l.bonds("chern"))
				for (auto& c : config.l.bonds("chern"))
					for (auto& d : config.l.bonds("chern"))
					{
						bond_t a_prime = bond_t{a.second, a.first};
						bond_t b_prime = bond_t{b.second, b.first};
						bond_t c_prime = bond_t{c.second, c.first};
						bond_t d_prime = bond_t{d.second, d.first};
						
						//std::cout << a.first << ", " << a.second << std::endl;
						//std::cout << b.first << ", " << b.second << std::endl;
						//std::cout << c.first << ", " << c.second << std::endl;
						//std::cout << d.first << ", " << d.second << std::endl << std::endl;
						
						chern4 += calculate_wick_det(et_gf, mat44, a, b, c, d);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b, c, d) << std::endl;
						chern4 -= calculate_wick_det(et_gf, mat44, a, b, c, d_prime);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b, c, d_prime) << std::endl;
						chern4 -= calculate_wick_det(et_gf, mat44, a, b, c_prime, d);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b, c_prime, d) << std::endl;
						chern4 += calculate_wick_det(et_gf, mat44, a, b, c_prime, d_prime);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b, c_prime, d_prime) << std::endl;
						
						chern4 -= calculate_wick_det(et_gf, mat44, a, b_prime, c, d);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b_prime, c, d) << std::endl;
						chern4 += calculate_wick_det(et_gf, mat44, a, b_prime, c, d_prime);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b_prime, c, d_prime) << std::endl;
						chern4 += calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d) << std::endl;
						chern4 -= calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d_prime);
						//std::cout << calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d_prime) << std::endl;
						//std::cout << "-----" << std::endl;
					}
		return std::real(2.*chern4) / std::pow(config.l.n_bonds(), 4);
	}
};

// M2(tau) = sum_ij <(n_i(tau) - 1/2)(n_j - 1/2)>
struct wick_static_M2
{
	configuration& config;
	Random& rng;

	wick_static_M2(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> M2 = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
				M2 += config.l.parity(i) * config.l.parity(j) * std::real(et_gf(i, j)
						* et_gf(i, j));
		return std::real(M2) / std::pow(config.l.n_sites(), 2.);
	}
};

// M4(tau) = sum_ij sum_kl <(n_i(tau) - 1/2)(n_j - 1/2) (n_k(tau) - 1/2)(n_l - 1/2)>
struct wick_static_M4
{
	configuration& config;
	Random& rng;

	wick_static_M4(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> M4 = 0.;
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				double delta_ij = (i==j) ? 1.0 : 0.0;
				for (int k = 0; k < config.l.n_sites(); ++k)
				{
					double delta_ki = (k==i) ? 1.0 : 0.0;
					double delta_kj = (k==j) ? 1.0 : 0.0;
					for (int l = 0; l < config.l.n_sites(); ++l)
					{
						double delta_li = (l==i) ? 1.0 : 0.0;
						double delta_lj = (l==j) ? 1.0 : 0.0;
						double delta_lk = (l==k) ? 1.0 : 0.0;
						
						mat44(0, 1) = et_gf(i, j);
						mat44(0, 2) = et_gf(i, k);
						mat44(0, 3) = et_gf(i, l);
						mat44(1, 2) = et_gf(j, k);
						mat44(1, 3) = et_gf(j, l);
						mat44(2, 3) = et_gf(k, l);
						
						mat44(1, 0) = et_gf(j, i) - delta_ij;
						mat44(2, 0) = et_gf(k, i) - delta_ki;
						mat44(2, 1) = et_gf(k, j) - delta_kj;
						mat44(3, 0) = et_gf(l, i) - delta_li;
						mat44(3, 1) = et_gf(l, j) - delta_lj;
						mat44(3, 2) = et_gf(l, k) - delta_lk;
						
						double parity = config.l.parity(i) * config.l.parity(j) * config.l.parity(k) * config.l.parity(l);
						M4 += parity * mat44.determinant();
					}
				}
			}
		return std::real(M4) / std::pow(config.l.n_sites(), 4.);
	}
};