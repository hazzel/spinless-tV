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
		for (auto& i : config.l.bonds("nearest neighbors"))
			energy += -config.l.parity(i.first) * config.param.t * std::imag(et_gf(i.second, i.first))
				+ config.param.V * std::real(et_gf(i.second, i.first) * et_gf(i.second, i.first)) / 2.;
		for (auto& i : config.l.bonds("d3_bonds"))
			energy += -config.l.parity(i.first) * config.param.tprime * std::imag(et_gf(i.second, i.first));
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
		for (auto& i : config.l.bonds("nearest neighbors"))
			epsilon -= config.l.parity(i.first) * et_gf(i.second, i.first);
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
		for (auto& i : config.l.bonds("chern"))
			chern += et_gf(i.second, i.first) - et_gf(i.first, i.second);
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
		for (auto& i : config.l.bonds("chern"))
			for (auto& j : config.l.bonds("chern"))
			{
				chern2 += et_gf(i.second, i.first) * et_gf(j.second, j.first)
					+ et_gf(j.second, i.first) * et_gf(j.first, i.second)
					- et_gf(i.first, i.second) * et_gf(j.second, j.first)
					- et_gf(j.second, i.second) * et_gf(j.first, i.first)
					- et_gf(i.second, i.first) * et_gf(j.first, j.second)
					- et_gf(j.first, i.first) * et_gf(j.second, i.second)
					+ et_gf(i.first, i.second) * et_gf(j.first, j.second)
					+ et_gf(j.first, i.second) * et_gf(j.second, i.first);
			}
		return std::imag(chern2) / config.l.n_bonds();
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
		/*
		int i = rng() * config.l.n_sites();
		for (int j = 0; j < config.l.n_sites(); ++j)
		{
			double delta_ij = (i==j) ? 1.0 : 0.0;
			for (int k = 0; k < j+1; ++k)
			{
				double delta_ki = (k==i) ? 1.0 : 0.0;
				double delta_kj = (k==j) ? 1.0 : 0.0;
				for (int l = 0; l < k+1; ++l)
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
					M4 += parity * 6. * mat44.determinant();
				}
			}
		}
		for (int l = 0; l < config.l.n_sites(); ++l)
		{
			int j = 0, k = 0;
			double delta_ij = (i==j) ? 1.0 : 0.0;
			double delta_ki = (k==i) ? 1.0 : 0.0;
			double delta_kj = (k==j) ? 1.0 : 0.0;
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
			M4 += parity * (3.*config.l.n_sites()-2.)*mat44.determinant();
		}
		*/
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