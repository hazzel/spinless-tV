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
#include "fast_update.h"
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
		double energy = 0.;
			for (auto& a : config.l.bonds("nearest neighbors"))
				energy += config.param.t * std::real(et_gf(a.second, a.first));
			for (auto& a : config.l.bonds("nearest neighbors"))
				energy += config.param.V * std::real((1. - et_gf(a.first, a.first)) * (1. - et_gf(a.second, a.second))
					- et_gf(a.second, a.first) * et_gf(a.first, a.second) - (et_gf(a.first, a.first) + et_gf(a.second, a.second))/2. + 1./4.)/2.;
			for (auto& a : config.l.bonds("d3_bonds"))
				energy += config.param.tprime
					* std::real(et_gf(a.second, a.first));
			for (int i = 0; i < config.l.n_sites(); ++i)
				energy += -config.l.parity(i) * config.param.stag_mu
					* std::real(et_gf(i, i));
		return energy;
	}
};

struct wick_static_h_t
{
	configuration& config;
	Random& rng;

	wick_static_h_t(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		double energy = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			energy += config.param.t * std::real(et_gf(a.second, a.first));
		for (auto& a : config.l.bonds("d3_bonds"))
			energy += config.param.tprime
				* std::real(et_gf(a.second, a.first));
		return energy;
	}
};

struct wick_static_h_v
{
	configuration& config;
	Random& rng;

	wick_static_h_v(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		double energy = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			energy += config.param.V * std::real((1. - et_gf(a.first, a.first)) * (1. - et_gf(a.second, a.second))
				- et_gf(a.second, a.first) * et_gf(a.first, a.second) - (et_gf(a.first, a.first) + et_gf(a.second, a.second))/2. + 1./4.)/2.;
		return energy;
	}
};

struct wick_static_h_mu
{
	configuration& config;
	Random& rng;

	wick_static_h_mu(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		double energy = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			energy += -config.l.parity(i) * config.param.stag_mu
				* std::real(et_gf(i, i));
		return energy;
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
		double epsilon = 0.;
		for (auto& a : config.l.bonds("nearest neighbors"))
			epsilon += std::real(et_gf(a.second, a.first));
		return epsilon / config.l.n_bonds();
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
		double chern = 0.;
		for (auto& a : config.l.bonds("chern"))
			chern += std::imag(et_gf(a.second, a.first) - et_gf(a.first, a.second));
		return chern / config.l.n_bonds();
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
	std::vector<int> non_zero_terms;
	std::vector<std::pair<double, int>> unique_values;
	std::vector<std::array<int, 4>> unique_bonds;
	bool initialzed = false;

	wick_static_chern4(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double calculate_wick_det(const matrix_t& et_gf, Eigen::Matrix4cd& mat44,
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
		
		mat44(1, 0) = et_gf(a.second, b.first) - ((a.second==b.first) ? 1. : 0.);
		mat44(2, 0) = et_gf(a.second, c.first) - ((a.second==c.first) ? 1. : 0.);
		mat44(2, 1) = et_gf(b.second, c.first) - ((b.second==c.first) ? 1. : 0.);
		mat44(3, 0) = et_gf(a.second, d.first) - ((a.second==d.first) ? 1. : 0.);
		mat44(3, 1) = et_gf(b.second, d.first) - ((b.second==d.first) ? 1. : 0.);
		mat44(3, 2) = et_gf(c.second, d.first) - ((c.second==d.first) ? 1. : 0.);
		
		return std::real(mat44.determinant());
	}
	
	void init(const matrix_t& et_gf)
	{
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		int n = config.l.bonds("chern").size();
		int cnt = 0;
		for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		for (int k = 0; k < n; ++k)
		for (int l = 0; l < n; ++l)
		{
			double done = static_cast<double>(i*n*n*n + j*n*n + k*n + l)
				/ static_cast<double>(n*n*n*n);
			//if (k == 0 && l == 0)
			//	std::cout << "Progress: " << done << std::endl;

			const bond_t& a = config.l.bonds("chern")[i];
			const bond_t& b = config.l.bonds("chern")[j];
			const bond_t& c = config.l.bonds("chern")[k];
			const bond_t& d = config.l.bonds("chern")[l];
			bond_t a_prime = bond_t{a.second, a.first};
			bond_t b_prime = bond_t{b.second, b.first};
			bond_t c_prime = bond_t{c.second, c.first};
			bond_t d_prime = bond_t{d.second, d.first};
			
			if (i == j && k == l)
				++cnt;
			else if (i == k && j == l)
				++cnt;
			else if (i == l && j == k)
				++cnt;
			else if (i == j && i == k && i != l)
				++cnt;
			else if (i == j && i == l && i != k)
				++cnt;
			else if(i == k && i == l && i != j)
				++cnt;
			else if(j == k && j == l && i != j)
				++cnt;

			bool print = false;
			if (i == j && k != l && i != l && i != k)
				print = true;
			if (print)
			{
			std::cout << "bonds: " << std::endl;
			std::cout << a.first << ", " << a.second << std::endl;
			std::cout << b.first << ", " << b.second << std::endl;
			std::cout << c.first << ", " << c.second << std::endl;
			std::cout << d.first << ", " << d.second << std::endl;
			}

			int mask = 0;
			double value = 0.;
			double w = calculate_wick_det(et_gf, mat44, a, b, c, d);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 1;
				if (print)
				std::cout << "mask 1" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b, c, d_prime);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 2;
				if (print)
				std::cout << "mask 2" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b, c_prime, d);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 4;
				if (print)
				std::cout << "mask 4" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b, c_prime, d_prime);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 8;
				if (print)
				std::cout << "mask 8" << std::endl;
			}
						
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c, d);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 16;
				if (print)
				std::cout << "mask 16" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c, d_prime);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 32;
				if (print)
				std::cout << "mask 32" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 64;
				if (print)
				std::cout << "mask 64" << std::endl;
			}
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d_prime);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
			{
				mask |= 128;
				if (print)
				std::cout << "mask 128" << std::endl;
			}
			non_zero_terms.push_back(mask);
			if (print)
			std::cout << "------" << std::endl;

			if (std::abs(value) > std::pow(10., -14))
			{
				bool found = false;
				for (auto& v : unique_values)
				{
					if (std::abs(v.first - value) < std::pow(10., -14))
					{
						++v.second;
						found = true;
						break;
					}
				}
				if (!found)
				{
					unique_values.push_back({value, 1});
					unique_bonds.push_back({i, j, k, l});
				}
			}
		}
		initialzed = true;
		
		std::cout << "chern4: " << unique_bonds.size() << " of "
			<< std::pow(n, 4) << std::endl;
		std::cout << "got: " << cnt << " out of " << n*n*n*n << std::endl;
		/*
		for (int b = 0; b < unique_bonds.size(); ++b)
		{
			int i = unique_bonds[b][0], j = unique_bonds[b][1],
				k = unique_bonds[b][2], l = unique_bonds[b][3];
			std::cout << "bond: (" << i << ", " << j << ", " << k << ", " << l
				<< "), value = " << unique_values[b].first << ", "
				<< unique_values[b].second << " times." << std::endl;
		}
		*/
	}
	
	double get_obs(const matrix_t& et_gf)
	{
		if (!initialzed)
			init(et_gf);
		double chern4 = 0.;
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		int n = config.l.bonds("chern").size();
		for (int t = 0; t < unique_bonds.size(); ++t)
		{
			int i = unique_bonds[t][0], j = unique_bonds[t][1],
				k = unique_bonds[t][2], l = unique_bonds[t][3];
			const bond_t& a = config.l.bonds("chern")[i];
			const bond_t& b = config.l.bonds("chern")[j];
			const bond_t& c = config.l.bonds("chern")[k];
			const bond_t& d = config.l.bonds("chern")[l];
			bond_t a_prime = bond_t{a.second, a.first};
			bond_t b_prime = bond_t{b.second, b.first};
			bond_t c_prime = bond_t{c.second, c.first};
			bond_t d_prime = bond_t{d.second, d.first};
			
			int index = l + k*n + j*n*n + i*n*n*n;
			double ch = 0.;
			if ((non_zero_terms[index] & 1) == 1)
				ch += calculate_wick_det(et_gf, mat44, a, b, c, d);
			if ((non_zero_terms[index] & 2) == 2)
				ch -= calculate_wick_det(et_gf, mat44, a, b, c, d_prime);
			if ((non_zero_terms[index] & 4) == 4)
				ch -= calculate_wick_det(et_gf, mat44, a, b, c_prime, d);
			if ((non_zero_terms[index] & 8) == 8)
				ch += calculate_wick_det(et_gf, mat44, a, b, c_prime, d_prime);
			
			if ((non_zero_terms[index] & 16) == 16)
				ch -= calculate_wick_det(et_gf, mat44, a, b_prime, c, d);
			if ((non_zero_terms[index] & 32) == 32)
				ch += calculate_wick_det(et_gf, mat44, a, b_prime, c, d_prime);
			if ((non_zero_terms[index] & 64) == 64)
				ch += calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d);
			if ((non_zero_terms[index] & 128) == 128)
				ch -= calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d_prime);
			ch *= unique_values[t].second;
			chern4 += ch;
		}
		return 2. * chern4 / std::pow(config.l.n_bonds(), 4);
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
		double M2 = 0.;
		for (int i = 0; i < config.l.n_sites(); ++i)
			for (int j = 0; j < config.l.n_sites(); ++j)
			{
				double delta_ij = i == j ? 1. : 0.;
				M2 += config.l.parity(i) * config.l.parity(j)
					* std::real((1. - et_gf(i, i)) * (1. - et_gf(j, j))
					+ (delta_ij - et_gf(j, i)) * et_gf(i, j) - (et_gf(i, i) + et_gf(j, j))/2. + 1./4.);
			}
		return M2 / std::pow(config.l.n_sites(), 2.);
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
	
	double evaluate(Eigen::Matrix4cd& mat44, const matrix_t& et_gf, int i, int j, int k, int l)
	{
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
		
		double parity;		
		if (config.param.decoupling == "majorana")
			parity = config.l.parity(i) * config.l.parity(j)
				* config.l.parity(k) * config.l.parity(l);
		else
			parity = 1.;
		return parity * std::real(mat44.determinant());
	}
	
	double get_obs(const matrix_t& et_gf)
	{
		double M4 = 0.;
		int n = config.l.n_sites();
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		
		M4 += evaluate(mat44, et_gf, 0, 0, 0, 0) * (3.*n*n - 2.*n);
		for (int i = 0; i < n; ++i)
			for (int j = i+1; j < n; ++j)
			{
				M4 += evaluate(mat44, et_gf, 0, 0, i, j) * (12.*n - 16.);
				for (int k = j+1; k < n; ++k)
					for (int l = k+1; l < n; ++l)
						M4 += evaluate(mat44, et_gf, i, j, k, l) * 24.;
			}
		return M4 / std::pow(config.l.n_sites(), 4.);
	}
};

struct wick_static_kek
{
	configuration& config;
	Random& rng;

	wick_static_kek(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	double get_obs(const matrix_t& et_gf)
	{
		std::complex<double> kek = 0.;
		std::array<const std::vector<std::pair<int, int>>*, 3> kek_bonds =
			{&config.l.bonds("kekule"), &config.l.bonds("kekule_2"),
			&config.l.bonds("kekule_3")};
		std::array<double, 3> factors = {2., -1., -1.};
		if (config.param.decoupling == "majorana")
		{
			for (int i = 0; i < kek_bonds.size(); ++i)
				for (int m = 0; m < kek_bonds.size(); ++m)
					for (int j = 0; j < kek_bonds[i]->size(); ++j)
						for (int n = 0; n < kek_bonds[m]->size(); ++n)
						{
							auto& a = (*kek_bonds[i])[j];
							auto& b = (*kek_bonds[m])[n];
							
							kek += factors[i] * factors[m]
								* config.l.parity(a.first) * config.l.parity(b.first)
								* (et_gf(a.second, a.first) * et_gf(b.first, b.second)
								+ et_gf(b.first, a.first) * et_gf(b.second, a.second));
						}
		}
		else
		{
		for (int i = 0; i < kek_bonds.size(); ++i)
			for (int m = 0; m < kek_bonds.size(); ++m)
				for (int j = 0; j < kek_bonds[i]->size(); ++j)
					for (int n = 0; n < kek_bonds[m]->size(); ++n)
					{
						auto& a = (*kek_bonds[i])[j];
						auto& b = (*kek_bonds[m])[n];
						
						double delta_im = a.first == b.first ? 1. : 0.;
						
						kek += factors[i] * factors[m]
								* (et_gf(a.second, a.first) * et_gf(b.first, b.second)
								+ (delta_im - et_gf(b.first, a.first)) * et_gf(b.second, a.second));
					}
		}
		return std::real(kek) / std::pow(config.l.n_bonds(), 2.);
	}
};

