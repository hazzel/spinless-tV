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
			energy += -config.l.parity(a.first) * config.param.t
				* std::imag(et_gf(a.second, a.first))
				+ config.param.V * std::real(et_gf(a.second, a.first)
				* et_gf(a.second, a.first))/2.;
		for (auto& a : config.l.bonds("d3_bonds"))
			energy += -config.l.parity(a.first) * config.param.tprime
				* std::imag(et_gf(a.second, a.first));
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
			epsilon -= config.l.parity(a.first)
				* std::imag(et_gf(a.second, a.first));
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
		for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		for (int k = 0; k < n; ++k)
		for (int l = 0; l < n; ++l)
		{
			const bond_t& a = config.l.bonds("chern")[i];
			const bond_t& b = config.l.bonds("chern")[j];
			const bond_t& c = config.l.bonds("chern")[k];
			const bond_t& d = config.l.bonds("chern")[l];
			bond_t a_prime = bond_t{a.second, a.first};
			bond_t b_prime = bond_t{b.second, b.first};
			bond_t c_prime = bond_t{c.second, c.first};
			bond_t d_prime = bond_t{d.second, d.first};
						
			int mask = 0;
			double value = 0.;
			double w = calculate_wick_det(et_gf, mat44, a, b, c, d);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 1;
			w = calculate_wick_det(et_gf, mat44, a, b, c, d_prime);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 2;
			w = calculate_wick_det(et_gf, mat44, a, b, c_prime, d);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 4;
			w = calculate_wick_det(et_gf, mat44, a, b, c_prime, d_prime);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 8;
						
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c, d);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 16;
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c, d_prime);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 32;
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d);
			value += w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 64;
			w = calculate_wick_det(et_gf, mat44, a, b_prime, c_prime, d_prime);
			value -= w;
			if (std::abs(w) > std::pow(10, -14.))
				mask |= 128;
			non_zero_terms.push_back(mask);

			bool found = false;
			for (auto& v : unique_values)
			{
				if (std::abs(value) > std::pow(10., -14) && std::abs(v.first - value) < std::pow(10., -14))
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
		initialzed = true;
		
		std::cout << "chern4: " << unique_bonds.size() << " of "
			<< std::pow(n, 4) << std::endl;
		for (int b = 0; b < unique_bonds.size(); ++b)
		{
			int i = unique_bonds[b][0], j = unique_bonds[b][1],
				k = unique_bonds[b][2], l = unique_bonds[b][3];
			std::cout << "bond: (" << i << ", " << j << ", " << k << ", " << l
				<< "), value = " << unique_values[b].first << ", "
				<< unique_values[b].second << " times." << std::endl;
		}
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
	std::vector<std::pair<double, int>> unique_values;
	std::vector<std::array<int, 4>> unique_bonds;
	bool initialzed = false;

	wick_static_M4(configuration& config_, Random& rng_)
		: config(config_), rng(rng_)
	{}
	
	void init(const matrix_t& et_gf)
	{
		/*
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		for (int i = 0; i < config.l.n_sites(); ++i)
		//int i = rng() * config.l.n_sites();
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
						
						double parity = config.l.parity(i) * config.l.parity(j)
							* config.l.parity(k) * config.l.parity(l);
						double value = parity * std::real(mat44.determinant());
						bool found = false;
						for (auto& v : unique_values)
						{
							if (std::abs(v.first - value) < std::pow(10., -14))
							{
								if (unique_values.size() > 1
									&& std::abs(value - unique_values[1].first) < 0.0000001)
									std::cout << "bond: (" << i << ", " << j << ", "
										<< k << ", " << l << ")" << std::endl;
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
			}
		initialzed = true;
		std::cout << "M4: " << unique_bonds.size() << " of "
			<< std::pow(config.l.n_sites(), 4) << std::endl;
		for (int b = 0; b < unique_bonds.size(); ++b)
		{
			int i = unique_bonds[b][0], j = unique_bonds[b][1],
				k = unique_bonds[b][2], l = unique_bonds[b][3];
			std::cout << "bond: (" << i << ", " << j << ", " << k << ", " << l
				<< "), value = " << unique_values[b].first << ", "
				<< unique_values[b].second << " times." << std::endl;
		}

		int cnt = 0;
		for (int i = 0; i < config.l.n_sites(); ++i)
		for (int j = 0; j < config.l.n_sites(); ++j)
		for (int k = 0; k < config.l.n_sites(); ++k)
		for (int l = 0; l < config.l.n_sites(); ++l)
			if ((i ==
				++cnt;
		std::cout << "cnt = " << cnt << std::endl;
		*/
		
		int n = config.l.n_sites();
		unique_values.push_back({0., 3*n*n - 2*n});
		unique_bonds.push_back({0, 0, 0, 0});
		for (int i = 0; i < n; ++i)
			for (int j = i+1; j < n; ++j)
			{
				unique_values.push_back({0., 12*n - 16});
				unique_bonds.push_back({0, 0, i, j});
				for (int k = j+1; k < n; ++k)
					for (int l = k+1; l < n; ++l)
					{
						unique_values.push_back({0., 24});
						unique_bonds.push_back({i, j, k, l});
					}
			}
		initialzed = true;
	}
	
	double get_obs(const matrix_t& et_gf)
	{
		if (!initialzed)
			init(et_gf);
		double M4 = 0.;
		int n = config.l.n_sites();
		Eigen::Matrix4cd mat44 = Eigen::Matrix4cd::Zero();
		for (int b = 0; b < unique_bonds.size(); ++b)
		{
			int i = unique_bonds[b][0], j = unique_bonds[b][1],
				k = unique_bonds[b][2], l = unique_bonds[b][3];
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
						
			double parity = config.l.parity(i) * config.l.parity(j)
				* config.l.parity(k) * config.l.parity(l);
			M4 += parity * std::real(mat44.determinant()) * unique_values[b].second;
		}
		return M4 / std::pow(config.l.n_sites(), 4.);
	}
};
