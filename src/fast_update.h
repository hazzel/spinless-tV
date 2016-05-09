#pragma once
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "dump.h"
#include "lattice.h"
#include "measurements.h"
#include "parameters.h"
#include "qr_stabilizer.h"

template<typename function_t, typename arg_t>
class fast_update
{
	public:
		using complex_t = std::complex<double>;
		template<int n, int m>
		using matrix_t = Eigen::Matrix<complex_t, n, m,
			Eigen::ColMajor>; 
		using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;
		using stabilizer_t = qr_stabilizer;

		fast_update(const function_t& function_, const lattice& l_,
			const parameters& param_, measurements& measure_)
			: function(function_), l(l_), param(param_), measure(measure_),
				cb_bonds(3), tau{0, 0},
				equal_time_gf(std::vector<dmatrix_t>(2)),
				time_displaced_gf(std::vector<dmatrix_t>(2)),
				stabilizer{measure, equal_time_gf, time_displaced_gf}
		{
			std::cout << "fast_update constructor" << std::endl;
		}

		void serialize(odump& out)
		{
			/*
			int size = vertices.size();
			out.write(size);
			for (arg_t& v : vertices)
				v.serialize(out);
			*/
		}

		void serialize(idump& in)
		{
			/*
			int size; in.read(size);
			for (int i = 0; i < size; ++i)
			{
				arg_t v;
				v.serialize(in);
				vertices.push_back(v);
			}
			max_tau = size;
			n_svd_interval = max_tau / param.n_svd;
			M.resize(l.n_sites(), l.n_sites());
			//rebuild();
			*/
		}
		
		void initialize()
		{
			M.resize(l.n_sites(), l.n_sites());
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			create_checkerboard();
		}

		const arg_t& vertex(int species, int index)
		{
			return vertices[species][index]; 
		}

		int get_tau(int species)
		{
			return tau[species];
		}

		int get_max_tau()
		{
			return max_tau;
		}

		void build(boost::multi_array<arg_t, 2>& args)
		{
			vertices = std::move(args);
			max_tau = vertices.shape()[1];
			n_svd_interval = max_tau / param.n_svd;
			rebuild();
		}

		void rebuild()
		{
			if (vertices.shape()[1] == 0) return;
			for (int i = 0; i < 2; ++i)
				for (int n = 1; n <= param.n_svd; ++n)
				{
					dmatrix_t b = propagator(i, n * n_svd_interval,
						(n - 1) * n_svd_interval);
					stabilizer.set(i, n, b);
				}
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver;
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_n; n > tau_m; --n)
			{
				dmatrix_t h = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				std::vector<dmatrix_t> h_cb(3);
				for (auto& m : h_cb)
					m = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				for (int i = 0; i < l.n_sites(); ++i)
				{
					auto& nn = l.neighbors(i, "nearest neighbors");
					for (auto j : nn)
						h(i, j) += complex_t(0., function(vertices[species][n],
							i, j));
				}
				for (int i = 0; i < cb_bonds.size(); ++i)
					for (int j = 0; j < cb_bonds[i].size(); ++j)
						h_cb[i](j, cb_bonds[i][j]) += complex_t(0.,
							function(vertices[species][n], j, cb_bonds[i][j]));

				solver.compute(h);
				dmatrix_t d = solver.eigenvalues().cast<complex_t>().
					unaryExpr([](complex_t e) { return std::exp(e); }).asDiagonal();
				b *= solver.eigenvectors() * d * solver.eigenvectors().adjoint();

				for (int i = 0; i < cb_bonds.size(); ++i)
				{
					solver.compute(h_cb[i]);
					dmatrix_t d = solver.eigenvalues().cast<complex_t>().
						unaryExpr([](complex_t e) { return std::exp(e); }).
						asDiagonal();
					h_cb[i] = solver.eigenvectors() * d
						* solver.eigenvectors().adjoint();
				}
			}
			return b;
		}

		void advance_forward()
		{
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t b = propagator(i, tau[i] + 2, tau[i] + 1);
				equal_time_gf[i] = b * equal_time_gf[i] * b.inverse();
				++tau[i];
			}
		}

		void advance_backward()
		{
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t b = propagator(i, tau[i] + 1, tau[i]);
				equal_time_gf[i] = b.inverse() * equal_time_gf[i] * b;
				--tau[i];
			}
		}
		
		void stabilize_forward()
		{
			for (int i = 0; i < 2; ++i)
			{
				if (tau[i] + 1 % param.n_delta != 0)
					return;
				// n = 0, ..., n_intervals - 1
				int n = tau[i] + 1 / param.n_delta - 1;
				dmatrix_t b = propagator(i, (n+1) * param.n_delta, n * param.n_delta);
				stabilizer.stabilize_forward(i, n, b);
			}
		}
	
		void stabilize_backward()
		{
			for (int i = 0; i < 2; ++i)
			{
				if (tau[i] + 1 % param.n_delta != 0)
					return;
				//n = n_intervals, ..., 1 
				int n = tau[i] + 1 / param.n_delta + 1;
				dmatrix_t b = propagator(i, n * param.n_delta, (n-1) * param.n_delta);
				stabilizer.stabilize_backward(i, n, b);
			}
		}

		double try_ising_flip(int species, int i, int j)
		{
			dmatrix_t h_old = propagator(species, tau[species] + 1, tau[species]);
			vertices[species][tau[species]](i, j) *= -1.;
			delta = propagator(species, tau[species] + 1, tau[species]) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		double try_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			dmatrix_t h_old = propagator(species, tau[species] + 1, tau[species]);
			for (auto& s : sites)
				vertices[species][tau[species]](s.first, s.second) *= -1.;
			delta = propagator(species, tau[species] + 1, tau[species]) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		void undo_ising_flip(int species, int i, int j)
		{
			vertices[species][tau[species]](i, j) *= -1.;
		}

		void undo_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			for (auto& s : sites)
				vertices[species][tau[species]](s.first, s.second) *= -1.;
		}

		void update_equal_time_gf_after_flip(int species)
		{
			Eigen::ComplexEigenSolver<dmatrix_t> solver(delta);
			dmatrix_t V = solver.eigenvectors().cast<complex_t>();
			Eigen::VectorXcd ev = solver.eigenvalues();
			equal_time_gf[species] = (V.inverse() * equal_time_gf[species] * V).eval();
			for (int i = 0; i < delta.rows(); ++i)
			{
				dmatrix_t g = equal_time_gf[species];
				for (int x = 0; x < equal_time_gf[species].rows(); ++x)
					for (int y = 0; y < equal_time_gf[species].cols(); ++y)
						equal_time_gf[species](x, y) -= g(x, i) * ev[i]
							* ((i == y ? 1.0 : 0.0) - g(i, y))
							/ (1.0 + ev[i] * (1. - g(i, i)));
			}
			equal_time_gf[species] = (V * equal_time_gf[species] * V.inverse()).
				eval();
		}

		std::vector<double> measure_M2()
		{
			std::vector<double> c(l.max_distance()+2, 0.0);
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf[0](j, i)
							* equal_time_gf[1](j, i));
						double im = std::imag(equal_time_gf[0](j, i)
							* equal_time_gf[1](j, i));
						//Correlation function
						c[l.distance(i, j)] += l.parity(i) * l.parity(j)
							* re / l.n_sites();
						//M2 structure factor
						c.back() += l.parity(i) * l.parity(j) * re
							/ std::pow(l.n_sites(), 2);
					}
			return c;
		}
	private:
		void create_checkerboard()
		{
			for (int i = 0; i < l.n_sites(); ++i)
			{
				auto& nn = l.neighbors(i, "nearest neighbors");
				for (int j : nn)
				{
					for (auto& b : cb_bonds)
					{
						if (!b.count(i) && !b.count(j))
						{
							b[i] = j;
							b[j] = i;
							break;
						}
					}
				}
			}
		}

		void print_matrix(const dmatrix_t& m)
		{
			for (int i = 0; i < 2; ++i)
				std::cout << "Tau " << i << " = " << tau[i] << std::endl;
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
	private:
		function_t function;
		const lattice& l;
		const parameters& param;
		measurements& measure;
		int n_svd_interval;
		std::vector<int> tau;
		int max_tau;
		boost::multi_array<arg_t, 2> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		std::vector<dmatrix_t> equal_time_gf;
		std::vector<dmatrix_t> time_displaced_gf;
		dmatrix_t id;
		dmatrix_t delta;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
