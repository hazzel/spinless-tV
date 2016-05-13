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

template<typename arg_t>
class fast_update
{
	public:
		using complex_t = std::complex<double>;
		template<int n, int m>
		using matrix_t = Eigen::Matrix<complex_t, n, m,
			Eigen::ColMajor>; 
		using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;
		using stabilizer_t = qr_stabilizer;

		fast_update(const lattice& l_, const parameters& param_,
			measurements& measure_)
			: l(l_), param(param_), measure(measure_),
				cb_bonds(3), tau{1, 1},
				equal_time_gf(std::vector<dmatrix_t>(2)),
				time_displaced_gf(std::vector<dmatrix_t>(2)),
				stabilizer{measure, equal_time_gf, time_displaced_gf}
		{}

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
			n_intervals = max_tau / param.n_delta;
			M.resize(l.n_sites(), l.n_sites());
			//rebuild();
			*/
		}
		
		void initialize()
		{
			M.resize(l.n_sites(), l.n_sites());
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int i = 0; i < 2; ++i)
			{
				equal_time_gf[i] = 0.5 * id;
				time_displaced_gf[i] = 0.5 * id;
			}
			create_checkerboard();
		}
		
		double action(const arg_t& x, int i, int j) const
		{
			double sign = 1.0;
			if (l.distance(i, j) == 1)
				return sign*(param.t * param.dtau - param.lambda * x(i, j));
			else
				return 0.;
		}

		const arg_t& vertex(int species, int index)
		{
			return vertices[species][index-1]; 
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
			vertices.resize(boost::extents[args.shape()[0]][args.shape()[1]]);
			vertices = args;
			max_tau = vertices.shape()[1];
			tau = {max_tau, max_tau};
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (vertices.shape()[1] == 0) return;
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t p = id;
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(i, n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set(i, n, b);
				}
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
					for (auto j : l.neighbors(i, "nearest neighbors"))
						h(i, j) += complex_t(0., action(vertices[species][n-1],
							i, j));
				for (int i = 0; i < cb_bonds.size(); ++i)
					for (int j = 0; j < cb_bonds[i].size(); ++j)
						h_cb[i](j, cb_bonds[i][j]) += complex_t(0.,
							action(vertices[species][n-1], j, cb_bonds[i][j]));

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
			std::cout << "G(0)" << std::endl;
			print_matrix(equal_time_gf[0]);
			//std::cout << "G(1)" << std::endl;
			//print_matrix(equal_time_gf[1]);
			//equal_time_gf[0] = (id + propagator(0, tau[0], 0) * propagator(0, max_tau, tau[0])).inverse();
			//equal_time_gf[1] = (id + propagator(1, tau[1], 0) * propagator(1, max_tau, tau[1])).inverse();
			//std::cout << "G(0)" << std::endl;
			//print_matrix(equal_time_gf[0]);
			//std::cout << "G(1)" << std::endl;
			//print_matrix(equal_time_gf[1]);
			//std::cout << std::endl << std::endl;
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t b = propagator(i, tau[i] + 1, tau[i]);
				equal_time_gf[i] = b * equal_time_gf[i] * b.inverse();
				++tau[i];
			}
		}

		void advance_backward()
		{
			std::cout << "G(0)" << std::endl;
			print_matrix(equal_time_gf[0]);
			//std::cout << "G(1)" << std::endl;
			//print_matrix(equal_time_gf[1]);
			//equal_time_gf[0] = (id + propagator(0, max_tau, 0)).inverse();
			//equal_time_gf[1] = (id + propagator(1, max_tau, 0)).inverse();
			//std::cout << "G(0)" << std::endl;
			//print_matrix(equal_time_gf[0]);
			//std::cout << "G(1)" << std::endl;
			//print_matrix(equal_time_gf[1]);
			//std::cout << std::endl << std::endl;
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t b = propagator(i, tau[i], tau[i] - 1);
				equal_time_gf[i] = b.inverse() * equal_time_gf[i] * b;
				--tau[i];
			}
		}
		
		void stabilize_forward()
		{
			if (tau[0] % param.n_delta != 0)
					return;
			for (int i = 0; i < 2; ++i)
			{
				// n = 0, ..., n_intervals - 1
				int n = tau[i] / param.n_delta - 1;
				dmatrix_t b = propagator(i, (n+1) * param.n_delta, n * param.n_delta);
				stabilizer.stabilize_forward(i, n, b);
			}
		}
	
		void stabilize_backward()
		{
			if (tau[0] % param.n_delta != 0)
					return;
			for (int i = 0; i < 2; ++i)
			{
				//n = n_intervals, ..., 1 
				int n = tau[i] / param.n_delta + 1;
				dmatrix_t b = propagator(i, n * param.n_delta, (n-1) * param.n_delta);
				stabilizer.stabilize_backward(i, n, b);
			}
		}

		double try_ising_flip(int species, int i, int j)
		{
			dmatrix_t h_old = propagator(species, tau[species], tau[species] - 1);
			vertices[species][tau[species]-1](i, j) *= -1.;
			delta = propagator(species, tau[species], tau[species] - 1) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		double try_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			dmatrix_t h_old = propagator(species, tau[species], tau[species] - 1);
			for (auto& s : sites)
				vertices[species][tau[species]-1](s.first, s.second) *= -1.;
			delta = propagator(species, tau[species], tau[species] - 1) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		void undo_ising_flip(int species, int i, int j)
		{
			vertices[species][tau[species]-1](i, j) *= -1.;
		}

		void undo_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			for (auto& s : sites)
				vertices[species][tau[species]-1](s.first, s.second) *= -1.;
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
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
	private:
		const lattice& l;
		const parameters& param;
		measurements& measure;
		int n_intervals;
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
