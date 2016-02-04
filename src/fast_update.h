#pragma once
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "dump.h"
#include "lattice.h"

template<typename function_t, typename arg_t>
class fast_update
{
	public:
		using complex_t = std::complex<double>;
		template<int n, int m>
		using matrix_t = Eigen::Matrix<complex_t, n, m,
			Eigen::ColMajor>; 
		using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;

		fast_update(const function_t& function_, const lattice& l_, int n_svd_)
			: function(function_), l(l_), n_svd(n_svd_),
			U(boost::extents[2][n_svd_]), D(boost::extents[2][n_svd_]),
			V(boost::extents[2][n_svd_]), cb_bonds(3)
		{
			for (int i = 0; i < 2; ++i)
				for (int n = 0; n < n_svd; ++n)
				{
					U[i][n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
					D[i][n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
					V[i][n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				}
			M.resize(l.n_sites(), l.n_sites());
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			create_checkerboard();
		}

		void serialize(odump& out)
		{
			int size = vertices.size();
			out.write(size);
			for (arg_t& v : vertices)
				v.serialize(out);
		}

		void serialize(idump& in)
		{
			vertices.clear();
			int size; in.read(size);
			for (int i = 0; i < size; ++i)
			{
				arg_t v;
				v.serialize(in);
				vertices.push_back(v);
			}
			max_tau = vertices.size();
			n_svd_interval = max_tau / n_svd;
			M.resize(l.n_sites(), l.n_sites());
			//rebuild();
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
			max_tau = vertices.extents()[1];
			n_svd_interval = max_tau / n_svd;
			rebuild();
		}

		void rebuild()
		{
			if (vertices.extents()[1] == 0) return;
			for (int i = 0; i < 2; ++i)
				for (int n = 1; n <= n_svd; ++n)
				{
					dmatrix_t b = propagator(i, n * n_svd_interval,
						(n - 1) * n_svd_interval);
					store_svd_forward(i, b, n);
				}
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver;
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_m; n < tau_n; ++n)
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
				b = solver.eigenvectors() * d * solver.eigenvectors().adjoint() * b;

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
		
		void start_forward_sweep()
		{
			for (int i = 0; i < 2; ++i)
			{
				equal_time_gf = (id + V[i][0] * D[i][0] * U[i][0]).inverse();
				U[i][0] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				D[i][0] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				V[i][0] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				tau[i] = 0;
			}
		}

		void start_backward_sweep()
		{
			for (int i = 0; i < 2; ++i)
			{
				equal_time_gf = (id + U[i][n_svd] * D[i][n_svd] * V[i][n_svd]).
					inverse();
				U[i][n_svd] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				D[i][n_svd] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				V[i][n_svd] = dmatrix_t::Identity(l.n_sites(), l.n_sites());
				tau[i] = max_tau - 1;
			}
		}

		void advance_forward()
		{
			for (int i = 0; i < 2; ++i)
			{
				if ((tau[i] + 2) % n_svd_interval == 0)
				{
					int n = (tau[i] + 2) / n_svd_interval;
					dmatrix_t b = propagator(i, n * n_svd_interval,
						(n-1) * n_svd_interval);
					store_svd_forward(i, b, n);
				}
				else
				{
					dmatrix_t b = propagator(i, tau[i] + 2, tau[i] + 1);
					equal_time_gf[i] = b * equal_time_gf[i] * b.inverse();
				}
				++tau[i];
			}
		}

		void advance_backward()
		{
			for (int i = 0; i < 2; ++i)
			{
				if ((tau[i] - 1 + 1) % n_svd_interval == 0)
				{
					int n = (tau[i] - 1 + 1) / n_svd_interval;
					dmatrix_t b = propagator(i, (n + 1) * n_svd_interval,
						n * n_svd_interval);
					store_svd_backward(i, b, n);
				}
				else
				{
					dmatrix_t b = propagator(i, tau + 1, tau);
					equal_time_gf[i] = b.inverse() * equal_time_gf[i] * b;
				}
				--tau[i];
			}
		}

		// n = 1, ..., n_svd
		void store_svd_forward(int species, const dmatrix_t& b, int n)
		{
			dmatrix_t U_l = U[species][n-1];
			dmatrix_t D_l = D[species][n-1];
			dmatrix_t V_l = V[species][n-1];
			if (n == 1)
			{
				svd_solver.compute(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
				V[species][n-1] = svd_solver.matrixV().adjoint();
			}
			else
			{
				svd_solver.compute(b * U[species][n-2] * D[species][n-2],
					Eigen::ComputeThinU | Eigen::ComputeThinV);
				V[species][n-1] = svd_solver.matrixV().adjoint() * V[species][n-2];
			}
			U[species][n-1] = svd_solver.matrixU();
			D[species][n-1] = svd_solver.singularValues().template cast<complex_t>().
				asDiagonal();
			// Recompute equal time gf
			compute_equal_time_gf(species, U_l, D_l, V_l, U[species][n-1],
				D[species][n-1], V[species][n-1]);
		}
	
		//n = n_svd - 1, ..., 1	
		void store_svd_backward(int species, const dmatrix_t& b, int n)
		{
			svd_solver.compute(D[species][n] * U[species][n] * b,
				Eigen::ComputeThinU | Eigen::ComputeThinV);
			dmatrix_t U_r = U[species][n-1];
			dmatrix_t D_r = D[species][n-1];
			dmatrix_t V_r = V[species][n-1];
			V[species][n-1] = V[species][n] * svd_solver.matrixU();
			D[species][n-1] = svd_solver.singularValues().template cast<complex_t>().
				asDiagonal();
			U[species][n-1] = svd_solver.matrixV().adjoint();
			// Recompute equal time gf
			compute_equal_time_gf(species, U[species][n-1], D[species][n-1],
				V[species][n-1], U_r, D_r, V_r);
		}

		void compute_equal_time_gf(int species, const dmatrix_t& U_l,
			const dmatrix_t& D_l, const dmatrix_t& V_l, const dmatrix_t& U_r,
			const dmatrix_t& D_r, const dmatrix_t& V_r)
		{
			svd_solver.compute(U_r.adjoint() * U_l.adjoint() + D_r * (V_r * V_l)
				* D_l);
			dmatrix_t D = svd_solver.singularValues().template cast<complex_t>().
				unaryExpr([](complex_t s) { return 1. / s; }).asDiagonal();
			equal_time_gf[species] = (U_l.adjoint() * svd_solver.matrixV()) * D
				* (svd_solver.matrixU().adjoint() * U_r.adjoint());
		}

		double try_ising_flip(int species, int i, int j)
		{
			dmatrix_t h_old = propagator(species, tau + 1, tau);
			vertices[species][tau](i, j) *= -1.;
			delta = propagator(species, tau + 1, tau) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		double try_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			dmatrix_t h_old = propagator(species, tau + 1, tau);
			for (auto& s : sites)
				vertices[species][tau](s.first, s.second) *= -1.;
			delta = propagator(species, tau + 1, tau) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
		}

		void undo_ising_flip(int species, int i, int j)
		{
			vertices[species][tau](i, j) *= -1.;
		}

		void undo_ising_flip(int species, std::vector<std::pair<int, int>>& sites)
		{
			for (auto& s : sites)
				vertices[species][tau](s.first, s.second) *= -1.;
		}

		void update_equal_time_gf_after_flip(int species)
		{
			Eigen::ComplexEigenSolver<dmatrix_t> solver(delta);
			dmatrix_t V = solver.eigenvectors().cast<complex_t>();
			Eigen::VectorXcd ev = solver.eigenvalues();
			equal_time_gf = (V.inverse() * equal_time_gf[species] * V).eval();
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
			std::cout << "Tau = " << tau << std::endl;
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
	private:
		function_t function;
		const lattice& l;
		int n_svd;
		int n_svd_interval;
		boost::multi_array<int, 2> tau;
		int max_tau;
		boost::multi_array<arg_t, 2> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		std::vector<dmatrix_t> equal_time_gf;
		dmatrix_t id;
		dmatrix_t delta;
		boost::multi_array<dmatrix_t, 2> U;
		boost::multi_array<dmatrix_t, 2> D;
		boost::multi_array<dmatrix_t, 2> V;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
		std::vector<std::map<int, int>> cb_bonds;
};
