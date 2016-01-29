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
			: function(function_), l(l_), n_svd(n_svd_), U(n_svd_), D(n_svd_),
			V(n_svd_)
		{
			for (int n = 0; n < n_svd; ++n)
			{
				U[n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				D[n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				V[n] = dmatrix_t::Zero(l.n_sites(), l.n_sites());
			}
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
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

		const arg_t& vertex(int index)
		{
			return vertices[index]; 
		}

		int get_tau()
		{
			return tau;
		}

		int get_max_tau()
		{
			return max_tau;
		}

		void build(std::vector<arg_t>& args)
		{
			vertices = std::move(args);
			max_tau = vertices.size();
			n_svd_interval = max_tau / n_svd;
			M.resize(l.n_sites(), l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (vertices.size() == 0) return;
			for (int n = 1; n <= n_svd; ++n)
			{
				dmatrix_t b = propagator(n * n_svd_interval,
					(n - 1) * n_svd_interval);
				store_svd_forward(b, n);
			}
		}

		dmatrix_t propagator(int tau_n, int tau_m)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver;
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_m; n < tau_n; ++n)
			{
				dmatrix_t h = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				std::vector<dmatrix_t> h_cb(3);
				for (auto& m : h_cb)
					m = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				for (int i = 0; i < b.rows(); ++i)
				{
					auto& nn = l.neighbors(i, "nearest neighbors");
					for (auto j : nn)
						h(i, j) += complex_t(0., function(vertices[n], i, j));
					for (int j = 0; j < nn.size(); ++j)
						if (i < nn[j])
						{
							h_cb[j](i, nn[j]) += complex_t(0.,
								function(vertices[n], i, nn[j]));
							h_cb[j](nn[j], i) += complex_t(0.,
								function(vertices[n], nn[j], i));
						}
				}
				solver.compute(h);
				dmatrix_t d = solver.eigenvalues().cast<complex_t>().
					unaryExpr([](complex_t e) { return std::exp(e); }).asDiagonal();
				dmatrix_t exp = solver.eigenvectors() * d
					* solver.eigenvectors().adjoint();
				std::cout << "exp(h)" << std::endl;
				print_matrix(exp);
				std::cout << "h_a" << std::endl;
				print_matrix(h_cb[0]);
				std::cout << "h_b" << std::endl;
				print_matrix(h_cb[1]);
				std::cout << "h_c" << std::endl;
				print_matrix(h_cb[2]);
				solver.compute(h_cb[0]);
				d = solver.eigenvalues().cast<complex_t>().
					unaryExpr([](complex_t e) { return std::exp(e); }).asDiagonal();
				exp = solver.eigenvectors() * d * solver.eigenvectors().adjoint();
				std::cout << "exp(h_a)" << std::endl;
				print_matrix(exp);
				std::cout << "\\\\\\\\\\\\\\\\\\\\" << std::endl;
				b = solver.eigenvectors() * d * solver.eigenvectors().adjoint() * b;
			}
			return b;
		}
		
		void start_forward_sweep()
		{
			equal_time_gf = (id + V.front() * D.front() * U.front()).inverse();
			U.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			D.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			V.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			tau = 0;
		}

		void start_backward_sweep()
		{
			equal_time_gf = (id + U.back() * D.back() * V.back()).inverse();
			U.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			D.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			V.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			tau = max_tau - 1;
		}

		void advance_forward()
		{
			if ((tau + 2) % n_svd_interval == 0)
			{
				int n = (tau + 2) / n_svd_interval;
				dmatrix_t b = propagator(n * n_svd_interval,
					(n-1) * n_svd_interval);
				store_svd_forward(b, n);
			}
			else
			{
				dmatrix_t b = propagator(tau + 2, tau + 1);
				equal_time_gf = b * equal_time_gf * b.inverse();
			}
			++tau;
		}

		void advance_backward()
		{
			if ((tau - 1 + 1) % n_svd_interval == 0)
			{
				int n = (tau - 1 + 1) / n_svd_interval;
				dmatrix_t b = propagator((n + 1) * n_svd_interval,
					n * n_svd_interval);
				store_svd_backward(b, n);
			}
			else
			{
				dmatrix_t b = propagator(tau + 1, tau);
				equal_time_gf = b.inverse() * equal_time_gf * b;
			}
			--tau;
		}

		// n = 1, ..., n_svd
		void store_svd_forward(const dmatrix_t& b, int n)
		{
			dmatrix_t U_l = U[n-1];
			dmatrix_t D_l = D[n-1];
			dmatrix_t V_l = V[n-1];
			if (n == 1)
			{
				svd_solver.compute(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
				V[n-1] = svd_solver.matrixV().adjoint();
			}
			else
			{
				svd_solver.compute(b * U[n-2] * D[n-2], Eigen::ComputeThinU |
					Eigen::ComputeThinV);
				V[n-1] = svd_solver.matrixV().adjoint() * V[n-2];
			}
			U[n-1] = svd_solver.matrixU();
			D[n-1] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
			// Recompute equal time gf
			compute_equal_time_gf(U_l, D_l, V_l, U[n-1], D[n-1], V[n-1]);
		}
	
		//n = n_svd - 1, ..., 1	
		void store_svd_backward(const dmatrix_t& b, int n)
		{
			svd_solver.compute(D[n] * U[n] * b, Eigen::ComputeThinU |
				Eigen::ComputeThinV);
			dmatrix_t U_r = U[n-1];
			dmatrix_t D_r = D[n-1];
			dmatrix_t V_r = V[n-1];
			V[n-1] = V[n] * svd_solver.matrixU();
			D[n-1] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
			U[n-1] = svd_solver.matrixV().adjoint();
			// Recompute equal time gf
			compute_equal_time_gf(U[n-1], D[n-1], V[n-1], U_r, D_r, V_r);
		}

		void compute_equal_time_gf(const dmatrix_t& U_l, const dmatrix_t& D_l,
			const dmatrix_t& V_l, const dmatrix_t& U_r, const dmatrix_t& D_r,
			const dmatrix_t& V_r)
		{
			svd_solver.compute(U_r.adjoint() * U_l.adjoint() + D_r * (V_r * V_l)
				* D_l);
			dmatrix_t D = svd_solver.singularValues().cast<complex_t>().
				unaryExpr([](complex_t s) { return 1. / s; }).asDiagonal();
			equal_time_gf = (U_l.adjoint() * svd_solver.matrixV()) * D
				* (svd_solver.matrixU().adjoint() * U_r.adjoint());
		}

		double try_ising_flip(int i, int j)
		{
			dmatrix_t h_old = propagator(tau + 1, tau);
			vertices[tau](i, j) *= -1.;
			delta = propagator(tau + 1, tau) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf;
			return std::abs(x.determinant());
		}

		double try_ising_flip(std::vector<std::pair<int, int>>& sites)
		{
			dmatrix_t h_old = propagator(tau + 1, tau);
			for (auto& s : sites)
				vertices[tau](s.first, s.second) *= -1.;
			delta = propagator(tau + 1, tau) * h_old.inverse() - id;
			dmatrix_t x = id + delta; x.noalias() -= delta * equal_time_gf;
			return std::abs(x.determinant());
		}

		void undo_ising_flip(int i, int j)
		{
			vertices[tau](i, j) *= -1.;
		}

		void undo_ising_flip(std::vector<std::pair<int, int>>& sites)
		{
			for (auto& s : sites)
				vertices[tau](s.first, s.second) *= -1.;
		}

		void update_equal_time_gf_after_flip()
		{
			Eigen::ComplexEigenSolver<dmatrix_t> solver(delta);
			dmatrix_t V = solver.eigenvectors().cast<complex_t>();
			Eigen::VectorXcd ev = solver.eigenvalues();
			equal_time_gf = (V.inverse() * equal_time_gf * V).eval();
			for (int i = 0; i < delta.rows(); ++i)
			{
				dmatrix_t g = equal_time_gf;
				for (int x = 0; x < equal_time_gf.rows(); ++x)
					for (int y = 0; y < equal_time_gf.cols(); ++y)
						equal_time_gf(x, y) -= g(x, i) * ev[i]
							* ((i == y ? 1.0 : 0.0) - g(i, y))
							/ (1.0 + ev[i] * (1. - g(i, i)));
			}
			equal_time_gf = (V * equal_time_gf * V.inverse()).eval();
		}

		std::vector<double> measure_M2()
		{
			std::vector<double> c(l.max_distance()+2, 0.0);
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf(j, i));
						double im = std::imag(equal_time_gf(j, i));
						//Correlation function
						c[l.distance(i, j)] += l.parity(i) * l.parity(j)
							* (re*re + im*im) / l.n_sites();
						//M2 structure factor
						c.back() += //l.parity(i) * l.parity(j)
							(re*re + im*im) / std::pow(l.n_sites(), 2); 
					}
			return c;
		}
	private:
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
		int tau;
		int max_tau;
		std::vector<arg_t> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		dmatrix_t equal_time_gf;
		dmatrix_t id;
		dmatrix_t delta;
		std::vector<dmatrix_t> U;
		std::vector<dmatrix_t> D;
		std::vector<dmatrix_t> V;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
};
