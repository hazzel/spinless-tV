#pragma once
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "dump.h"
#include "lattice.h"

struct helper_matrices
{
	template<int n, int m>
	using matrix_t = Eigen::Matrix<std::complex<double>, n, m, Eigen::ColMajor>;
	
	matrix_t<Eigen::Dynamic, Eigen::Dynamic> m;

};

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
			M.resize(l.n_sites(), l.n_sites());
			rebuild();
		}

		const arg_t& vertex(int index)
		{
			return vertices[index]; 
		}

		void build(std::vector<arg_t>& args)
		{
			vertices = std::move(args);
			n_svd_interval = vertices.size() / n_svd;
			M.resize(l.n_sites(), l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (M.rows() == 0) return;
			M = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			dmatrix_t btest = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = 1; n <= n_svd; ++n)
			{
				dmatrix_t b = propagator(n * n_svd_interval,
					(n - 1) * n_svd_interval);
				store_svd_forward(b, n);
				btest *= b;
				std::cout << "n = " << n << std::endl;
				print_matrix(btest);
				std::cout << "##########" << std::endl;
				print_matrix(U[n-1] * D[n-1] * V[n-1]);
				std::cout << std::endl << std::endl;
			}
			M += btest;
			start_backward_sweep();
		}

		dmatrix_t propagator(int tau_n, int tau_m)
		{	
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver;
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_m; n < tau_n; ++n)
			{
				dmatrix_t h = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				for (int i = 0; i < b.rows(); ++i)
					for (int j = 0; j < b.cols(); ++j)
						h(i, j) += complex_t(0., function(vertices[n], i, j));
				solver.compute(h);
				dmatrix_t d = solver.eigenvalues().asDiagonal();
				for (int i = 0; i < d.rows(); ++i)
					d(i, i) = std::exp(d(i, i));
				b *= solver.eigenvectors().adjoint() * d * solver.eigenvectors();
			}
			return b;
		}
		
		void start_forward_sweep()
		{
			equal_time_gf = (dmatrix_t::Identity(l.n_sites(), l.n_sites())
				+ V.front() * D.front() * U.front()).inverse();
			U.front() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			D.front() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			V.front() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
		}

		void start_backward_sweep()
		{
			equal_time_gf = (dmatrix_t::Identity(l.n_sites(), l.n_sites())
				+ U.back() * D.back() * V.back()).inverse();
			U.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			D.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			V.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
		}

		void advance_forward()
		{
			if ((tau + 2) % n_svd_interval == 0)
			{
				int n = (tau + 2) / n_svd_interval;
				dmatrix_t b = propagator((n + 1) * n_svd_interval,
					n * n_svd_interval);
				store_svd_forward(b, n);
			}
			else
			{
				dmatrix_t b = propagator(tau + 1, tau);
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
				dmatrix_t b = propagator(tau, tau - 1);
				equal_time_gf = b.inverse() * equal_time_gf * b;
			}
			--tau;
		}

		// n = 1, ..., n_svd
		void store_svd_forward(const dmatrix_t& b, int n)
		{
			dmatrix_t r = V[n-1] * D[n-1] * U[n-1];
			if (n == 1)
			{
				svd_solver.compute(b, Eigen::ComputeThinU | Eigen::ComputeThinV);
				V[n-1] = svd_solver.matrixV().transpose();
			}
			else
			{
				svd_solver.compute(b * U[n-2] * D[n-2], Eigen::ComputeThinU |
					Eigen::ComputeThinV);
				V[n-1] = svd_solver.matrixV().transpose() * V[n-2];
			}
			U[n-1] = svd_solver.matrixU();
			D[n-1] = svd_solver.singularValues().asDiagonal();
			// Recompute equal time gf
			equal_time_gf = (dmatrix_t::Identity(l.n_sites(), l.n_sites())
				+ U[n-1] * D[n-1] * V[n-1] * r).inverse();
		}
	
		//n = n_svd - 1, ..., 1	
		void store_svd_backward(const dmatrix_t& b, int n)
		{
			svd_solver.compute(D[n] * U[n] * b, Eigen::ComputeThinU |
				Eigen::ComputeThinV);
			dmatrix_t r = U[n-1] * D[n-1] * V[n-1];
			V[n-1] = V[n] * svd_solver.matrixU();
			D[n-1] = svd_solver.singularValues().asDiagonal();
			U[n-1] = svd_solver.matrixV().transpose();
			// Recompute equal time gf
			equal_time_gf = (dmatrix_t::Identity(l.n_sites(), l.n_sites())
				+ r * V[n-1] * D[n-1] * U[n-1]).inverse();
		}

		template<int N>
		double try_shift(std::vector<arg_t>& args)
		{
		}
		
		template<int N>
		void finish_shift()
		{
		}
	private:
		void print_matrix(const dmatrix_t& m)
		{
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl;
		}
	private:
		function_t function;
		const lattice& l;
		int n_svd;
		int n_svd_interval;
		int tau;
		std::vector<arg_t> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		dmatrix_t equal_time_gf;
		std::vector<dmatrix_t> U;
		std::vector<dmatrix_t> D;
		std::vector<dmatrix_t> V;
		helper_matrices helper;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
};
