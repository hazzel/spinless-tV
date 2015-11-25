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
			if (vertices.size() == 0) return;
			for (int n = 1; n <= n_svd; ++n)
			{
				dmatrix_t b = propagator(n * n_svd_interval,
					(n - 1) * n_svd_interval);
				store_svd_forward(b, n);
			}
			std::cout << "Start forwards sweep." << std::endl;
			start_backward_sweep();
			while (tau > 0)
				advance_backward();
			std::cout << "after backward advancement" << std::endl;
			dmatrix_t btest = V.front() * D.front() * U.front();
			print_matrix(btest);
			std::cout << "greens function" << std::endl;
			print_matrix(equal_time_gf);
			start_forward_sweep();
			while (tau < vertices.size()/2 - 1)
				advance_forward();
			std::cout << "after forward advancement" << std::endl;
			std::cout << "tau = " << tau << std::endl;
			try_flip();
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
				dmatrix_t d = solver.eigenvalues().unaryExpr([](double e)
					{ return std::exp(e); }).asDiagonal();
				b = solver.eigenvectors() * d * solver.eigenvectors().adjoint() * b;
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
			tau = 0;
		}

		void start_backward_sweep()
		{
			equal_time_gf = (dmatrix_t::Identity(l.n_sites(), l.n_sites())
				+ U.back() * D.back() * V.back()).inverse();
			U.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			D.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			V.back() = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			tau = vertices.size() - 1;
		}

		void advance_forward()
		{
			if ((tau + 2) % n_svd_interval == 0)
			{
				int n = (tau + 2) / n_svd_interval;
				dmatrix_t b = propagator(n * n_svd_interval,
					(n - 1) * n_svd_interval);
				store_svd_forward(b, n);
			}
			else
			{
				dmatrix_t b = propagator(tau + 1, tau);
				equal_time_gf = b * equal_time_gf * b.inverse();
			}
			std::cout << "tau = " << tau << std::endl;
			print_matrix(equal_time_gf);
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
			std::cout << "tau = " << tau << std::endl;
			print_matrix(equal_time_gf);
			--tau;
		}

		// n = 1, ..., n_svd
		void store_svd_forward(const dmatrix_t& b, int n)
		{
			dmatrix_t U_r = U[n-1];
			dmatrix_t D_r = D[n-1];
			dmatrix_t V_r = V[n-1];
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
			D[n-1] = svd_solver.singularValues().asDiagonal();
			// Recompute equal time gf
			compute_equal_time_gf(U[n-1], D[n-1], V[n-1], U_r, D_r, V_r);
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
			D[n-1] = svd_solver.singularValues().asDiagonal();
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
			dmatrix_t D = svd_solver.singularValues().unaryExpr([](double s)
				{ return 1. / s; }).asDiagonal();
			equal_time_gf = (U_l.adjoint() * svd_solver.matrixV()) * D
				* (svd_solver.matrixU().adjoint() * U_r.adjoint());
		}

		void try_flip()
		{
			dmatrix_t id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			int i = 1;
			int j = l.neighbors(i, "nearest neighbors")[0];
			dmatrix_t h_old = propagator(tau + 1, tau);
			dmatrix_t b_old = id + propagator(vertices.size(), 0);
			
			dmatrix_t g = (id + propagator(tau, 0) * propagator(vertices.size(),
				tau)).inverse();
			print_matrix(g);
			print_matrix(equal_time_gf);
			
			vertices[tau](i, j) *= -1.;

			dmatrix_t h_new = propagator(tau + 1, tau);
			dmatrix_t delta = h_new * h_old.inverse() - dmatrix_t::Identity(h_new.rows(), h_new.cols());
			std::cout << std::endl;
			
			Eigen::EigenSolver<dmatrix_t> solver;
			dmatrix_t b_new = id + propagator(vertices.size(), 0);

			dmatrix_t x = id + delta;
			x.noalias() -= delta * equal_time_gf;
			std::cout << "ratio fast " << std::abs(x.determinant()) << std::endl;
			std::cout << "ratio slow " << std::abs(b_new.determinant()
				/ b_old.determinant()) << std::endl;
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
			std::cout << m.format(clean) << std::endl << std::endl;
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
