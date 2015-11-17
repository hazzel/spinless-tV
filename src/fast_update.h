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
		{}

		const arg_t& vertex(int index)
		{
			return vertices[index]; 
		}

		void build(std::vector<arg_t>& args)
		{
			vertices = std::move(args);
			M.resize(l.n_sites(), l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (M.rows() == 0) return;
			Eigen::JacobiSVD<dmatrix_t> svd_solver;
			M = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = 1; n <= n_svd; ++n)
			{
				dmatrix_t b = propagator(n * n_svd, (n - 1) * n_svd);
				if (n == 1)
				{
					svd_solver.compute(b, Eigen::ComputeThinU |
						Eigen::ComputeThinV);
					V[n-1] = svd_solver.matrixV();
				}
				else
				{
					svd_solver.compute(b * U[n-2] * D[n-2],
						Eigen::ComputeThinU | Eigen::ComputeThinV);
					V[n-1] = svd_solver.matrixV() * V[n-2];
				}
				U[n-1] = svd_solver.matrixU();
				D[n-1] = svd_solver.singularValues().asDiagonal();
			}
			M += U.back() * D.back() * V.back(); 
			std::cout << std::abs(M.determinant()) / std::pow(2., vertices.size())
				<< std::endl;
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

		void advance_equal_time_gf(int direction)
		{
			dmatrix_t b;
			if (direction == -1)
				b = propagator(tau, tau - 1);
			else
				b = propagator(tau + 1, tau);
			equal_time_gf = b.inverse() * equal_time_gf * b;
			tau += direction;
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
					{
						h(i, j) += complex_t(0., function(vertices[n], i, j));
					}
				solver.compute(h);
				dmatrix_t d = solver.eigenvalues().asDiagonal();
				for (int i = 0; i < d.rows(); ++i)
					d(i, i) = std::exp(d(i, i));
				b *= solver.eigenvectors().adjoint() * d * solver.eigenvectors();
			}
			return b;
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
};
