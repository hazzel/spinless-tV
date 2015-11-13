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

		fast_update(const function_t& function_, const lattice& l_)
			: function(function_), l(l_)
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
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver;
			M = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (arg_t& v : vertices)
			{
				dmatrix_t h = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				for (int i = 0; i < b.rows(); ++i)
					for (int j = 0; j < b.cols(); ++j)
						h(i, j) += complex_t(0., function(v, i, j));
				solver.compute(h);
				dmatrix_t d = solver.eigenvalues().asDiagonal();
				for (int l = 0; l < d.rows(); ++l)
					d(l, l) = std::exp(d(l, l));
				b *= solver.eigenvectors().adjoint() * d * solver.eigenvectors();
				//std::cout << b(1, 0) << std::endl;
			}
			M += b;
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
		std::vector<arg_t> vertices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		helper_matrices helper;
};
