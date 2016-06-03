#pragma once
#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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
		using sparse_t = Eigen::SparseMatrix<complex_t>;
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
			delta.resize(2, 2);
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			id_2 = dmatrix_t::Identity(2, 2);
			for (int i = 0; i < 2; ++i)
			{
				equal_time_gf[i] = 0.5 * id;
				time_displaced_gf[i] = 0.5 * id;
			}
			create_checkerboard();
		}

		int get_bond_type(const std::pair<int, int>& bond) const
		{
			for (int i = 0; i < cb_bonds.size(); ++i)
				if (cb_bonds[i].at(bond.first) == bond.second)
					return i;
		}
		
		double action(const arg_t& x, int i, int j) const
		{
			double a = (get_bond_type({i, j}) < cb_bonds.size() - 1) ? 0.5 : 1.0;
			double sign = 1.0;
			if (l.distance(i, j) == 1)
				return a * sign * (param.t * param.dtau - param.lambda * x(i, j));
			else
				return 0.;
		}
		
		double action(double x, int bond_type) const
		{
			double a = (bond_type < cb_bonds.size() - 1) ? 0.5 : 1.0;
			double sign = 1.0;
			return a * sign * (param.t * param.dtau - param.lambda * x);
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
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(i, n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set(i, n, b);
				}
			}
		}

		sparse_t vertex_matrix(int bond_type, const arg_t& vertex)
		{
			sparse_t v(l.n_sites(), l.n_sites());
			std::vector<Eigen::Triplet<complex_t>> triplets;
			for (int i = 0; i < cb_bonds[bond_type].size(); ++i)
			{
				triplets.push_back({i, cb_bonds[bond_type][i], complex_t(0.,
					std::sin(action(vertex, i, cb_bonds[bond_type][i])))});
				triplets.push_back({i, i, complex_t(std::cos(action(vertex, i,
					cb_bonds[bond_type][i])), 0.)});
			}
			v.setFromTriplets(triplets.begin(), triplets.end());
			return v;
		}
		
		sparse_t inv_vertex_matrix(int bond_type, const arg_t& vertex)
		{
			sparse_t v(l.n_sites(), l.n_sites());
			std::vector<Eigen::Triplet<complex_t>> triplets;
			for (int i = 0; i < cb_bonds[bond_type].size(); ++i)
			{
				triplets.push_back({i, cb_bonds[bond_type][i], complex_t(0.,
					-std::sin(action(vertex, i, cb_bonds[bond_type][i])))});
				triplets.push_back({i, i, complex_t(std::cos(action(vertex, i,
					cb_bonds[bond_type][i])), 0.)});
			}
			v.setFromTriplets(triplets.begin(), triplets.end());
			return v;
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_n; n > tau_m; --n)
			{
				std::vector<sparse_t> h_cb;
				for (int i = 0; i < cb_bonds.size(); ++i)
					h_cb.push_back(vertex_matrix(i, vertices[species][n-1]));
				b *= h_cb[0] * h_cb[1] * h_cb[2] * h_cb[1] * h_cb[0];
			}
			return b;
		}

		void advance_forward()
		{
			for (int i = 0; i < 2; ++i)
			{
				dmatrix_t b = propagator(i, tau[i] + 1, tau[i]);
				equal_time_gf[i] = b * equal_time_gf[i] * b.inverse();
				++tau[i];
			}
		}

		void advance_backward()
		{
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
				dmatrix_t b = propagator(i, n*param.n_delta, (n-1)*param.n_delta);
				stabilizer.stabilize_backward(i, n, b);
			}
		}

		double try_ising_flip(int species, int i, int j)
		{
			/*
			dmatrix_t h_old = propagator(species, tau[species], tau[species] - 1);
			vertices[species][tau[species]-1](i, j) *= -1.;
			delta = propagator(species, tau[species], tau[species] - 1)
				* h_old.inverse() - id;
			dmatrix_t x = id + delta;
			x.noalias() -= delta * equal_time_gf[species];
			return std::abs(x.determinant());
			*/
			auto& vertex = vertices[species][tau[species]-1];
			double sigma = vertex(i, j);
			int bond_type = get_bond_type({i, j});
			if (bond_type < cb_bonds.size() - 1)
				return 0.;
			last_flip = {i, j};
			complex_t c = {std::cos(action(-sigma, bond_type)
				- action(sigma, bond_type)), 0.};
			complex_t s = {0., std::sin(action(-sigma, bond_type)
				- action(sigma, bond_type))};
			delta << c - 1., s, s, c - 1.;
	
			if (bond_type == 1)
				equal_time_gf[species] = inv_vertex_matrix(0, vertex)
					* equal_time_gf[species] * vertex_matrix(0, vertex);
			else if (bond_type == 2)
				equal_time_gf[species] = inv_vertex_matrix(1, vertex)
					* inv_vertex_matrix(0, vertex) * equal_time_gf[species]
					* vertex_matrix(0, vertex) * vertex_matrix(1, vertex);

			dmatrix_t x = id;
			x.row(i) -= delta(0, 0) * equal_time_gf[species].row(i)
				+ delta(0, 1) * equal_time_gf[species].row(j);
			x.row(j) -= delta(1, 0) * equal_time_gf[species].row(i)
				+ delta(1, 1) * equal_time_gf[species].row(j);
			x(i, i) += delta(0, 0); x(i, j) += delta(0, 1);
			x(j, i) += delta(1, 0); x(j, j) += delta(1, 1);
			return std::abs(x.determinant());
		}

		void undo_ising_flip(int species, int i, int j)
		{
			vertices[species][tau[species]-1](i, j) *= -1.;
		}

		void update_equal_time_gf_after_flip(int species)
		{
			/*
			Eigen::ComplexEigenSolver<dmatrix_t> solver(delta);
			dmatrix_t V = solver.eigenvectors();
			Eigen::VectorXcd ev = solver.eigenvalues();
			equal_time_gf[species] = (V.inverse() * equal_time_gf[species] * V)
				.eval();
			for (int i = 0; i < delta.rows(); ++i)
			{
				dmatrix_t g = equal_time_gf[species];
				for (int x = 0; x < equal_time_gf[species].rows(); ++x)
					for (int y = 0; y < equal_time_gf[species].cols(); ++y)
						equal_time_gf[species](x, y) -= g(x, i) * ev[i]
							* ((i == y ? 1.0 : 0.0) - g(i, y))
							/ (1.0 + ev[i] * (1. - g(i, i)));
			}
			equal_time_gf[species] = (V * equal_time_gf[species] * V.inverse())
				.eval();
			*/
			dmatrix_t g_old = equal_time_gf[species];
			int indices[2] = {last_flip.first, last_flip.second};
			auto& vertex = vertices[species][tau[species]-1];
			dmatrix_t Delta = dmatrix_t::Zero(l.n_sites(), l.n_sites());
			Delta(indices[0], indices[0]) = delta(0, 0);
			Delta(indices[0], indices[1]) = delta(0, 1);
			Delta(indices[1], indices[0]) = delta(1, 0);
			Delta(indices[1], indices[1]) = delta(1, 1);
			Eigen::ComplexEigenSolver<dmatrix_t> solver(Delta);
			dmatrix_t V = solver.eigenvectors();
			print_matrix(V);
		
			solver.compute(delta);
			dmatrix_t U = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			U(indices[0], indices[0]) = solver.eigenvectors()(0, 0);
			U(indices[0], indices[1]) = solver.eigenvectors()(0, 1);
			U(indices[1], indices[0]) = solver.eigenvectors()(1, 0);
			U(indices[1], indices[1]) = solver.eigenvectors()(1, 1);
			print_matrix(U);
			print_matrix(U.inverse());

			matrix_t<2, 2> u = solver.eigenvectors();
			matrix_t<2, 2> u_inv = u.inverse();
			matrix_t<2, 2> g(2, 2);
			g << equal_time_gf[species](indices[0], indices[0]),
				equal_time_gf[species](indices[0], indices[1]),
				equal_time_gf[species](indices[1], indices[0]),
				equal_time_gf[species](indices[1], indices[1]);
			g = u_inv * g * u;
			equal_time_gf[species](indices[0], indices[0]) = g(0, 0);
			equal_time_gf[species](indices[0], indices[1]) = g(0, 1);
			equal_time_gf[species](indices[1], indices[0]) = g(1, 0);
			equal_time_gf[species](indices[1], indices[1]) = g(1, 1);
			auto& ev = solver.eigenvalues();
			
			print_matrix(U.inverse() * g_old * U);
			print_matrix(equal_time_gf[species]);
			print_matrix(equal_time_gf[species] - U.inverse() * g_old * U);
			for (int i = 0; i < delta.rows(); ++i)
			{
				dmatrix_t g = equal_time_gf[species];
				for (int x = 0; x < equal_time_gf[species].rows(); ++x)
					for (int y = 0; y < equal_time_gf[species].cols(); ++y)
						equal_time_gf[species](x, y) -= g(x, indices[i]) * ev[i]
							* ((indices[i] == y ? 1.0 : 0.0) - g(indices[i], y))
							/ (1.0 + ev[i] * (1. - g(indices[i], indices[i])));
			}
			equal_time_gf[species] = (U * equal_time_gf[species] * U.inverse())
				.eval();
			
			equal_time_gf[species] = vertex_matrix(0, vertex)
				* vertex_matrix(1, vertex) * equal_time_gf[species]
				* inv_vertex_matrix(1, vertex) * inv_vertex_matrix(0, vertex);
			vertex(indices[0], indices[1]) *= -1.;

			std::cout << "new" << std::endl;
			print_matrix(equal_time_gf[species]);
			std::cout << "correct" << std::endl;
			print_matrix((id + propagator(species, tau[species], 0)
				* propagator(species, max_tau, tau[species])).inverse());
			
		}

		void static_measure(std::vector<double>& c, double& m2)
		{
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf[0](j, i)
							* equal_time_gf[1](j, i));
						double im = std::imag(equal_time_gf[0](j, i)
							* equal_time_gf[1](j, i));
						//Correlation function
						c[l.distance(i, j)] += re / l.n_sites();
						//M2 structure factor
						m2 += l.parity(i) * l.parity(j) * re
							/ std::pow(l.n_sites(), 2);
					}
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
			Eigen::IOFormat clean(6, 0, ", ", "\n", "[", "]");
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
		dmatrix_t id_2;
		dmatrix_t delta;
		std::pair<int, int> last_flip;
		arg_t last_vertex;
		arg_t flipped_vertex;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
