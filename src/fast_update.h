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
				gf_buffer(std::vector<dmatrix_t>(2)),
				gf_buffer_partial_vertex(std::vector<int>(2)),
				gf_buffer_tau(std::vector<int>(2)),
				stabilizer{measure, equal_time_gf, time_displaced_gf}
		{}

		void serialize(odump& out)
		{
			/*
			int size = aux_spins.size();
			out.write(size);
			for (arg_t& v : aux_spins)
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
				aux_spins.push_back(v);
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
			delta.resize(2, dmatrix_t(2, 2));
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

		int get_partial_vertex(int species) const
		{
			return partial_vertex[species];
		}
		
		double action(int species, const arg_t& x, int i, int j) const
		{
			double a = (get_bond_type({i, j}) < cb_bonds.size() - 1) ? 0.5 : 1.0;
			double sign = (i < j) ? 1. : -1.;
			if (species == 1)
				sign *= l.parity(i) * l.parity(j);
			if (l.distance(i, j) == 1)
				return a * sign * (param.t * param.dtau - param.lambda * x(i, j));
			else
				return 0.;
		}
		
		double action(int species, double x, int i, int j) const
		{
			double a = (get_bond_type({i, j}) < cb_bonds.size() - 1) ? 0.5 : 1.0;
			double sign = (i < j) ? 1. : -1.;
			if (species == 1)
				sign *= l.parity(i) * l.parity(j);
			if (l.distance(i, j) == 1)
				return a * sign * (param.t * param.dtau - param.lambda * x);
			else
				return 0.;
		}
		
		const arg_t& vertex(int species, int index)
		{
			return aux_spins[species][index-1]; 
		}

		int get_tau(int species)
		{
			return tau[species];
		}

		int get_max_tau()
		{
			return max_tau;
		}

		const std::map<int, int>& get_cb_bonds(int i) const
		{
			return cb_bonds[i];
		}

		void flip_spin(int species, const std::pair<int, int>& b)
		{
			aux_spins[species][tau[species]-1](b.first, b.second) *= -1.;
		}

		void buffer_equal_time_gf()
		{
			for (int i = 0; i < equal_time_gf.size(); ++i)
			{
				gf_buffer[i] = equal_time_gf[i];
				gf_buffer_partial_vertex[i] = partial_vertex[i];
				gf_buffer_tau[i] = tau[i];
			}
		}

		void reset_equal_time_gf_to_buffer()
		{
			for (int i = 0; i < equal_time_gf.size(); ++i)
			{
				equal_time_gf[i] = gf_buffer[i];
				partial_vertex[i] = gf_buffer_partial_vertex[i];
				tau[i] = gf_buffer_tau[i];
			}
		}

		void build(boost::multi_array<arg_t, 2>& args)
		{
			aux_spins.resize(boost::extents[args.shape()[0]][args.shape()[1]]);
			aux_spins = args;
			max_tau = aux_spins.shape()[1];
			tau = {max_tau, max_tau};
			partial_vertex = {0, 0};
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (aux_spins.shape()[1] == 0) return;
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
		
		sparse_t vertex_matrix(int species, int bond_type, const arg_t& vertex)
		{
			sparse_t v(l.n_sites(), l.n_sites());
			std::vector<Eigen::Triplet<complex_t>> triplets;
			for (int i = 0; i < cb_bonds[bond_type].size(); ++i)
			{
				triplets.push_back({i, cb_bonds[bond_type][i], complex_t(0.,
					std::sinh(action(species, vertex, i, cb_bonds[bond_type][i])))});
				triplets.push_back({i, i, complex_t(std::cosh(action(species,vertex,
					i, cb_bonds[bond_type][i])), 0.)});
			}
			v.setFromTriplets(triplets.begin(), triplets.end());
			return v;
		}
		
		sparse_t inv_vertex_matrix(int species, int bond_type,const arg_t& vertex)
		{
			sparse_t v(l.n_sites(), l.n_sites());
			std::vector<Eigen::Triplet<complex_t>> triplets;
			for (int i = 0; i < cb_bonds[bond_type].size(); ++i)
			{
				triplets.push_back({i, cb_bonds[bond_type][i], complex_t(0.,
					-std::sinh(action(species, vertex, i,cb_bonds[bond_type][i])))});
				triplets.push_back({i,i, complex_t(std::cosh(action(species, vertex,
					i, cb_bonds[bond_type][i])), 0.)});
			}
			v.setFromTriplets(triplets.begin(), triplets.end());
			return v;
		}

		void multiply_vertex_from_left(int species, dmatrix_t& m,
			int bond_type, const arg_t& vertex, int inv)
		{
			dmatrix_t old_m = m;
			for (int i = 0; i < m.cols(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				complex_t c = {std::cosh(action(species, vertex, i, j))};
				complex_t s = {0., inv * std::sinh(action(species, vertex, i, j))};
				m.row(i) = old_m.row(i) * c + old_m.row(j) * s;
			}
		}

		void multiply_vertex_from_right(int species, dmatrix_t& m,
			int bond_type, const arg_t& vertex, int inv)
		{
			dmatrix_t old_m = m;
			for (int i = 0; i < m.rows(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				complex_t c = {std::cosh(action(species, vertex, i, j))};
				complex_t s = {0., -inv * std::sinh(action(species, vertex, i, j))};
				m.col(i) = old_m.col(i) * c + old_m.col(j) * s;
			}
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_n; n > tau_m; --n)
			{
//				std::vector<sparse_t> h_cb;
//				for (int i = 0; i < cb_bonds.size(); ++i)
//					h_cb.push_back(vertex_matrix(species, i, aux_spins[species][n-1]));
//				b *= h_cb[0] * h_cb[1] * h_cb[2] * h_cb[1] * h_cb[0];
				auto& vertex = aux_spins[species][n-1];
				multiply_vertex_from_right(species, b, 0, vertex, 1);
				multiply_vertex_from_right(species, b, 1, vertex, 1);
				multiply_vertex_from_right(species, b, 2, vertex, 1);
				multiply_vertex_from_right(species, b, 1, vertex, 1);
				multiply_vertex_from_right(species, b, 0, vertex, 1);
			}
			return b;
		}

		void partial_advance(int species, int partial_n)
		{
			int& p = partial_vertex[species];
			auto& vertex = aux_spins[species][tau[species]-1];
			while (partial_n > p)
			{
//				equal_time_gf[species] = inv_vertex_matrix(species, p % 3,
//					aux_spins[species][tau[species]-1]) * equal_time_gf[species]
//					* vertex_matrix(species, p % 3, aux_spins[species][tau[species]-1]);

				int bond_type = (p < cb_bonds.size()) ? p : 2*(cb_bonds.size()-1)-p;
				multiply_vertex_from_left(species, equal_time_gf[species],
					bond_type, vertex, -1);
				multiply_vertex_from_right(species, equal_time_gf[species],
					bond_type, vertex, 1);
				++p;
			}
			while (partial_n < p)
			{
				--p;
//				equal_time_gf[species] = vertex_matrix(species, p % 3,
//					aux_spins[species][tau[species]-1]) * equal_time_gf[species]
//					* inv_vertex_matrix(species, p % 3, aux_spins[species][tau[species]-1]);

				int bond_type = (p < cb_bonds.size()) ? p : 2*(cb_bonds.size()-1)-p;
				multiply_vertex_from_left(species, equal_time_gf[species],
					bond_type, vertex, 1);
				multiply_vertex_from_right(species, equal_time_gf[species],
					bond_type, vertex, -1);
			}
		}

		void advance_forward()
		{
			for (int i = 0; i < 2; ++i)
			{
//				dmatrix_t b = propagator(i, tau[i] + 1, tau[i]);
//				equal_time_gf[i] = b * equal_time_gf[i] * b.inverse();

				auto& vertex = aux_spins[i][tau[i]];
				multiply_vertex_from_left(i, equal_time_gf[i], 0, vertex, 1);
				multiply_vertex_from_left(i, equal_time_gf[i], 1, vertex, 1);
				multiply_vertex_from_left(i, equal_time_gf[i], 2, vertex, 1);
				multiply_vertex_from_left(i, equal_time_gf[i], 1, vertex, 1);
				multiply_vertex_from_left(i, equal_time_gf[i], 0, vertex, 1);
				multiply_vertex_from_right(i, equal_time_gf[i], 0, vertex, -1);
				multiply_vertex_from_right(i, equal_time_gf[i], 1, vertex, -1);
				multiply_vertex_from_right(i, equal_time_gf[i], 2, vertex, -1);
				multiply_vertex_from_right(i, equal_time_gf[i], 1, vertex, -1);
				multiply_vertex_from_right(i, equal_time_gf[i], 0, vertex, -1);
				++tau[i];
			}
		}

		void advance_backward()
		{
			for (int i = 0; i < 2; ++i)
			{
//				dmatrix_t b = propagator(i, tau[i], tau[i] - 1);
//				equal_time_gf[i] = b.inverse() * equal_time_gf[i] * b;
				
				auto& vertex = aux_spins[i][tau[i] - 1];
				multiply_vertex_from_left(i, equal_time_gf[i], 0, vertex, -1);
				multiply_vertex_from_left(i, equal_time_gf[i], 1, vertex, -1);
				multiply_vertex_from_left(i, equal_time_gf[i], 2, vertex, -1);
				multiply_vertex_from_left(i, equal_time_gf[i], 1, vertex, -1);
				multiply_vertex_from_left(i, equal_time_gf[i], 0, vertex, -1);
				multiply_vertex_from_right(i, equal_time_gf[i], 0, vertex, 1);
				multiply_vertex_from_right(i, equal_time_gf[i], 1, vertex, 1);
				multiply_vertex_from_right(i, equal_time_gf[i], 2, vertex, 1);
				multiply_vertex_from_right(i, equal_time_gf[i], 1, vertex, 1);
				multiply_vertex_from_right(i, equal_time_gf[i], 0, vertex, 1);
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
				dmatrix_t b = propagator(i, (n+1)*param.n_delta, n*param.n_delta);
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
			auto& vertex = aux_spins[species][tau[species]-1];
			double sigma = vertex(i, j);
			last_flip = {i, j};
			for (int a = 0; a < 2; ++a)
			{
				complex_t cp = {std::cosh(action(a, -sigma, std::min(i,j),
					std::max(i,j)))};
				complex_t c = {std::cosh(action(a, sigma, std::min(i,j),
					std::max(i,j)))};
				complex_t sp={0,std::sinh(action(a, -sigma, std::min(i,j),
					std::max(i,j)))};
				complex_t s={0,std::sinh(action(a, sigma, std::min(i,j),
					std::max(i,j)))};
				delta[a] << cp*c + sp*s - 1., -cp*s + sp*c, -sp*c + cp*s,
					sp*s + cp*c-1.;
			}
	
			matrix_t<2, 2> x(2, 2);
			x(0, 0) = 1. + delta[species](0, 0) - (delta[species](0, 0)
				* equal_time_gf[species](i, i) + delta[species](0, 1)
				* equal_time_gf[species](j, i));
			x(0, 1) = delta[species](0, 1) - (delta[species](0, 0)
				* equal_time_gf[species](i, j) + delta[species](0, 1)
				* equal_time_gf[species](j, j));
			x(1, 0) = delta[species](1, 0) - (delta[species](1, 0)
				* equal_time_gf[species](i, i) + delta[species](1, 1)
				* equal_time_gf[species](j, i));
			x(1, 1) = 1. + delta[species](1, 1) - (delta[species](1, 0)
				* equal_time_gf[species](i, j) + delta[species](1, 1)
				* equal_time_gf[species](j, j));
			return std::abs(x.determinant());
		}

		void update_equal_time_gf_after_flip(int species)
		{
			int indices[2] = {std::min(last_flip.first, last_flip.second),
				std::max(last_flip.first, last_flip.second)};

			complex_t i = {0, 1.};
			complex_t ev[] = {delta[species](0, 0) - i*delta[species](0, 1),
				delta[species](0, 0) + i*delta[species](0, 1)};
			matrix_t<2, 2> u(2, 2);
			u << i, -i, 1., 1;
			matrix_t<2, 2> u_inv(2, 2);
			u_inv << -i/2., 1./2., i/2., 1./2.;

			// u_inv * G
			dmatrix_t row_0 = equal_time_gf[species].row(indices[0]);
			equal_time_gf[species].row(indices[0]) *= u_inv(0, 0);
			equal_time_gf[species].row(indices[0]).noalias() += u_inv(0, 1)
				* equal_time_gf[species].row(indices[1]);
			equal_time_gf[species].row(indices[1]) *= u_inv(1, 1);
			equal_time_gf[species].row(indices[1]).noalias() += u_inv(1, 0)*row_0;

			// G' * u
			dmatrix_t col_0 = equal_time_gf[species].col(indices[0]);
			equal_time_gf[species].col(indices[0]) *= u(0, 0);
			equal_time_gf[species].col(indices[0]).noalias() +=
				equal_time_gf[species].col(indices[1]) * u(1, 0);
			equal_time_gf[species].col(indices[1]) *= u(1, 1);
			equal_time_gf[species].col(indices[1]).noalias() += col_0 * u(0, 1);
			
			// Sherman-Morrison
			for (int i = 0; i < delta[species].rows(); ++i)
			{
				dmatrix_t g = equal_time_gf[species];
				for (int x = 0; x < equal_time_gf[species].rows(); ++x)
					for (int y = 0; y < equal_time_gf[species].cols(); ++y)
						equal_time_gf[species](x, y) -= g(x, indices[i]) * ev[i]
							* ((indices[i] == y ? 1.0 : 0.0) - g(indices[i], y))
							/ (1.0 + ev[i] * (1. - g(indices[i], indices[i])));
			}

			// u * G
			row_0 = equal_time_gf[species].row(indices[0]);
			equal_time_gf[species].row(indices[0]) *= u(0, 0);
			equal_time_gf[species].row(indices[0]).noalias() += u(0, 1)
				* equal_time_gf[species].row(indices[1]);
			equal_time_gf[species].row(indices[1]) *= u(1, 1);
			equal_time_gf[species].row(indices[1]).noalias() += u(1, 0) * row_0;

			// G' * u_inv
			col_0 = equal_time_gf[species].col(indices[0]);
			equal_time_gf[species].col(indices[0]) *= u_inv(0, 0);
			equal_time_gf[species].col(indices[0]).noalias() +=
				equal_time_gf[species].col(indices[1]) * u_inv(1, 0);
			equal_time_gf[species].col(indices[1]) *= u_inv(1, 1);
			equal_time_gf[species].col(indices[1]).noalias() += col_0*u_inv(0, 1);
		}

		void static_measure(std::vector<double>& c, double& m2)
		{
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf[0](j, i)
							* equal_time_gf[0](j, i));
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
		std::vector<int> partial_vertex;
		int max_tau;
		boost::multi_array<arg_t, 2> aux_spins;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		dmatrix_t M;
		std::vector<dmatrix_t> equal_time_gf;
		std::vector<dmatrix_t> time_displaced_gf;
		std::vector<dmatrix_t> gf_buffer;
		std::vector<int> gf_buffer_partial_vertex;
		std::vector<int> gf_buffer_tau;
		dmatrix_t id;
		dmatrix_t id_2;
		std::vector<dmatrix_t> delta;
		std::pair<int, int> last_flip;
		arg_t last_vertex;
		arg_t flipped_vertex;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
