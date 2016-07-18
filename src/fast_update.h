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
#include "wick_base.h"

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
				cb_bonds(3),
				update_time_displaced_gf(false),
				n_species(1),
				equal_time_gf(std::vector<dmatrix_t>(n_species)),
				time_displaced_gf(std::vector<dmatrix_t>(n_species)),
				proj_W_l(std::vector<dmatrix_t>(n_species)),
				proj_W_r(std::vector<dmatrix_t>(n_species)),
				proj_W(std::vector<dmatrix_t>(n_species)),
				gf_buffer(std::vector<dmatrix_t>(n_species)),
				gf_buffer_partial_vertex(std::vector<int>(n_species)),
				gf_buffer_tau(std::vector<int>(n_species)),
				stabilizer{measure, equal_time_gf, time_displaced_gf,
				proj_W_l, proj_W_r, proj_W, n_species}
		{
			tau.resize(n_species, 1);
		}

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
			delta.resize(2, dmatrix_t(2, 2));
			id = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			id_2 = dmatrix_t::Identity(2, 2);
			for (int i = 0; i < n_species; ++i)
			{
				equal_time_gf[i] = 0.5 * id;
				time_displaced_gf[i] = 0.5 * id;
			}
			expH0 = dmatrix_t::Zero(l.n_sites(), l.n_sites());
			for (auto& a : l.bonds("nearest neighbors"))
			{
				double sign = (a.first < a.second) ? 1. : -1.;
				expH0(a.first, a.second) = {0., -sign * param.t * param.dtau / 2.};
			}
			Eigen::ComplexEigenSolver<dmatrix_t> solver(expH0);
			expH0.setZero();
			for (int i = 0; i < expH0.rows(); ++i)
				expH0(i, i) = std::exp(solver.eigenvalues()[i]);
			expH0 = solver.eigenvectors() * expH0 * solver.eigenvectors()
				.inverse();
			invExpH0 = expH0.inverse();
			P = solver.eigenvectors().block(0, 0, l.n_sites(), l.n_sites() / 2);
			Pt = P.adjoint();
			create_checkerboard();
			stabilizer.set_method(param.use_projector);
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
			if (l.distance(i, j) == 1)
//				return a * sign * (param.t * param.dtau + param.lambda * x(i, j));
				return a * sign * param.lambda * x(i, j);
			else
				return 0.;
		}
		
		double action(int species, double x, int i, int j) const
		{
			double a = (get_bond_type({i, j}) < cb_bonds.size() - 1) ? 0.5 : 1.0;
			double sign = (i < j) ? 1. : -1.;
			if (l.distance(i, j) == 1)
//				return a * sign * (param.t * param.dtau + param.lambda * x);
				return a * sign * param.lambda * x;
			else
				return 0.;
		}
		
		const arg_t& vertex(int index)
		{
			return aux_spins[index-1]; 
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

		void flip_spin(const std::pair<int, int>& b)
		{
			aux_spins[tau[0]-1](b.first, b.second) *= -1.;
		}

		void buffer_equal_time_gf()
		{
			for (int i = 0; i < n_species; ++i)
			{
				gf_buffer[i] = equal_time_gf[i];
				gf_buffer_partial_vertex[i] = partial_vertex[i];
				gf_buffer_tau[i] = tau[i];
			}
		}

		void reset_equal_time_gf_to_buffer()
		{
			for (int i = 0; i < n_species; ++i)
			{
				equal_time_gf[i] = gf_buffer[i];
				partial_vertex[i] = gf_buffer_partial_vertex[i];
				tau[i] = gf_buffer_tau[i];
			}
		}
		
		void enable_time_displaced_gf(int direction)
		{
			update_time_displaced_gf = true;
			stabilizer.enable_time_displaced_gf(direction);
		}

		void disable_time_displaced_gf()
		{
			update_time_displaced_gf = false;
			stabilizer.disable_time_displaced_gf();
		}

		void build(std::vector<arg_t>& args)
		{
			aux_spins.swap(args);
			max_tau = aux_spins.size();
			tau = {max_tau, max_tau};
			partial_vertex = {0, 0};
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, l.n_sites());
			rebuild();
		}

		void rebuild()
		{
			if (aux_spins.size() == 0) return;
			for (int i = 0; i < n_species; ++i)
			{
				if (param.use_projector)
				{
					stabilizer.set_proj_l(i, n_intervals, id, Pt);
					for (int n = n_intervals - 1; n >= 0; --n)
					{
						dmatrix_t b = propagator(i, (n + 1) * param.n_delta,
							n * param.n_delta);
						stabilizer.set_proj_l(i, n, b, Pt);
					}
					stabilizer.set_proj_r(i, 0, id, P);
					for (int n = 1; n <= n_intervals; ++n)
					{
						dmatrix_t b = propagator(i, n * param.n_delta,
							(n - 1) * param.n_delta);
						stabilizer.set_proj_r(i, n, b, P);
					}
				}
				else
				{
					for (int n = 1; n <= n_intervals; ++n)
					{
						dmatrix_t b = propagator(i, n * param.n_delta,
							(n - 1) * param.n_delta);
						stabilizer.set(i, n, b);
					}
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
			for (int i = 0; i < m.rows(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				complex_t c = {std::cosh(action(species, vertex, i, j))};
				complex_t s = {0., std::sinh(action(species, vertex, i, j))};
				m.row(i) = old_m.row(i) * c + old_m.row(j) * s * inv;
			}
		}

		void multiply_vertex_from_right(int species, dmatrix_t& m,
			int bond_type, const arg_t& vertex, int inv)
		{
			dmatrix_t old_m = m;
			for (int i = 0; i < m.cols(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				complex_t c = {std::cosh(action(species, vertex, i, j))};
				complex_t s = {0., std::sinh(action(species, vertex, i, j))};
				m.col(i) = old_m.col(i) * c - old_m.col(j) * s * inv;
			}
		}

		void prepare_flip(int species)
		{
			equal_time_gf[species] = invExpH0 * equal_time_gf[species] * expH0;
		}

		void prepare_measurement(int species)
		{
			equal_time_gf[species] = expH0 * equal_time_gf[species] * invExpH0;
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
//			return exact_propagator(species, tau_n, tau_m);
			dmatrix_t b = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			for (int n = tau_n; n > tau_m; --n)
			{
				auto& vertex = aux_spins[n-1];
				b *= expH0;
				multiply_vertex_from_right(species, b, 0, vertex, 1);
				multiply_vertex_from_right(species, b, 1, vertex, 1);
				multiply_vertex_from_right(species, b, 2, vertex, 1);
				multiply_vertex_from_right(species, b, 1, vertex, 1);
				multiply_vertex_from_right(species, b, 0, vertex, 1);
				b *= expH0;
			}
			return b;
		}
		
		dmatrix_t exact_propagator(int species, int tau_n, int tau_m,
			double s=1.)
		{
			dmatrix_t x = dmatrix_t::Identity(l.n_sites(), l.n_sites());
			Eigen::ComplexEigenSolver<dmatrix_t> solver;
			for (int n = tau_n; n > tau_m; --n)
			{
				auto& vertex = aux_spins[n-1];
				dmatrix_t h = dmatrix_t::Zero(l.n_sites(), l.n_sites());
				for (auto& b : l.bonds("nearest neighbors"))
				{
					double sign = (b.first < b.second) ? 1. : -1.;
					h(b.first, b.second) = {0., sign * (param.t * param.dtau
						+ param.lambda * vertex(b.first, b.second))};
				}
				solver.compute(h);
				dmatrix_t es = solver.eigenvectors();
				dmatrix_t D = solver.eigenvalues().asDiagonal();
				for (int i = 0; i < D.rows(); ++i)
					D(i, i) = std::exp(D(i, i));
				x *= es * D * es.inverse();
			}
			return x;
		}

		void partial_advance(int species, int partial_n)
		{
			int& p = partial_vertex[species];
			auto& vertex = aux_spins[tau[species]-1];
			while (partial_n > p)
			{
				int bond_type = (p < cb_bonds.size()) ? p : 2*(cb_bonds.size()-1)-p;
				if (param.use_projector)
				{
					multiply_vertex_from_left(species, proj_W_r[species],
						bond_type, vertex, -1);
					multiply_vertex_from_right(species, proj_W_l[species],
						bond_type, vertex, 1);
				}
				else
				{
					multiply_vertex_from_left(species, equal_time_gf[species],
						bond_type, vertex, -1);
					multiply_vertex_from_right(species, equal_time_gf[species],
						bond_type, vertex, 1);
				}
				++p;
			}
			while (partial_n < p)
			{
				--p;
				int bond_type = (p < cb_bonds.size()) ? p : 2*(cb_bonds.size()-1)-p;
				if (param.use_projector)
				{
					multiply_vertex_from_left(species, proj_W_r[species],
						bond_type, vertex, 1);
					multiply_vertex_from_right(species, proj_W_l[species],
						bond_type, vertex, -1);
				}
				else
				{
					multiply_vertex_from_left(species, equal_time_gf[species],
						bond_type, vertex, 1);
					multiply_vertex_from_right(species, equal_time_gf[species],
						bond_type, vertex, -1);
				}
			}
		}

		void advance_forward()
		{
			for (int i = 0; i < n_species; ++i)
			{
				auto& vertex = aux_spins[tau[i]];
				if (update_time_displaced_gf)
				{
					time_displaced_gf[i] = expH0 * time_displaced_gf[i];
					multiply_vertex_from_left(i, time_displaced_gf[i], 0, vertex, 1);
					multiply_vertex_from_left(i, time_displaced_gf[i], 1, vertex, 1);
					multiply_vertex_from_left(i, time_displaced_gf[i], 2, vertex, 1);
					multiply_vertex_from_left(i, time_displaced_gf[i], 1, vertex, 1);
					multiply_vertex_from_left(i, time_displaced_gf[i], 0, vertex, 1);
					time_displaced_gf[i] = expH0 * time_displaced_gf[i];
				}
				if (param.use_projector)
				{
					proj_W_r[i] = expH0 * proj_W_r[i];
					multiply_vertex_from_left(i, proj_W_r[i], 0, vertex, 1);
					multiply_vertex_from_left(i, proj_W_r[i], 1, vertex, 1);
					multiply_vertex_from_left(i, proj_W_r[i], 2, vertex, 1);
					multiply_vertex_from_left(i, proj_W_r[i], 1, vertex, 1);
					multiply_vertex_from_left(i, proj_W_r[i], 0, vertex, 1);
					proj_W_r[i] = expH0 * proj_W_r[i];
					
					proj_W_l[i] = proj_W_l[i] * invExpH0;
					multiply_vertex_from_right(i, proj_W_l[i], 0, vertex, -1);
					multiply_vertex_from_right(i, proj_W_l[i], 1, vertex, -1);
					multiply_vertex_from_right(i, proj_W_l[i], 2, vertex, -1);
					multiply_vertex_from_right(i, proj_W_l[i], 1, vertex, -1);
					multiply_vertex_from_right(i, proj_W_l[i], 0, vertex, -1);
					proj_W_l[i] = proj_W_l[i] * invExpH0;
				}
				else
				{
					dmatrix_t& gf = equal_time_gf[i];
					gf = expH0 * gf * invExpH0;
					multiply_vertex_from_left(i, gf, 0, vertex, 1);
					multiply_vertex_from_left(i, gf, 1, vertex, 1);
					multiply_vertex_from_left(i, gf, 2, vertex, 1);
					multiply_vertex_from_left(i, gf, 1, vertex, 1);
					multiply_vertex_from_left(i, gf, 0, vertex, 1);
					multiply_vertex_from_right(i, gf, 0, vertex, -1);
					multiply_vertex_from_right(i, gf, 1, vertex, -1);
					multiply_vertex_from_right(i, gf, 2, vertex, -1);
					multiply_vertex_from_right(i, gf, 1, vertex, -1);
					multiply_vertex_from_right(i, gf, 0, vertex, -1);
					gf = expH0 * gf * invExpH0;
				}
				++tau[i];
			}
		}

		void advance_backward()
		{
			for (int i = 0; i < n_species; ++i)
			{
				auto& vertex = aux_spins[tau[i] - 1];
				if (update_time_displaced_gf)
				{
					time_displaced_gf[i] = time_displaced_gf[i] * expH0;
					multiply_vertex_from_right(i,time_displaced_gf[i],0,vertex,1);
					multiply_vertex_from_right(i,time_displaced_gf[i],1,vertex,1);
					multiply_vertex_from_right(i,time_displaced_gf[i],2,vertex,1);
					multiply_vertex_from_right(i,time_displaced_gf[i],1,vertex,1);
					multiply_vertex_from_right(i,time_displaced_gf[i],0,vertex,1);
					time_displaced_gf[i] = time_displaced_gf[i] * expH0;
				}
				if (param.use_projector)
				{
					proj_W_r[i] = invExpH0 * proj_W_r[i];
					multiply_vertex_from_left(i, proj_W_r[i], 0, vertex, -1);
					multiply_vertex_from_left(i, proj_W_r[i], 1, vertex, -1);
					multiply_vertex_from_left(i, proj_W_r[i], 2, vertex, -1);
					multiply_vertex_from_left(i, proj_W_r[i], 1, vertex, -1);
					multiply_vertex_from_left(i, proj_W_r[i], 0, vertex, -1);
					proj_W_r[i] = invExpH0 * proj_W_r[i];
					
					proj_W_l[i] = proj_W_l[i] * expH0;
					multiply_vertex_from_right(i, proj_W_l[i], 0, vertex, 1);
					multiply_vertex_from_right(i, proj_W_l[i], 1, vertex, 1);
					multiply_vertex_from_right(i, proj_W_l[i], 2, vertex, 1);
					multiply_vertex_from_right(i, proj_W_l[i], 1, vertex, 1);
					multiply_vertex_from_right(i, proj_W_l[i], 0, vertex, 1);
					proj_W_l[i] = proj_W_l[i] * expH0;
				}
				else
				{
					dmatrix_t& gf = equal_time_gf[i];
					gf = invExpH0 * gf * expH0;
					multiply_vertex_from_left(i, gf, 0, vertex, -1);
					multiply_vertex_from_left(i, gf, 1, vertex, -1);
					multiply_vertex_from_left(i, gf, 2, vertex, -1);
					multiply_vertex_from_left(i, gf, 1, vertex, -1);
					multiply_vertex_from_left(i, gf, 0, vertex, -1);
					multiply_vertex_from_right(i, gf, 0, vertex, 1);
					multiply_vertex_from_right(i, gf, 1, vertex, 1);
					multiply_vertex_from_right(i, gf, 2, vertex, 1);
					multiply_vertex_from_right(i, gf, 1, vertex, 1);
					multiply_vertex_from_right(i, gf, 0, vertex, 1);
					gf = invExpH0 * gf * expH0;
				}
				--tau[i];
			}
		}
		
		void stabilize_forward()
		{
			if (tau[0] % param.n_delta != 0)
					return;
			for (int i = 0; i < n_species; ++i)
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
			for (int i = 0; i < n_species; ++i)
			{
				//n = n_intervals, ..., 1 
				int n = tau[i] / param.n_delta + 1;
				dmatrix_t b = propagator(i, n*param.n_delta, (n-1)*param.n_delta);
				stabilizer.stabilize_backward(i, n, b);
			}
		}

		double try_ising_flip(int species, int i, int j)
		{
			auto& vertex = aux_spins[tau[species]-1];
			double sigma = vertex(i, j);
			last_flip = {i, j};
			int m = std::min(i, j), n = std::max(i, j);
			for (int a = 0; a < n_species; ++a)
			{
				complex_t cp = {std::cosh(action(a, -sigma, m, n))};
				complex_t c = {std::cosh(action(a, sigma, m, n))};
				complex_t sp = {0, std::sinh(action(a, -sigma, m, n))};
				complex_t s = {0, std::sinh(action(a, sigma, m, n))};
				delta[a] << cp*c + sp*s - 1., -cp*s + sp*c, -sp*c + cp*s,
					sp*s + cp*c-1.;
			}
	
			if (param.use_projector)
			{
				dmatrix_t b_l(P.cols(), 2);
				b_l.col(0) = proj_W_l[species].col(i);
				b_l.col(1) = proj_W_l[species].col(j);

				dmatrix_t delta_W_r(2, P.cols());
				delta_W_r.row(0) = delta[species](0, 0) * proj_W_r[species].row(i)
					+ delta[species](0, 1) * proj_W_r[species].row(j);
				delta_W_r.row(1) = delta[species](1, 0) * proj_W_r[species].row(i)
					+ delta[species](1, 1) * proj_W_r[species].row(j);

				dmatrix_t x = delta_W_r * proj_W[species] * b_l;
				return std::abs(x.determinant());
			}
			else
			{
				dmatrix_t& gf = equal_time_gf[species];
				matrix_t<2, 2> x(2, 2);
				x(0, 0) = 1. + delta[species](0, 0) - (delta[species](0, 0)
					* gf(m, m) + delta[species](0, 1) * gf(n, m));
				x(0, 1) = delta[species](0, 1) - (delta[species](0, 0)
					* gf(m, n) + delta[species](0, 1) * gf(n, n));
				x(1, 0) = delta[species](1, 0) - (delta[species](1, 0)
					* gf(m, m) + delta[species](1, 1) * gf(n, m));
				x(1, 1) = 1. + delta[species](1, 1) - (delta[species](1, 0)
					* gf(m, n) + delta[species](1, 1) * gf(n, n));
				return std::abs(x.determinant());
			}
		}
		
		double exact_try_ising_flip(int species, int i, int j)
		{
			auto& vertex = aux_spins[tau[species]-1];
			last_flip = {i, j};
			vertex(i, j) *= -1.;
			dmatrix_t v2 = exact_propagator(species, tau[species], tau[species]-1);
			vertex(i, j) *= -1.;
			dmatrix_t v1 = exact_propagator(species, tau[species], tau[species]-1);
			delta[species] = id + (v2*v1.inverse() - id) * (id
				- equal_time_gf[species]);
			return std::abs(delta[species].determinant());
		}

		void update_equal_time_gf_after_flip(int species)
		{
			int indices[2] = {std::min(last_flip.first, last_flip.second),
				std::max(last_flip.first, last_flip.second)};

			if (param.use_projector)
			{
				dmatrix_t delta_W_r(2, P.cols());
				delta_W_r.row(0) = delta[species](0, 0) * proj_W_r[species].row(indices[0])
					+ delta[species](0, 1) * proj_W_r[species].row(indices[1]);
				delta_W_r.row(1) = delta[species](1, 0) * proj_W_r[species].row(indices[0])
					+ delta[species](1, 1) * proj_W_r[species].row(indices[1]);
				
				dmatrix_t W_l_delta_W_r(P.cols(), P.cols());
				for (int i = 0; i < P.cols(); ++i)
					W_l_delta_W_r.row(i) = proj_W_l[species](i, indices[0]) * delta_W_r.row(0)
						+ proj_W_l[species](i, indices[1]) * delta_W_r.row(1);
				
				proj_W[species] -= proj_W[species] * W_l_delta_W_r * proj_W[species];
				proj_W_r[species].row(indices[0]).noalias() += delta_W_r.row(0);
				proj_W_r[species].row(indices[1]).noalias() += delta_W_r.row(1);
			}
			else
			{
				dmatrix_t& gf = equal_time_gf[species];
				matrix_t<2, 2> g(2, 2);
				g << gf(indices[0], indices[0]), gf(indices[0], indices[1]),
				gf(indices[1], indices[0]), gf(indices[1], indices[1]);
				dmatrix_t M = id_2 + (id_2 - g) * delta[species];
				dmatrix_t g_cols(l.n_sites(), 2);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				dmatrix_t g_rows(2, l.n_sites());
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				gf.noalias() += (g_cols * delta[species]) * (M.inverse() * g_rows);
			}
		}
		
		void exact_update_equal_time_gf_after_flip(int species)
		{
			equal_time_gf[species] = equal_time_gf[species] * delta[species]
				.inverse();
		}

		void static_measure(std::vector<double>& c, double& m2)
		{
			dmatrix_t* G;
			if (param.use_projector)
			{
				dmatrix_t g = id - proj_W_r[0] * proj_W[0] * proj_W_l[0];
				G = &g;
			}
			else
				G = &equal_time_gf[0];
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real((*G)(j, i) * (*G)(j, i));
						//Correlation function
						c[l.distance(i, j)] += re / l.n_sites();
						//M2 structure factor
						m2 += l.parity(i) * l.parity(j) * re
							/ std::pow(l.n_sites(), 2);
					}
		}
		
		void measure_dynamical_observable(std::vector<std::vector<double>>&
			dyn_tau, const std::vector<wick_base<dmatrix_t>>& obs)
		{
			// 1 = forward, -1 = backward
			int direction = tau[0] == 0 ? 1 : -1;
			dmatrix_t et_gf_0 = equal_time_gf[0];
			enable_time_displaced_gf(direction);
			time_displaced_gf[0] = equal_time_gf[0];
			for (int n = 0; n <= max_tau; ++n)
			{
				if (n % (max_tau / param.n_discrete_tau) == 0)
				{
					int t = n / (max_tau / param.n_discrete_tau);
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][t] = obs[i].get_obs(et_gf_0, equal_time_gf[0],
							time_displaced_gf[0]);
				}
				if (direction == 1 && tau[0] < max_tau)
				{
					advance_forward();
					stabilize_forward();
				}
				else if (direction == -1 && tau[0] > 0)
				{
					advance_backward();
					stabilize_backward();
				}
			}
			disable_time_displaced_gf();
			if (direction == 1)
				tau[0] = 0;
			else if (direction == -1)
				tau[0] = max_tau;
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
		std::vector<arg_t> aux_spins;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		bool update_time_displaced_gf;
		int n_species;
		std::vector<dmatrix_t> equal_time_gf;
		std::vector<dmatrix_t> time_displaced_gf;
		std::vector<dmatrix_t> proj_W_l;
		std::vector<dmatrix_t> proj_W_r;
		std::vector<dmatrix_t> proj_W;
		std::vector<dmatrix_t> gf_buffer;
		std::vector<int> gf_buffer_partial_vertex;
		std::vector<int> gf_buffer_tau;
		dmatrix_t id;
		dmatrix_t id_2;
		dmatrix_t expH0;
		dmatrix_t invExpH0;
		dmatrix_t P;
		dmatrix_t Pt;
		std::vector<dmatrix_t> delta;
		std::pair<int, int> last_flip;
		arg_t last_vertex;
		arg_t flipped_vertex;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
