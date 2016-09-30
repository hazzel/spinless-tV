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

template <class data_t, class index_t>
class SortIndicesInc
{
	protected:
		const data_t& data;
	public:
		SortIndicesInc(const data_t& data_) : data(data_) {}
		bool operator()(const index_t& i, const index_t& j) const
		{
			return data[i] < data[j];
		}
};

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

		fast_update(Random& rng_, const lattice& l_, const parameters& param_,
			measurements& measure_)
			: rng(rng_), l(l_), param(param_), measure(measure_),
				cb_bonds(3),
				update_time_displaced_gf(false),
				n_species(1),
				equal_time_gf(std::vector<dmatrix_t>(n_species)),
				time_displaced_gf(std::vector<dmatrix_t>(n_species)),
				proj_W_l(std::vector<dmatrix_t>(n_species)),
				proj_W_r(std::vector<dmatrix_t>(n_species)),
				proj_W(std::vector<dmatrix_t>(n_species)),
				gf_buffer(std::vector<dmatrix_t>(n_species)),
				W_l_buffer(std::vector<dmatrix_t>(n_species)),
				W_r_buffer(std::vector<dmatrix_t>(n_species)),
				W_buffer(std::vector<dmatrix_t>(n_species)),
				gf_buffer_partial_vertex(std::vector<int>(n_species)),
				gf_buffer_tau(std::vector<int>(n_species)),
				stabilizer{measure, equal_time_gf, time_displaced_gf,
				proj_W_l, proj_W_r, proj_W, n_species}
		{
			tau.resize(n_species, 1);
		}

		void serialize(odump& out)
		{
			int size = aux_spins.size();
			int cnt = 0;
			out.write(size);
			for (arg_t& v : aux_spins)
			{
				v.serialize(out);
				++cnt;
			}
		}

		void serialize(idump& in)
		{
			int size; in.read(size);
			aux_spins.resize(size);
			for (int i = 0; i < size; ++i)
			{
				arg_t v;
				v.serialize(in);
				aux_spins[i] = v;
			}
			max_tau = size;
			tau = {max_tau, max_tau};
			partial_vertex = {0, 0};
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}
		
		void initialize()
		{
			decoupled = !(param.mu > 0. || param.mu < 0.);
			n_vertex_size = decoupled ? 2 : 4;
			n_matrix_size = decoupled ? l.n_sites() : 2*l.n_sites();

			delta.resize(n_species, dmatrix_t(n_vertex_size, n_vertex_size));
			delta_W_r.resize(n_species, dmatrix_t(n_vertex_size,
				n_matrix_size / 2));
			delta_W_r_W.resize(n_species, dmatrix_t(n_vertex_size,
				n_matrix_size / 2));
			M.resize(n_species, dmatrix_t(n_vertex_size, n_vertex_size));
			create_checkerboard();
			id = dmatrix_t::Identity(n_matrix_size, n_matrix_size);
			id_2 = dmatrix_t::Identity(n_vertex_size, n_vertex_size);
			for (int i = 0; i < n_species; ++i)
			{
				equal_time_gf[i] = 0.5 * id;
				time_displaced_gf[i] = 0.5 * id;
			}
			build_vertex_matrices();
			
			dmatrix_t H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (auto& a : l.bonds("nearest neighbors"))
			{
				H0(a.first, a.second) = {0., l.parity(a.first) * param.t
					* param.dtau};
				if (!decoupled)
					H0(a.first+l.n_sites(), a.second+l.n_sites())
						= {0., l.parity(a.first) * param.t * param.dtau};
			}
			if (!decoupled)
				for (int i = 0; i < l.n_sites(); ++i)
				{
					H0(i, i) = param.mu * param.dtau;
					H0(i+l.n_sites(), i+l.n_sites()) = param.mu * param.dtau;
					H0(i, i+l.n_sites()) = {0., param.mu * param.dtau};
					H0(i+l.n_sites(), i) = {0., -param.mu * param.dtau};
				}
				
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H0);
			expH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			invExpH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < expH0.rows(); ++i)
			{
				expH0(i, i) = std::exp(-solver.eigenvalues()[i] / 2.);
				invExpH0(i, i) = std::exp(solver.eigenvalues()[i] / 2.);
			}
			expH0 = solver.eigenvectors() * expH0 * solver.eigenvectors()
				.inverse();
			invExpH0 = solver.eigenvectors() * invExpH0 * solver.eigenvectors()
				.inverse();
			
			if (param.use_projector)
			{
				dmatrix_t broken_H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
				for (auto& a : l.bonds("nearest neighbors"))
				{
					if (decoupled)
					{
						if (static_cast<int>(std::sqrt(l.n_sites()/2)) % 3 == 0
							&& get_bond_type(a) == 0)
						{
							broken_H0(a.first, a.second) = {0., l.parity(a.first)
								* 1.001 * param.dtau / 4.};
						}
						else
							broken_H0(a.first, a.second) = {0., l.parity(a.first)
								* param.t * param.dtau / 4.};
					}
					else
					{
						broken_H0(a.first, a.second) = {0., l.parity(a.first)
							* (0.9+rng()*0.2) * param.dtau / 4.};
						broken_H0(a.first+l.n_sites(), a.second+l.n_sites()) = 
							{0., l.parity(a.first) * (0.9+rng()*0.2) * param.dtau / 4.};
					}
				}
				if (!decoupled)
					for (int i = 0; i < l.n_sites(); ++i)
					{
						broken_H0(i, i) = param.mu * param.dtau;
						broken_H0(i+l.n_sites(), i+l.n_sites()) = param.mu * param.dtau;
						broken_H0(i, i+l.n_sites()) = {0., param.mu * param.dtau};
						broken_H0(i+l.n_sites(), i) = {0., -param.mu * param.dtau};
					}

				solver.compute(broken_H0);
				std::vector<int> indices(n_matrix_size);
				for (int i = 0; i < n_matrix_size; ++i)
					indices[i] = i;
				SortIndicesInc<Eigen::VectorXd, int> inc(solver.eigenvalues());
				std::sort(indices.begin(), indices.end(), inc);
				P = dmatrix_t::Zero(n_matrix_size, n_matrix_size / 2);
				for (int i = 0; i < n_matrix_size / 2; ++i)
					P.col(i) = solver.eigenvectors().col(indices[i]);
				Pt = P.adjoint();
			}
			stabilizer.set_method(param.use_projector);
		}
		
		void build_decoupled_vertex(int cnt, double parity, double spin)
		{
			double x = parity * (param.t * param.dtau + param.lambda * spin);
			double xp = parity * (param.t * param.dtau - param.lambda * spin);
			complex_t c = {std::cosh(x), 0};
			complex_t s = {0, std::sinh(x)};
			complex_t cp = {std::cosh(xp), 0};
			complex_t sp = {0, std::sinh(xp)};
			vertex_matrices[cnt] << c, s, -s, c;
			inv_vertex_matrices[cnt] << c, -s, s, c;
			delta_matrices[cnt] << cp*c + sp*s - 1., -cp*s + sp*c, -sp*c
				+ cp*s, sp*s + cp*c-1.;
		}
		
		void build_coupled_vertex(int cnt, double parity, double spin)
		{
			double x = parity * (param.t * param.dtau + param.lambda * spin);
			complex_t cm = {std::cosh(param.mu/3.*param.dtau), 0};
			complex_t cx = {std::cosh(x), 0};
			complex_t sm = {std::sinh(param.mu/3.*param.dtau), 0};
			complex_t sx = {std::sinh(x), 0};
			complex_t im = {0, 1.};
			vertex_matrices[cnt] << cm*cx, im*cm*sx, im*sm*cx, -sm*sx,
				-im*cm*sx, cm*cx, sm*sx, im*sm*cx,
				-im*sm*cx, sm*sx, cm*cx, im*cm*sx,
				-sm*sx, -im*sm*cx, -im*cm*sx, cm*cx;
			vertex_matrices[cnt] *= std::exp(param.mu/3.*param.dtau);
			inv_vertex_matrices[cnt] = vertex_matrices[cnt].inverse();
		}
		
		void build_vertex_matrices()
		{
			vertex_matrices.resize(4, dmatrix_t(n_vertex_size, n_vertex_size));
			inv_vertex_matrices.resize(4, dmatrix_t(n_vertex_size, n_vertex_size));
			delta_matrices.resize(4, dmatrix_t(n_vertex_size, n_vertex_size));
			int cnt = 0;
			for (double parity : {1., -1.})
				for (double spin : {1., -1.})
				{
					if (decoupled)
						build_decoupled_vertex(cnt, parity, spin);
					else
						build_coupled_vertex(cnt, parity, spin);
					++cnt;
				}
			if (!decoupled)
			{
				delta_matrices[0] = vertex_matrices[1] * inv_vertex_matrices[0] - id_2;
				delta_matrices[1] = vertex_matrices[0] * inv_vertex_matrices[1] - id_2;
				delta_matrices[2] = vertex_matrices[3] * inv_vertex_matrices[2] - id_2;
				delta_matrices[3] = vertex_matrices[2] * inv_vertex_matrices[3] - id_2;
			}
		}
		
		dmatrix_t& get_vertex_matrix(int species, int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return vertex_matrices[species*n_species + i%2*2*n_species
				+ static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_inv_vertex_matrix(int species, int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return inv_vertex_matrices[species*n_species + i%2*2*n_species
				+ static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_delta_matrix(int species, int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			return delta_matrices[species*n_species + i%2*2*n_species
				+ static_cast<int>(s<0)];
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

		double action(int species, double x, int i, int j) const
		{
			return l.parity(i) * (param.t * param.dtau + param.lambda * x);
			
			//return l.parity(i) * param.lambda * x;
			
			//double a = (get_bond_type({i, j}) < cb_bonds.size() - 1) ? 0.5 : 1.0;
			//return l.parity(i) * a * param.lambda * x;
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
		
		int bond_index(int i, int j) const
		{
			return bond_indices.at({std::min(i, j), std::max(i, j)});
		}

		const std::map<int, int>& get_cb_bonds(int i) const
		{
			return cb_bonds[i];
		}

		void flip_spin(const std::pair<int, int>& b)
		{
			aux_spins[tau[0]-1].flip(bond_index(b.first, b.second));
		}

		void buffer_equal_time_gf()
		{
			for (int i = 0; i < n_species; ++i)
			{
				if (param.use_projector)
				{
					//W_l_buffer[i] = proj_W_l[i];
					//W_r_buffer[i] = proj_W_r[i];
					//W_buffer[i] = proj_W[i];
					
					gf_buffer[i] = equal_time_gf[i];
				}
				else
					gf_buffer[i] = equal_time_gf[i];
				gf_buffer_partial_vertex[i] = partial_vertex[i];
				gf_buffer_tau[i] = tau[i];
			}
		}

		void reset_equal_time_gf_to_buffer()
		{
			for (int i = 0; i < n_species; ++i)
			{
				if (param.use_projector)
				{
					//proj_W_l[i] = W_l_buffer[i];
					//proj_W_r[i] = W_r_buffer[i];
					//proj_W[i] = W_buffer[i];
					
					equal_time_gf[i] = gf_buffer[i];
				}
				else
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
			stabilizer.resize(n_intervals, n_matrix_size);
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

		void multiply_vertex_from_left(int species, dmatrix_t& m,
			int bond_type, const arg_t& vertex, int inv)
		{
			dmatrix_t old_m = m;
			m.setZero();
			for (int i = 0; i < l.n_sites(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				if (i > j) continue;
				double sigma = vertex.get(bond_index(i, j));
				dmatrix_t* vm;
				if(inv == 1)
					vm = &get_vertex_matrix(species, i, j, sigma);
				else
					vm = &get_inv_vertex_matrix(species, i, j, sigma);
				if (decoupled)
				{
					m.row(i) = old_m.row(i) * (*vm)(0, 0) + old_m.row(j) * (*vm)(0, 1);
					m.row(j) = old_m.row(i) * (*vm)(1, 0) + old_m.row(j) * (*vm)(1, 1);
				}
				else
				{
					int ns = l.n_sites();
					m.row(i) = old_m.row(i) * (*vm)(0, 0) + old_m.row(j) * (*vm)(0, 1)
						+ old_m.row(i+ns) * (*vm)(0, 2) + old_m.row(j+ns) * (*vm)(0, 3);
					m.row(j) = old_m.row(i) * (*vm)(1, 0) + old_m.row(j) * (*vm)(1, 1)
						+ old_m.row(i+ns) * (*vm)(1, 2) + old_m.row(j+ns) * (*vm)(1, 3);
					m.row(i+ns) = old_m.row(i) * (*vm)(2, 0) + old_m.row(j) * (*vm)(2, 1)
						+ old_m.row(i+ns) * (*vm)(2, 2) + old_m.row(j+ns) * (*vm)(2, 3);
					m.row(j+ns) = old_m.row(i) * (*vm)(3, 0) + old_m.row(j) * (*vm)(3, 1)
						+ old_m.row(i+ns) * (*vm)(3, 2) + old_m.row(j+ns) * (*vm)(3, 3);
				}
			}
		}

		void multiply_vertex_from_right(int species, dmatrix_t& m,
			int bond_type, const arg_t& vertex, int inv)
		{
			dmatrix_t old_m = m;
			m.setZero();
			for (int i = 0; i < l.n_sites(); ++i)
			{
				int j = cb_bonds[bond_type][i];
				if (i > j) continue;
				double sigma = vertex.get(bond_index(i, j));
				dmatrix_t* vm;
				if(inv == 1)
					vm = &get_vertex_matrix(species, i, j, sigma);
				else
					vm = &get_inv_vertex_matrix(species, i, j, sigma);
				if (decoupled)
				{
					m.col(i) = old_m.col(i) * (*vm)(0, 0) + old_m.col(j) * (*vm)(1, 0);
					m.col(j) = old_m.col(i) * (*vm)(0, 1) + old_m.col(j) * (*vm)(1, 1);
				}
				else
				{
					int ns = l.n_sites();
					m.col(i) = old_m.col(i) * (*vm)(0, 0) + old_m.col(j) * (*vm)(1, 0)
						+ old_m.col(i+ns) * (*vm)(2, 0) + old_m.col(j+ns) * (*vm)(3, 0);
					m.col(j) = old_m.col(i) * (*vm)(0, 1) + old_m.col(j) * (*vm)(1, 1)
						+ old_m.col(i+ns) * (*vm)(2, 1) + old_m.col(j+ns) * (*vm)(3, 1);
					m.col(i+ns) = old_m.col(i) * (*vm)(0, 2) + old_m.col(j) * (*vm)(1, 2)
						+ old_m.col(i+ns) * (*vm)(2, 2) + old_m.col(j+ns) * (*vm)(3, 2);
					m.col(j+ns) = old_m.col(i) * (*vm)(0, 3) + old_m.col(j) * (*vm)(1, 3)
						+ old_m.col(i+ns) * (*vm)(2, 3) + old_m.col(j+ns) * (*vm)(3, 3);
				}
			}
		}

		void prepare_flip(int species)
		{
			if (param.use_projector)
			{
				//proj_W_l[species] = proj_W_l[species] * expH0;
				//proj_W_r[species] = invExpH0 * proj_W_r[species];
				
				equal_time_gf[species] = invExpH0 * equal_time_gf[species] * expH0;
			}
			else
				equal_time_gf[species] = invExpH0 * equal_time_gf[species] * expH0;
		}

		void prepare_measurement(int species)
		{
			if (param.use_projector)
			{
				//proj_W_l[species] = proj_W_l[species] * invExpH0;
				//proj_W_r[species] = expH0 * proj_W_r[species];
				
				equal_time_gf[species] = expH0 * equal_time_gf[species] * invExpH0;
			}
			else
				equal_time_gf[species] = expH0 * equal_time_gf[species] * invExpH0;
		}

		dmatrix_t propagator(int species, int tau_n, int tau_m)
		{
			dmatrix_t b = id;
			for (int n = tau_n; n > tau_m; --n)
			{
				auto& vertex = aux_spins[n-1];
				//b *= expH0;
				
				multiply_vertex_from_right(species, b, 0, vertex, 1);
				multiply_vertex_from_right(species, b, 1, vertex, 1);
				multiply_vertex_from_right(species, b, 2, vertex, 1);
				
				//multiply_vertex_from_right(species, b, 1, vertex, 1);
				//multiply_vertex_from_right(species, b, 0, vertex, 1);
				//b *= expH0;
			}
			return b;
		}
		
		void multiply_propagator_from_left(int i, dmatrix_t& m, const arg_t& vertex, int inv)
		{
			if (inv == 1)
			{
				//m = expH0 * m;
				//multiply_vertex_from_left(i, m, 0, vertex, 1);
				//multiply_vertex_from_left(i, m, 1, vertex, 1);
				
				multiply_vertex_from_left(i, m, 2, vertex, 1);
				multiply_vertex_from_left(i, m, 1, vertex, 1);
				multiply_vertex_from_left(i, m, 0, vertex, 1);
				
				//m = expH0 * m;
			}
			else if (inv == -1)
			{
				//m = invExpH0 * m;
				
				multiply_vertex_from_left(i, m, 0, vertex, -1);
				multiply_vertex_from_left(i, m, 1, vertex, -1);
				multiply_vertex_from_left(i, m, 2, vertex, -1);
				
				//multiply_vertex_from_left(i, m, 1, vertex, -1);
				//multiply_vertex_from_left(i, m, 0, vertex, -1);
				//m = invExpH0 * m;
			}
		}
		
		void multiply_propagator_from_right(int i, dmatrix_t& m, const arg_t& vertex, int inv)
		{
			if (inv == 1)
			{
				//m = m * expH0;
				
				multiply_vertex_from_right(i, m, 0, vertex, 1);
				multiply_vertex_from_right(i, m, 1, vertex, 1);
				multiply_vertex_from_right(i, m, 2, vertex, 1);
				
				//multiply_vertex_from_right(i, m, 1, vertex, 1);
				//multiply_vertex_from_right(i, m, 0, vertex, 1);
				//m = m * expH0;
			}
			else if (inv == -1)
			{
				//m = m * invExpH0;
				//multiply_vertex_from_right(i, m, 0, vertex, -1);
				//multiply_vertex_from_right(i, m, 1, vertex, -1);
				
				multiply_vertex_from_right(i, m, 2, vertex, -1);
				multiply_vertex_from_right(i, m, 1, vertex, -1);
				multiply_vertex_from_right(i, m, 0, vertex, -1);
				
				//m = m * invExpH0;
			}
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
					//multiply_vertex_from_left(species, proj_W_r[species],
					//	bond_type, vertex, -1);
					//multiply_vertex_from_right(species, proj_W_l[species],
					//	bond_type, vertex, 1);
					
					multiply_vertex_from_left(species, equal_time_gf[species],
						bond_type, vertex, -1);
					multiply_vertex_from_right(species, equal_time_gf[species],
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
					//multiply_vertex_from_left(species, proj_W_r[species],
					//	bond_type, vertex, 1);
					//multiply_vertex_from_right(species, proj_W_l[species],
					//	bond_type, vertex, -1);
					
					multiply_vertex_from_left(species, equal_time_gf[species],
						bond_type, vertex, 1);
					multiply_vertex_from_right(species, equal_time_gf[species],
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
				if (param.use_projector)
				{
					//multiply_propagator_from_left(i, proj_W_r[i], vertex, 1);
					//multiply_propagator_from_right(i, proj_W_l[i], vertex, -1);
					
					multiply_propagator_from_left(i, equal_time_gf[i], vertex, 1);
					multiply_propagator_from_right(i, equal_time_gf[i], vertex, -1);
				}
				else
				{
					if (update_time_displaced_gf)
						multiply_propagator_from_left(i, time_displaced_gf[i], vertex, 1);
					multiply_propagator_from_left(i, equal_time_gf[i], vertex, 1);
					multiply_propagator_from_right(i, equal_time_gf[i], vertex, -1);
				}
				++tau[i];
			}
		}

		void advance_backward()
		{
			for (int i = 0; i < n_species; ++i)
			{
				auto& vertex = aux_spins[tau[i] - 1];
				if (param.use_projector)
				{
					//multiply_propagator_from_left(i, proj_W_r[i], vertex, -1);
					//multiply_propagator_from_right(i, proj_W_l[i], vertex, 1);
					
					multiply_propagator_from_left(i, equal_time_gf[i], vertex, -1);
					multiply_propagator_from_right(i, equal_time_gf[i], vertex, 1);
				}
				else
				{
					if (update_time_displaced_gf)
						multiply_propagator_from_right(i, time_displaced_gf[i], vertex, 1);
					multiply_propagator_from_left(i, equal_time_gf[i], vertex, -1);
					multiply_propagator_from_right(i, equal_time_gf[i], vertex, 1);
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

		complex_t try_ising_flip(int species, int i, int j)
		{
			auto& vertex = aux_spins[tau[species]-1];
			double sigma = vertex.get(bond_index(i, j));
			int m = std::min(i, j), n = std::max(i, j);
			last_flip = {m, n};
			for (int a = 0; a < n_species; ++a)
				delta[a] = get_delta_matrix(a, m, n, sigma);
	
			if (param.use_projector)
			{
				/*
				dmatrix_t b_l(P.cols(), 2);
				b_l.col(0) = proj_W_l[species].col(m);
				b_l.col(1) = proj_W_l[species].col(n);

				delta_W_r[species].row(0) = delta[species](0, 0) * proj_W_r[species].row(m)
					+ delta[species](0, 1) * proj_W_r[species].row(n);
				delta_W_r[species].row(1) = delta[species](1, 0) * proj_W_r[species].row(m)
					+ delta[species](1, 1) * proj_W_r[species].row(n);
					
				delta_W_r_W[species].setZero();
				for (int i = 0; i < P.cols(); ++i)
				{
					delta_W_r_W[species].row(0) += delta_W_r[species](0, i) * proj_W[species].row(i);
					delta_W_r_W[species].row(1) += delta_W_r[species](1, i) * proj_W[species].row(i);
				}

				M[species] = id_2; M[species].noalias() += delta_W_r_W[species] * b_l;
				*/
				
				dmatrix_t& gf = equal_time_gf[species];
				dmatrix_t g(n_vertex_size, n_vertex_size);
				if (decoupled)
				{
					g << 1.-gf(m, m), -gf(m, n), -gf(n, m), 1.-gf(n, n);
					M[species] = id_2; M[species].noalias() += g * delta[species];
					return M[species].determinant();
				}
				else
				{
					int ns = l.n_sites();
					g << 1.-gf(m, m), -gf(m, n), -gf(m, m+ns), -gf(m, n+ns),
					-gf(n, m), 1.-gf(n, n), -gf(n, m+ns), -gf(n, n+ns),
					-gf(m+ns, m), -gf(m+ns, n), 1.-gf(m+ns, m+ns), -gf(m+ns, n+ns),
					-gf(n+ns, m), -gf(n+ns, n), -gf(n+ns, m+ns), 1.-gf(n+ns, n+ns);
					M[species] = id_2; M[species].noalias() += g * delta[species];
					return std::sqrt(M[species].determinant());
				}
			}
			else
			{
				dmatrix_t& gf = equal_time_gf[species];
				dmatrix_t g(n_vertex_size, n_vertex_size);
				if (decoupled)
				{
					g << 1.-gf(m, m), -gf(m, n), -gf(n, m), 1.-gf(n, n);
					M[species] = id_2; M[species].noalias() += g * delta[species];
					return M[species].determinant();
				}
				else
				{
					int ns = l.n_sites();
					g << 1.-gf(m, m), -gf(m, n), -gf(m, m+ns), -gf(m, n+ns),
					-gf(n, m), 1.-gf(n, n), -gf(n, m+ns), -gf(n, n+ns),
					-gf(m+ns, m), -gf(m+ns, n), 1.-gf(m+ns, m+ns), -gf(m+ns, n+ns),
					-gf(n+ns, m), -gf(n+ns, n), -gf(n+ns, m+ns), 1.-gf(n+ns, n+ns);
					M[species] = id_2; M[species].noalias() += g * delta[species];
					return std::sqrt(M[species].determinant());
				}
			}
		}

		void update_equal_time_gf_after_flip(int species)
		{
			int indices[2] = {last_flip.first, last_flip.second};

			if (param.use_projector)
			{
				/*
				M[species] = M[species].inverse().eval();
				
				dmatrix_t W_l_M(Pt.rows(), 2);
				W_l_M.col(0) = proj_W_l[species].col(indices[0]) * M[species](0, 0)
					+ proj_W_l[species].col(indices[1]) * M[species](1, 0);
				W_l_M.col(1) = proj_W_l[species].col(indices[0]) * M[species](0, 1)
					+ proj_W_l[species].col(indices[1]) * M[species](1, 1);
				
				proj_W_r[species].row(indices[0]) += delta_W_r[species].row(0);
				proj_W_r[species].row(indices[1]) += delta_W_r[species].row(1);
				proj_W[species] -= proj_W[species] * (W_l_M * delta_W_r_W[species]);
				*/
				
				M[species] = M[species].inverse().eval();
				dmatrix_t& gf = equal_time_gf[species];
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				if (!decoupled)
				{
					g_cols.col(2) = gf.col(indices[0]+l.n_sites());
					g_cols.col(3) = gf.col(indices[1]+l.n_sites());
				}
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				if (!decoupled)
				{
					g_rows.row(2) = gf.row(indices[0]+l.n_sites());
					g_rows.row(3) = gf.row(indices[1]+l.n_sites());
					g_rows(2, indices[0]+l.n_sites()) -= 1.;
					g_rows(3, indices[1]+l.n_sites()) -= 1.;
				}
				gf.noalias() += (g_cols * delta[species]) * (M[species] * g_rows);
				
			}
			else
			{
				M[species] = M[species].inverse().eval();
				dmatrix_t& gf = equal_time_gf[species];
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				if (!decoupled)
				{
					g_cols.col(2) = gf.col(indices[0]+l.n_sites());
					g_cols.col(3) = gf.col(indices[1]+l.n_sites());
				}
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				if (!decoupled)
				{
					g_rows.row(2) = gf.row(indices[0]+l.n_sites());
					g_rows.row(3) = gf.row(indices[1]+l.n_sites());
					g_rows(2, indices[0]+l.n_sites()) -= 1.;
					g_rows(3, indices[1]+l.n_sites()) -= 1.;
				}
				gf.noalias() += (g_cols * delta[species]) * (M[species] * g_rows);
			}
		}

		void static_measure(std::vector<double>& c, complex_t& n, double& m2, double& epsilon)
		{
			//if (param.use_projector)
			//	equal_time_gf[0] = id - proj_W_r[0] * proj_W[0] * proj_W_l[0];
			complex_t im = {0., 1.};
			for (int i = 0; i < l.n_sites(); ++i)
			{
				n += equal_time_gf[0](i, i) / complex_t(l.n_sites());
				if (!decoupled)
				{
					n += (equal_time_gf[0](i+l.n_sites(), i+l.n_sites())
						- im*equal_time_gf[0](i, i+l.n_sites()) + im*equal_time_gf[0](i+l.n_sites(), i))
						/ complex_t(l.n_sites());
				}
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf[0](i, j)
							* equal_time_gf[0](i, j));
						//Correlation function
						c[l.distance(i, j)] += re / l.n_sites();
						//M2 structure factor
						m2 += l.parity(i) * l.parity(j) * re
							/ std::pow(l.n_sites(), 2);
					}
			}
			if (!decoupled)
				n /= 2.;
			for (auto& i : l.bonds("nearest neighbors"))
				epsilon += l.parity(i.first) * std::imag(equal_time_gf[0](i.first, i.second))
					/ l.n_bonds();
		}

		void measure_dynamical_observable(std::vector<std::vector<double>>&
			dyn_tau, const std::vector<wick_base<dmatrix_t>>& obs)
		{
			if (param.use_projector)
			{
				buffer_equal_time_gf();
				stabilizer.set_buffer();
				std::vector<dmatrix_t> et_gf_L(param.n_discrete_tau);
				std::vector<dmatrix_t> et_gf_R(param.n_discrete_tau);
				std::vector<dmatrix_t> et_gf_T(param.n_discrete_tau);
				time_displaced_gf[0] = id;
				
				for (int n = 0; n < param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_backward();
						stabilize_backward();
					}
					//equal_time_gf[0] = id - proj_W_r[0] * proj_W[0] * proj_W_l[0];
					et_gf_L[n] = equal_time_gf[0];
				}
				dmatrix_t et_gf_0 = equal_time_gf[0];
				for (int n = 0; n < 2*param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_backward();
						stabilize_backward();
					}
					//equal_time_gf[0] = id - proj_W_r[0] * proj_W[0] * proj_W_l[0];
					if (n < param.n_discrete_tau)
						et_gf_R[n] = equal_time_gf[0];
					if (n % 2 == 1)
						et_gf_T[n/2] = equal_time_gf[0];
				}
				
				for (int i = 0; i < dyn_tau.size(); ++i)
					dyn_tau[i][0] = obs[i].get_obs(et_gf_0, et_gf_0, et_gf_0);
				for (int n = 1; n <= param.n_discrete_tau; ++n)
				{
					dmatrix_t g_l = propagator(0, max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau) * et_gf_L[et_gf_L.size() - n];
					int n_r = max_tau/2 - n;
					dmatrix_t g_r = propagator(0, max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau) * et_gf_R[n-1];
					time_displaced_gf[0] = g_l * time_displaced_gf[0] * g_r;
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][n] = obs[i].get_obs(et_gf_0, et_gf_T[n-1],
							time_displaced_gf[0]);
				}
				
				reset_equal_time_gf_to_buffer();
				stabilizer.restore_buffer();
			}
			else
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
		}
	private:
		void create_checkerboard()
		{
			int cnt = 0;
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = i+1; j < l.n_sites(); ++j)
					if (l.distance(i, j) == 1)
					{
						bond_indices[{i, j}] = cnt;
						++cnt;
					}
				
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
		Random& rng;
		const lattice& l;
		const parameters& param;
		measurements& measure;
		int n_intervals;
		std::vector<int> tau;
		std::vector<int> partial_vertex;
		int max_tau;
		std::vector<arg_t> aux_spins;
		std::map<std::pair<int, int>, int> bond_indices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		bool update_time_displaced_gf;
		bool decoupled;
		int n_species;
		int n_vertex_size;
		int n_matrix_size;
		std::vector<dmatrix_t> vertex_matrices;
		std::vector<dmatrix_t> inv_vertex_matrices;
		std::vector<dmatrix_t> delta_matrices;
		std::vector<dmatrix_t> equal_time_gf;
		std::vector<dmatrix_t> time_displaced_gf;
		std::vector<dmatrix_t> proj_W_l;
		std::vector<dmatrix_t> proj_W_r;
		std::vector<dmatrix_t> proj_W;
		std::vector<dmatrix_t> gf_buffer;
		std::vector<dmatrix_t> W_l_buffer;
		std::vector<dmatrix_t> W_r_buffer;
		std::vector<dmatrix_t> W_buffer;
		std::vector<int> gf_buffer_partial_vertex;
		std::vector<int> gf_buffer_tau;
		dmatrix_t id;
		dmatrix_t id_2;
		dmatrix_t expH0;
		dmatrix_t invExpH0;
		dmatrix_t P;
		dmatrix_t Pt;
		std::vector<dmatrix_t> delta;
		std::vector<dmatrix_t> delta_W_r;
		std::vector<dmatrix_t> delta_W_r_W;
		std::vector<dmatrix_t> M;
		std::pair<int, int> last_flip;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
