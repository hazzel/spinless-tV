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
#include "wick_static_base.h"

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
		using stabilizer_t = qr_stabilizer;

		fast_update(Random& rng_, const lattice& l_, const parameters& param_,
			measurements& measure_)
			: rng(rng_), l(l_), param(param_), measure(measure_), tau(1),
				update_time_displaced_gf(false),
				stabilizer{measure, equal_time_gf, time_displaced_gf,
				proj_W_l, proj_W_r, proj_W}
		{}

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
			tau = max_tau;
			partial_vertex = 0;
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}
		
		void initialize()
		{
			if (param.decoupling == "majorana")
			{
				if (param.mu != 0. || param.stag_mu != 0.)
					decoupled = false;
				else
					decoupled = true;
			}
			else
				decoupled = true;

			n_vertex_size = decoupled ? 2 : 4;
			n_matrix_size = decoupled ? l.n_sites() : 2*l.n_sites();
			
			if (param.geometry == "hex")
				cb_bonds.resize(2);
			else
				cb_bonds.resize(3);

			delta = dmatrix_t(n_vertex_size, n_vertex_size);
			delta_W_r = dmatrix_t(n_vertex_size, n_matrix_size / 2);
			W_W_l = dmatrix_t(n_matrix_size / 2, n_vertex_size);
			M = dmatrix_t(n_vertex_size, n_vertex_size);
			create_checkerboard();
			id = dmatrix_t::Identity(n_matrix_size, n_matrix_size);
			id_2 = dmatrix_t::Identity(n_vertex_size, n_vertex_size);
			equal_time_gf = 0.5 * id;
			time_displaced_gf = 0.5 * id;
			build_vertex_matrices();
			
			dmatrix_t H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			if (param.decoupling == "majorana")
				build_majorana_H0(H0);
			else
				build_dirac_H0(H0);
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H0);
			expH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			invExpH0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < expH0.rows(); ++i)
			{
				expH0(i, i) = std::exp(solver.eigenvalues()[i] / 2.);
				invExpH0(i, i) = std::exp(-solver.eigenvalues()[i] / 2.);
			}
			expH0 = solver.eigenvectors() * expH0 * solver.eigenvectors()
				.inverse();
			invExpH0 = solver.eigenvectors() * invExpH0 * solver.eigenvectors()
				.inverse();
			
			if (param.use_projector)
			{
				dmatrix_t broken_H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
				if (param.decoupling == "majorana")
					build_broken_majorana_H0(broken_H0);
				else
					build_broken_dirac_H0(broken_H0);
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
				stabilizer.set_P(P, Pt);
				//std::cout << solver.eigenvalues() << std::endl;
			}
			stabilizer.set_method(param.use_projector);
		}
		
		void build_majorana_H0(dmatrix_t& H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
			{
				H0(a.first, a.second) = {0., l.parity(a.first) * param.t
					* param.dtau};
				if (!decoupled)
					H0(a.first+l.n_sites(), a.second+l.n_sites())
						= {0., l.parity(a.first) * param.t * param.dtau};
			}
			for (auto& a : l.bonds("d3_bonds"))
			{
				H0(a.first, a.second) = {0., l.parity(a.first) * param.tprime
					* param.dtau};
				if (!decoupled)
					H0(a.first+l.n_sites(), a.second+l.n_sites())
						= {0., l.parity(a.first) * param.tprime * param.dtau};
			}
			if (!decoupled)
				for (int i = 0; i < l.n_sites(); ++i)
				{
					double m = param.mu+l.parity(i)*param.stag_mu;
					H0(i, i) = m * param.dtau;
					H0(i+l.n_sites(), i+l.n_sites()) = m * param.dtau;
					H0(i, i+l.n_sites()) = {0., m * param.dtau};
					H0(i+l.n_sites(), i) = {0., -m * param.dtau};
				}
		}
		
		void build_broken_majorana_H0(dmatrix_t& broken_H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
			{
				if (a.first > a.second)
					continue;
				
				double tp;
				if (param.L % 3 == 0 && get_bond_type(a) == 0)
				//auto& kek_bonds = l.bonds("kekule");
				//if (param.L % 3 == 0 && std::find(kek_bonds.begin(), kek_bonds.end(), a) != kek_bonds.end())
				{
					tp = param.t * 1.000001;
					//tp = param.t * (0.9999+rng()*0.0002);
				}
				else
				{
					tp = param.t;
				}
				broken_H0(a.first, a.second) = {0., l.parity(a.first)
					* tp / 4.};
				broken_H0(a.second, a.first) = {0., l.parity(a.second)
					* tp / 4.};
					
				if (!decoupled)
				{
					broken_H0(a.first+l.n_sites(), a.second+l.n_sites()) = 
						{0., l.parity(a.first) * tp / 4.};
					broken_H0(a.second+l.n_sites(), a.first+l.n_sites()) = 
						{0., l.parity(a.second) * tp / 4.};
				}
			}
			for (auto& a : l.bonds("d3_bonds"))
			{
				broken_H0(a.first, a.second) = {0., l.parity(a.first)
					* param.tprime / 4.};
				if (!decoupled)
					broken_H0(a.first+l.n_sites(), a.second+l.n_sites()) = 
						{0., l.parity(a.first) * param.tprime / 4.};
			}
			
			if (!decoupled)
				for (int i = 0; i < l.n_sites(); ++i)
				{
					double m = param.mu+l.parity(i)*param.stag_mu;
					broken_H0(i, i) = m;
					broken_H0(i+l.n_sites(), i+l.n_sites()) = m;
					broken_H0(i, i+l.n_sites()) = {0., m};
					broken_H0(i+l.n_sites(), i) = {0., -m};
				}
			
			/*
			for (auto& a : l.bonds("kekule"))
			{
				double tp = param.t * 1.00000001;
				broken_H0(a.first, a.second) = {0., l.parity(a.first)
					* tp / 4.};
			}
			for (auto& a : l.bonds("kekule_2"))
			{
				double tp = param.t;
				broken_H0(a.first, a.second) = {0., l.parity(a.first)
					* tp / 4.};
			}
			for (auto& a : l.bonds("kekule_3"))
			{
				double tp = param.t;
				broken_H0(a.first, a.second) = {0., l.parity(a.first)
					* tp / 4.};
			}
			*/
			
			/*
			for (int i = 0; i < l.n_sites(); ++i)
				for (int j = i; j < l.n_sites(); ++j)
				{
					double r = rng();
					broken_H0(i, j) = {0., l.parity(i) * r};
					broken_H0(j, i) = {0., l.parity(j) * r};
				}
			*/
		}		
		
		void build_dirac_H0(dmatrix_t& H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
				H0(a.first, a.second) = {param.t * param.dtau, 0.};
			for (auto& a : l.bonds("d3_bonds"))
				H0(a.first, a.second) = {param.tprime * param.dtau, 0.};
			for (int i = 0; i < l.n_sites(); ++i)
				H0(i, i) = l.parity(i) * param.stag_mu;
		}
		
		void build_broken_dirac_H0(dmatrix_t& broken_H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
			{
				if (a.first > a.second)
					continue;
				
				double tp;
				if (param.L % 3 == 0 && get_bond_type(a) == 0)
				//auto& kek_bonds = l.bonds("kekule");
				//if (param.L % 3 == 0 && std::find(kek_bonds.begin(), kek_bonds.end(), a) != kek_bonds.end())
				{
					tp = param.t * 1.000001;
					//tp = param.t * (0.9999+rng()*0.0002);
				}
				else
				{
					tp = param.t;
				}
				broken_H0(a.first, a.second) = {tp, 0.};
				broken_H0(a.second, a.first) = {tp, 0.};
			}
			for (auto& a : l.bonds("d3_bonds"))
				broken_H0(a.first, a.second) = {param.tprime, 0.};
			//for (int i = 0; i < l.n_sites(); ++i)
			//	broken_H0(i, i) = l.parity(i) * param.stag_mu;
		}
		
		void build_decoupled_majorana_vertex(int cnt, double parity, double spin, bool symmetry_broken)
		{
			double x, xp;
			if (param.tprime > 0. || param.tprime < 0.)
			{
				// e^{-H dtau} = e^{- K/2 dtau} e^{- V dtau} e^{- K/2 dtau}
				x = parity * param.lambda * spin;
				xp = - parity * param.lambda * spin;
			}
			else
			{
				// e^{-H dtau} = e^{- (K+V) dtau}
				double tp;
				if (symmetry_broken)
					tp = param.t * 1.000001;
					//tp = param.t * (0.9999+rng()*0.0002);
				else
					tp = param.t;
				x = parity * (tp * param.dtau + param.lambda * spin);
				xp = parity * (tp * param.dtau - param.lambda * spin);
			}
			complex_t c = {std::cosh(x), 0};
			complex_t s = {0, std::sinh(x)};
			complex_t cp = {std::cosh(xp), 0};
			complex_t sp = {0, std::sinh(xp)};
			vertex_matrices[cnt] << c, s, -s, c;
			inv_vertex_matrices[cnt] << c, -s, s, c;
			delta_matrices[cnt] << cp*c + sp*s - 1., -cp*s + sp*c, -sp*c
				+ cp*s, sp*s + cp*c - 1.;
		}
		
		void build_coupled_majorana_vertex(int cnt, double parity, double spin, bool symmetry_broken)
		{
			double x;
			if (param.tprime > 0. || param.tprime < 0.)
			{
				// e^{-H dtau} = e^{- K/2 dtau} e^{- V dtau} e^{- K/2 dtau}
				x = parity * param.lambda * spin;
			}
			else
			{
				// e^{-H dtau} = e^{- (K+V) dtau}
				x = parity * (param.t * param.dtau + param.lambda * spin);
			}
			double m = param.mu + parity*param.stag_mu;
			double e = std::exp(m/3.*param.dtau);
			complex_t cm = {e*std::cosh(m/3.*param.dtau), 0};
			complex_t cx = {std::cosh(x), 0};
			complex_t sm = {e*std::sinh(m/3.*param.dtau), 0};
			complex_t sx = {std::sinh(x), 0};
			complex_t im = {0, 1.};
			vertex_matrices[cnt] << cm*cx, im*cm*sx, im*sm*cx, -sm*sx,
				-im*cm*sx, cm*cx, sm*sx, im*sm*cx,
				-im*sm*cx, sm*sx, cm*cx, im*cm*sx,
				-sm*sx, -im*sm*cx, -im*cm*sx, cm*cx;
			inv_vertex_matrices[cnt] = vertex_matrices[cnt].inverse();
		}
		
		void build_dirac_vertex(int cnt, double parity, double spin, bool symmetry_broken)
		{
			double x, xp;
			if (param.tprime > 0. || param.tprime < 0.)
			{
				// e^{-H dtau} = e^{- K/2 dtau} e^{- V dtau} e^{- K/2 dtau}
				x = param.lambda * spin;
				xp = - param.lambda * spin;
			}
			else
			{
				// e^{-H dtau} = e^{- (K+V) dtau}
				double tp;
				if (symmetry_broken)
					tp = param.t * 1.000000;
					//tp = param.t * (0.9999+rng()*0.0002);
				else
					tp = param.t;
				x = tp * param.dtau + param.lambda * spin;
				xp = tp * param.dtau - param.lambda * spin;
			}
			double m = parity * param.stag_mu;
			double mx = std::sqrt(m*m + x*x);
			complex_t c1 = {std::cosh(mx) + m/mx * std::sinh(mx), 0.};
			complex_t c2 = {std::cosh(mx) - m/mx * std::sinh(mx), 0.};
			complex_t s = {x/mx * std::sinh(mx), 0.};
			
			vertex_matrices[cnt] << c1, s, s, c2;
			inv_vertex_matrices[cnt] = vertex_matrices[cnt].inverse();
		}
		
		void build_vertex_matrices()
		{
			vertex_matrices.resize(8, dmatrix_t(n_vertex_size, n_vertex_size));
			inv_vertex_matrices.resize(8, dmatrix_t(n_vertex_size, n_vertex_size));
			delta_matrices.resize(8, dmatrix_t(n_vertex_size, n_vertex_size));
			int cnt = 0;
			for (bool symmetry_broken : {false, true})
				for (double parity : {1., -1.})
					for (double spin : {1., -1.})
					{
						if (param.decoupling == "majorana")
						{
							if (decoupled)
								build_decoupled_majorana_vertex(cnt, parity, spin, symmetry_broken);
							else
								build_coupled_majorana_vertex(cnt, parity, spin, symmetry_broken);
						}
						else
							build_dirac_vertex(cnt, parity, spin, symmetry_broken);
						++cnt;
					}
			if (!(param.decoupling == "majorana" && decoupled))
			{
				for (int i = 0; i < 2; ++i)
				{
					delta_matrices[4*i+0] = vertex_matrices[4*i+1]
						* inv_vertex_matrices[4*i+0] - id_2;
					delta_matrices[4*i+1] = vertex_matrices[4*i+0]
						* inv_vertex_matrices[4*i+1] - id_2;
					delta_matrices[4*i+2] = vertex_matrices[4*i+3]
						* inv_vertex_matrices[4*i+2] - id_2;
					delta_matrices[4*i+3] = vertex_matrices[4*i+2]
						* inv_vertex_matrices[4*i+3] - id_2;
				}
			}
		}
		
		dmatrix_t& get_vertex_matrix(int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			//int symmetry_broken = (get_bond_type({i, j}) == 0 ? 1 : 0);
			int symmetry_broken;
			//if (param.use_projector && param.L % 3 == 0 && get_bond_type({i, j}) == 0)
			auto& kek_bonds = l.bonds("kekule");
			if (param.use_projector && param.L % 3 == 0 && std::find(kek_bonds.begin(), kek_bonds.end(), std::make_pair(i, j)) != kek_bonds.end())
				symmetry_broken = 1;
			else
				symmetry_broken = 0;
				
			return vertex_matrices[4*symmetry_broken + i%2*2 + static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_inv_vertex_matrix(int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			//int symmetry_broken = (get_bond_type({i, j}) == 0 ? 1 : 0);
			int symmetry_broken;
			//if (param.use_projector && param.L % 3 == 0 && get_bond_type({i, j}) == 0)
			auto& kek_bonds = l.bonds("kekule");
			if (param.use_projector && param.L % 3 == 0 && std::find(kek_bonds.begin(), kek_bonds.end(), std::make_pair(i, j)) != kek_bonds.end())
				symmetry_broken = 1;
			else
				symmetry_broken = 0;
			
			return inv_vertex_matrices[4*symmetry_broken + i%2*2 + static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_delta_matrix(int i, int j, int s)
		{
			// Assume i < j and fix sublattice 0 => p=1
			//int symmetry_broken = (get_bond_type({i, j}) == 0 ? 1 : 0);
			int symmetry_broken;
			//if (param.use_projector && param.L % 3 == 0 && get_bond_type({i, j}) == 0)
			auto& kek_bonds = l.bonds("kekule");
			if (param.use_projector && param.L % 3 == 0 && std::find(kek_bonds.begin(), kek_bonds.end(), std::make_pair(i, j)) != kek_bonds.end())
				symmetry_broken = 1;
			else
				symmetry_broken = 0;
			
			return delta_matrices[4*symmetry_broken + i%2*2 + static_cast<int>(s<0)];
		}

		int get_bond_type(const std::pair<int, int>& bond) const
		{
			for (int i = 0; i < cb_bonds.size(); ++i)
				if (cb_bonds[i].at(bond.first) == bond.second)
					return i;
		}

		int get_partial_vertex() const
		{
			return partial_vertex;
		}
		
		const arg_t& vertex(int index)
		{
			return aux_spins[index-1]; 
		}

		int get_tau()
		{
			return tau;
		}

		int get_max_tau()
		{
			return max_tau;
		}
		
		int bond_index(int i, int j) const
		{
			return bond_indices.at({std::min(i, j), std::max(i, j)});
		}

		int n_cb_bonds() const
		{
			return cb_bonds.size();
		}
		
		const std::map<int, int>& get_cb_bonds(int i) const
		{
			return cb_bonds[i];
		}

		void flip_spin(const std::pair<int, int>& b)
		{
			aux_spins[tau-1].flip(bond_index(b.first, b.second));
		}

		void buffer_equal_time_gf()
		{
			if (param.use_projector)
			{
				W_l_buffer = proj_W_l;
				W_r_buffer = proj_W_r;
				W_buffer = proj_W;
				
				//gf_buffer = equal_time_gf;
			}
			else
				gf_buffer = equal_time_gf;
			gf_buffer_partial_vertex = partial_vertex;
			gf_buffer_tau = tau;
		}

		void reset_equal_time_gf_to_buffer()
		{
			if (param.use_projector)
			{
				proj_W_l = W_l_buffer;
				proj_W_r = W_r_buffer;
				proj_W = W_buffer;
				
				//equal_time_gf = gf_buffer;
			}
			else
				equal_time_gf = gf_buffer;
			partial_vertex = gf_buffer_partial_vertex;
			tau = gf_buffer_tau;
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
			tau = max_tau;
			partial_vertex = 0;
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}

		void rebuild()
		{
			if (aux_spins.size() == 0) return;
			if (param.use_projector)
			{
				stabilizer.set_proj_l(n_intervals, id);
				for (int n = n_intervals - 1; n >= 0; --n)
				{
					dmatrix_t b = propagator((n + 1) * param.n_delta,
						n * param.n_delta);
					stabilizer.set_proj_l(n, b);
				}
				stabilizer.set_proj_r(0, id);
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set_proj_r(n, b);
				}
			}
			else
			{
				for (int n = 1; n <= n_intervals; ++n)
				{
					dmatrix_t b = propagator(n * param.n_delta,
						(n - 1) * param.n_delta);
					stabilizer.set(n, b);
				}
			}
		}

		void multiply_vertex_from_left(dmatrix_t& m,
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
					vm = &get_vertex_matrix(i, j, sigma);
				else
					vm = &get_inv_vertex_matrix(i, j, sigma);
				if (decoupled)
				{
					m.row(i).noalias() = old_m.row(i) * (*vm)(0, 0) + old_m.row(j) * (*vm)(0, 1);
					m.row(j).noalias() = old_m.row(i) * (*vm)(1, 0) + old_m.row(j) * (*vm)(1, 1);
				}
				else
				{
					int ns = l.n_sites();
					m.row(i).noalias() = old_m.row(i) * (*vm)(0, 0)
						+ old_m.row(j) * (*vm)(0, 1) + old_m.row(i+ns) * (*vm)(0, 2)
						+ old_m.row(j+ns) * (*vm)(0, 3);
					m.row(j).noalias() = old_m.row(i) * (*vm)(1, 0)
						+ old_m.row(j) * (*vm)(1, 1) + old_m.row(i+ns) * (*vm)(1, 2)
						+ old_m.row(j+ns) * (*vm)(1, 3);
					m.row(i+ns).noalias() = old_m.row(i) * (*vm)(2, 0)
						+ old_m.row(j) * (*vm)(2, 1) + old_m.row(i+ns) * (*vm)(2, 2)
						+ old_m.row(j+ns) * (*vm)(2, 3);
					m.row(j+ns).noalias() = old_m.row(i) * (*vm)(3, 0)
						+ old_m.row(j) * (*vm)(3, 1) + old_m.row(i+ns) * (*vm)(3, 2)
						+ old_m.row(j+ns) * (*vm)(3, 3);
				}
			}
		}

		void multiply_vertex_from_right(dmatrix_t& m,
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
					vm = &get_vertex_matrix(i, j, sigma);
				else
					vm = &get_inv_vertex_matrix(i, j, sigma);
				if (decoupled)
				{
					m.col(i).noalias() = old_m.col(i) * (*vm)(0, 0)
						+ old_m.col(j) * (*vm)(1, 0);
					m.col(j).noalias() = old_m.col(i) * (*vm)(0, 1)
						+ old_m.col(j) * (*vm)(1, 1);
				}
				else
				{
					int ns = l.n_sites();
					m.col(i).noalias() = old_m.col(i) * (*vm)(0, 0)
						+ old_m.col(j) * (*vm)(1, 0) + old_m.col(i+ns) * (*vm)(2, 0)
						+ old_m.col(j+ns) * (*vm)(3, 0);
					m.col(j).noalias() = old_m.col(i) * (*vm)(0, 1)
						+ old_m.col(j) * (*vm)(1, 1) + old_m.col(i+ns) * (*vm)(2, 1)
						+ old_m.col(j+ns) * (*vm)(3, 1);
					m.col(i+ns).noalias() = old_m.col(i) * (*vm)(0, 2)
						+ old_m.col(j) * (*vm)(1, 2) + old_m.col(i+ns) * (*vm)(2, 2)
						+ old_m.col(j+ns) * (*vm)(3, 2);
					m.col(j+ns).noalias() = old_m.col(i) * (*vm)(0, 3)
						+ old_m.col(j) * (*vm)(1, 3) + old_m.col(i+ns) * (*vm)(2, 3)
						+ old_m.col(j+ns) * (*vm)(3, 3);
				}
			}
		}

		void prepare_flip()
		{
			if (param.use_projector)
			{
				proj_W_l = proj_W_l * expH0;
				proj_W_r = invExpH0 * proj_W_r;
				
				//equal_time_gf = invExpH0 * equal_time_gf * expH0;
			}
			else
				equal_time_gf = invExpH0 * equal_time_gf * expH0;
		}

		void prepare_measurement()
		{
			if (param.use_projector)
			{
				proj_W_l = proj_W_l * invExpH0;
				proj_W_r = expH0 * proj_W_r;
				
				//equal_time_gf = expH0 * equal_time_gf * invExpH0;
			}
			else
				equal_time_gf = expH0 * equal_time_gf * invExpH0;
		}

		dmatrix_t propagator(int tau_n, int tau_m)
		{
			dmatrix_t b = id;
			for (int n = tau_n; n > tau_m; --n)
			{
				auto& vertex = aux_spins[n-1];
				if (param.tprime > 0. || param.tprime < 0.)
					b *= expH0;
				
				for (int bt = 0; bt < cb_bonds.size(); ++bt)
					multiply_vertex_from_right(b, bt, vertex, 1);
				
				if (param.tprime > 0. || param.tprime < 0.)
					b *= expH0;
			}
			return b;
		}
		
		void multiply_propagator_from_left(dmatrix_t& m, const arg_t& vertex, int inv)
		{
			if (inv == 1)
			{
				if (param.tprime > 0. || param.tprime < 0.)
					m = expH0 * m;
				
				for (int bt = cb_bonds.size() - 1; bt >= 0; --bt)
					multiply_vertex_from_left(m, bt, vertex, 1);
				
				if (param.tprime > 0. || param.tprime < 0.)
					m = expH0 * m;
			}
			else if (inv == -1)
			{
				if (param.tprime > 0. || param.tprime < 0.)
					m = invExpH0 * m;
				
				for (int bt = 0; bt < cb_bonds.size(); ++bt)
					multiply_vertex_from_left(m, bt, vertex, -1);
				
				if (param.tprime > 0. || param.tprime < 0.)
					m = invExpH0 * m;
			}
		}
		
		void multiply_propagator_from_right(dmatrix_t& m, const arg_t& vertex, int inv)
		{
			if (inv == 1)
			{
				if (param.tprime > 0. || param.tprime < 0.)
					m = m * expH0;
				
				for (int bt = 0; bt < cb_bonds.size(); ++bt)
					multiply_vertex_from_right(m, bt, vertex, 1);
				
				if (param.tprime > 0. || param.tprime < 0.)
					m = m * expH0;
			}
			else if (inv == -1)
			{
				if (param.tprime > 0. || param.tprime < 0.)
					m = m * invExpH0;
				
				for (int bt = cb_bonds.size() - 1; bt >= 0; --bt)
					multiply_vertex_from_right(m, bt, vertex, -1);
				
				if (param.tprime > 0. || param.tprime < 0.)
					m = m * invExpH0;
			}
		}

		void partial_advance(int partial_n)
		{
			int& p = partial_vertex;
			auto& vertex = aux_spins[tau-1];
			while (partial_n > p)
			{
				int bond_type = (p < cb_bonds.size()) ? p : 2*(cb_bonds.size()-1)-p;
				if (param.use_projector)
				{
					multiply_vertex_from_left(proj_W_r,
						bond_type, vertex, -1);
					multiply_vertex_from_right(proj_W_l,
						bond_type, vertex, 1);
					
					//multiply_vertex_from_left(equal_time_gf,
					//	bond_type, vertex, -1);
					//multiply_vertex_from_right(equal_time_gf,
					//	bond_type, vertex, 1);
				}
				else
				{
					multiply_vertex_from_left(equal_time_gf,
						bond_type, vertex, -1);
					multiply_vertex_from_right(equal_time_gf,
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
					multiply_vertex_from_left(proj_W_r,
						bond_type, vertex, 1);
					multiply_vertex_from_right(proj_W_l,
						bond_type, vertex, -1);
					
					//multiply_vertex_from_left(equal_time_gf,
					//	bond_type, vertex, 1);
					//multiply_vertex_from_right(equal_time_gf,
					//	bond_type, vertex, -1);
				}
				else
				{
					multiply_vertex_from_left(equal_time_gf,
						bond_type, vertex, 1);
					multiply_vertex_from_right(equal_time_gf,
						bond_type, vertex, -1);
				}
			}
		}

		void advance_forward()
		{
			auto& vertex = aux_spins[tau];
			if (param.use_projector)
			{
				multiply_propagator_from_left(proj_W_r, vertex, 1);
				multiply_propagator_from_right(proj_W_l, vertex, -1);
				
				//multiply_propagator_from_left(equal_time_gf, vertex, 1);
				//multiply_propagator_from_right(equal_time_gf, vertex, -1);
			}
			else
			{
				if (update_time_displaced_gf)
					multiply_propagator_from_left(time_displaced_gf, vertex, 1);
				multiply_propagator_from_left(equal_time_gf, vertex, 1);
				multiply_propagator_from_right(equal_time_gf, vertex, -1);
			}
			++tau;
		}

		void advance_backward()
		{
			auto& vertex = aux_spins[tau - 1];
			if (param.use_projector)
			{
				multiply_propagator_from_left(proj_W_r, vertex, -1);
				multiply_propagator_from_right(proj_W_l, vertex, 1);
				
				//multiply_propagator_from_left(equal_time_gf, vertex, -1);
				//multiply_propagator_from_right(equal_time_gf, vertex, 1);
			}
			else
			{
				if (update_time_displaced_gf)
					multiply_propagator_from_right(time_displaced_gf, vertex, 1);
				multiply_propagator_from_left(equal_time_gf, vertex, -1);
				multiply_propagator_from_right(equal_time_gf, vertex, 1);
			}
			--tau;
		}
		
		void stabilize_forward()
		{
			if (tau % param.n_delta != 0)
					return;
			// n = 0, ..., n_intervals - 1
			int n = tau / param.n_delta - 1;
			dmatrix_t b = propagator((n+1)*param.n_delta, n*param.n_delta);
			stabilizer.stabilize_forward(n, b);
		}
	
		void stabilize_backward()
		{
			if (tau % param.n_delta != 0)
					return;
			//n = n_intervals, ..., 1 
			int n = tau / param.n_delta + 1;
			dmatrix_t b = propagator(n*param.n_delta, (n-1)*param.n_delta);
			stabilizer.stabilize_backward(n, b);
		}

		complex_t try_ising_flip(int i, int j)
		{
			auto& vertex = aux_spins[tau-1];
			double sigma = vertex.get(bond_index(i, j));
			int m = std::min(i, j), n = std::max(i, j);
			last_flip = {m, n};
			delta = get_delta_matrix(m, n, sigma);
	
			if (param.use_projector)
			{
				dmatrix_t b_l(P.cols(), n_vertex_size);
				b_l.col(0) = proj_W_l.col(m);
				b_l.col(1) = proj_W_l.col(n);
				int ns = l.n_sites();
				if (param.decoupling == "majorana" && (!decoupled))
				{
					b_l.col(2) = proj_W_l.col(m+ns);
					b_l.col(3) = proj_W_l.col(n+ns);
				}
				W_W_l.noalias() = proj_W * b_l;
				dmatrix_t b_r(n_vertex_size, P.cols());
				b_r.row(0) = proj_W_r.row(m);
				b_r.row(1) = proj_W_r.row(n);
				if (param.decoupling == "majorana" && (!decoupled))
				{
					b_r.row(2) = proj_W_r.row(m+ns);
					b_r.row(3) = proj_W_r.row(n+ns);
				}
				delta_W_r.noalias() = delta * b_r;
				
				M = id_2;
				M.noalias() += delta_W_r * W_W_l;
				if (param.decoupling == "majorana" && decoupled)
					return M.determinant();
				else if (param.decoupling == "majorana" && (!decoupled))
					return std::sqrt(M.determinant());
				else
					return M.determinant();
				/*
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g(n_vertex_size, n_vertex_size);
				if (decoupled)
				{
					g << 1.-gf(m, m), -gf(m, n), -gf(n, m), 1.-gf(n, n);
					M = id_2; M.noalias() += g * delta;
					return M.determinant();
				}
				else
				{
					int ns = l.n_sites();
					g << 1.-gf(m, m), -gf(m, n), -gf(m, m+ns), -gf(m, n+ns),
					-gf(n, m), 1.-gf(n, n), -gf(n, m+ns), -gf(n, n+ns),
					-gf(m+ns, m), -gf(m+ns, n), 1.-gf(m+ns, m+ns), -gf(m+ns, n+ns),
					-gf(n+ns, m), -gf(n+ns, n), -gf(n+ns, m+ns), 1.-gf(n+ns, n+ns);
					M = id_2; M.noalias() += g * delta;
					return std::sqrt(M.determinant());
				}
				*/
			}
			else
			{
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g(n_vertex_size, n_vertex_size);
				if (decoupled)
				{
					g << 1.-gf(m, m), -gf(m, n), -gf(n, m), 1.-gf(n, n);
					M = id_2; M.noalias() += g * delta;
					return M.determinant();
				}
				else
				{
					int ns = l.n_sites();
					g << 1.-gf(m, m), -gf(m, n), -gf(m, m+ns), -gf(m, n+ns),
					-gf(n, m), 1.-gf(n, n), -gf(n, m+ns), -gf(n, n+ns),
					-gf(m+ns, m), -gf(m+ns, n), 1.-gf(m+ns, m+ns), -gf(m+ns, n+ns),
					-gf(n+ns, m), -gf(n+ns, n), -gf(n+ns, m+ns), 1.-gf(n+ns, n+ns);
					M = id_2; M.noalias() += g * delta;
					return std::sqrt(M.determinant());
				}
			}
		}

		void update_equal_time_gf_after_flip()
		{
			int indices[2] = {last_flip.first, last_flip.second};

			if (param.use_projector)
			{
				proj_W_r.row(indices[0]).noalias() += delta_W_r.row(0);
				proj_W_r.row(indices[1]).noalias() += delta_W_r.row(1);
				if (param.decoupling == "majorana" && (!decoupled))
				{
					int ns = l.n_sites();
					proj_W_r.row(indices[0]+ns).noalias() += delta_W_r.row(2);
					proj_W_r.row(indices[1]+ns).noalias() += delta_W_r.row(3);
				}
				
				M = M.inverse().eval();
				dmatrix_t delta_W_r_W = delta_W_r * proj_W;
				dmatrix_t W_W_l_M = W_W_l * M;
				proj_W.noalias() -= W_W_l_M * delta_W_r_W;
				
				/*
				M = M.inverse().eval();
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				if (param.decoupling == "majorana" && (!decoupled))
				{
					g_cols.col(2) = gf.col(indices[0]+l.n_sites());
					g_cols.col(3) = gf.col(indices[1]+l.n_sites());
				}
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				if (param.decoupling == "majorana" && (!decoupled))
				{
					g_rows.row(2) = gf.row(indices[0]+l.n_sites());
					g_rows.row(3) = gf.row(indices[1]+l.n_sites());
					g_rows(2, indices[0]+l.n_sites()) -= 1.;
					g_rows(3, indices[1]+l.n_sites()) -= 1.;
				}
				dmatrix_t gd = g_cols * delta;
				dmatrix_t mg = M * g_rows;
				gf.noalias() += gd * mg;
				*/
				
			}
			else
			{
				M = M.inverse().eval();
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				if (param.decoupling == "majorana" && (!decoupled))
				{
					g_cols.col(2) = gf.col(indices[0]+l.n_sites());
					g_cols.col(3) = gf.col(indices[1]+l.n_sites());
				}
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				if (param.decoupling == "majorana" && (!decoupled))
				{
					g_rows.row(2) = gf.row(indices[0]+l.n_sites());
					g_rows.row(3) = gf.row(indices[1]+l.n_sites());
					g_rows(2, indices[0]+l.n_sites()) -= 1.;
					g_rows(3, indices[1]+l.n_sites()) -= 1.;
				}
				dmatrix_t gd = g_cols * delta;
				dmatrix_t mg = M * g_rows;
				gf.noalias() += gd * mg;
			}
		}

		void static_measure(std::vector<double>& c, complex_t& n, complex_t& energy, complex_t& m2, complex_t& epsilon, complex_t& chern)
		{
			if (param.use_projector)
				equal_time_gf = id - proj_W_r * proj_W * proj_W_l;
			complex_t im = {0., 1.};
			for (int i = 0; i < l.n_sites(); ++i)
			{
				n += equal_time_gf(i, i) / complex_t(l.n_sites());
				if (!decoupled)
				{
					n += (equal_time_gf(i+l.n_sites(), i+l.n_sites())
						- im*equal_time_gf(i, i+l.n_sites()) + im*equal_time_gf(i+l.n_sites(), i))
						/ complex_t(l.n_sites());
				}
				for (int j = 0; j < l.n_sites(); ++j)
					{
						double re = std::real(equal_time_gf(i, j)
							* equal_time_gf(i, j));
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
			{
				energy += -l.parity(i.first) * param.t * std::imag(equal_time_gf(i.second, i.first))
					+ param.V * std::real(equal_time_gf(i.second, i.first) * equal_time_gf(i.second, i.first)) / 2.;
				
				epsilon += im * l.parity(i.first) * equal_time_gf(i.second, i.first) / complex_t(l.n_bonds());
			}
			for (auto& i : l.bonds("chern"))
				chern += im * (equal_time_gf(i.second, i.first) - equal_time_gf(i.first, i.second)) / complex_t(l.n_bonds());
		}
		
		void measure_static_observable(std::vector<double>& values,
			const std::vector<wick_static_base<dmatrix_t>>& obs)
		{
			if (param.use_projector)
				equal_time_gf = id - proj_W_r * proj_W * proj_W_l;
			for (int i = 0; i < values.size(); ++i)
					values[i] = obs[i].get_obs(equal_time_gf);
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
				time_displaced_gf = id;
				
				for (int n = 0; n < param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_backward();
						stabilize_backward();
					}
					et_gf_L[n] = id;
					et_gf_L[n].noalias() -= proj_W_r * proj_W * proj_W_l;
					//et_gf_L[n] = equal_time_gf;
				}
				dmatrix_t et_gf_0 = id;
				et_gf_0.noalias() -= proj_W_r * proj_W * proj_W_l;
				//dmatrix_t et_gf_0 = equal_time_gf;
				for (int n = 0; n < 2*param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_backward();
						stabilize_backward();
					}
					equal_time_gf = id;
					equal_time_gf.noalias() -= proj_W_r * proj_W * proj_W_l;
					if (n < param.n_discrete_tau)
						et_gf_R[n] = equal_time_gf;
					if (n % 2 == 1)
						et_gf_T[n/2] = equal_time_gf;
					//if (n < param.n_discrete_tau)
					//	et_gf_T[n] = equal_time_gf;
				}
				
				for (int i = 0; i < dyn_tau.size(); ++i)
					dyn_tau[i][0] = obs[i].get_obs(et_gf_0, et_gf_0, et_gf_0);
				for (int n = 1; n <= param.n_discrete_tau; ++n)
				{
					dmatrix_t g_l = propagator(max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau) * et_gf_L[et_gf_L.size() - n];
					int n_r = max_tau/2 - n;
					dmatrix_t g_r = propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau) * et_gf_R[n-1];
					time_displaced_gf = g_l * time_displaced_gf * g_r;
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][n] = obs[i].get_obs(et_gf_0, et_gf_T[n-1],
							time_displaced_gf);
				}
				
				reset_equal_time_gf_to_buffer();
				stabilizer.restore_buffer();
			}
			else
			{
				// 1 = forward, -1 = backward
				int direction = tau == 0 ? 1 : -1;
				dmatrix_t et_gf_0 = equal_time_gf;
				enable_time_displaced_gf(direction);
				time_displaced_gf = equal_time_gf;
				for (int n = 0; n <= max_tau; ++n)
				{
					if (n % (max_tau / param.n_discrete_tau) == 0)
					{
						int t = n / (max_tau / param.n_discrete_tau);
						for (int i = 0; i < dyn_tau.size(); ++i)
							dyn_tau[i][t] = obs[i].get_obs(et_gf_0, equal_time_gf,
								time_displaced_gf);
					}
					if (direction == 1 && tau < max_tau)
					{
						advance_forward();
						stabilize_forward();
					}
					else if (direction == -1 && tau > 0)
					{
						advance_backward();
						stabilize_backward();
					}
				}
				disable_time_displaced_gf();
				if (direction == 1)
					tau = 0;
				else if (direction == -1)
					tau = max_tau;
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
		int tau;
		int partial_vertex;
		int max_tau;
		std::vector<arg_t> aux_spins;
		std::map<std::pair<int, int>, int> bond_indices;
		std::vector<arg_t> arg_buffer;
		std::vector<int> pos_buffer;
		bool update_time_displaced_gf;
		bool decoupled;
		int n_vertex_size;
		int n_matrix_size;
		std::vector<dmatrix_t> vertex_matrices;
		std::vector<dmatrix_t> inv_vertex_matrices;
		std::vector<dmatrix_t> delta_matrices;
		dmatrix_t equal_time_gf;
		dmatrix_t time_displaced_gf;
		dmatrix_t proj_W_l;
		dmatrix_t proj_W_r;
		dmatrix_t proj_W;
		dmatrix_t gf_buffer;
		dmatrix_t W_l_buffer;
		dmatrix_t W_r_buffer;
		dmatrix_t W_buffer;
		int gf_buffer_partial_vertex;
		int gf_buffer_tau;
		dmatrix_t id;
		dmatrix_t id_2;
		dmatrix_t expH0;
		dmatrix_t invExpH0;
		dmatrix_t P;
		dmatrix_t Pt;
		dmatrix_t delta;
		dmatrix_t delta_W_r;
		dmatrix_t W_W_l;
		dmatrix_t M;
		std::pair<int, int> last_flip;
		std::vector<std::map<int, int>> cb_bonds;
		stabilizer_t stabilizer;
};
