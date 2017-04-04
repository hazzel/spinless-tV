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

		fast_update(Random& rng_, const lattice& l_, parameters& param_,
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
			n_intervals = max_tau / param.n_delta;
			stabilizer.resize(n_intervals, n_matrix_size);
			rebuild();
		}
		
		void initialize()
		{
			decoupled = true;

			n_vertex_size = 2;
			n_matrix_size = l.n_sites();
			
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
			build_dirac_H0(H0);
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H0);
			T = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			invT = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < T.rows(); ++i)
			{
				T(i, i) = std::exp(-solver.eigenvalues()[i] * param.dtau);
				invT(i, i) = std::exp(solver.eigenvalues()[i] * param.dtau);
			}
			T = solver.eigenvectors() * T * solver.eigenvectors()
				.inverse();
			invT = solver.eigenvectors() * invT * solver.eigenvectors()
				.inverse();
			
			U = dmatrix_t(2, 2), invU = dmatrix_t(2, 2);
			U << 1./std::sqrt(2.), 1./std::sqrt(2.),
				-1./std::sqrt(2.), 1./std::sqrt(2.);
			invU << 1./std::sqrt(2.), -1./std::sqrt(2.),
				1./std::sqrt(2.), 1./std::sqrt(2.);
			
			for (int i = 0; i < nn_bonds.size(); ++i)
			{
				fullUForward.push_back(id);
				fullInvUForward.push_back(id);
				fullUBackward.push_back(id);
				fullInvUBackward.push_back(id);
			}
			for (int bt = 0; bt < nn_bonds.size(); ++bt)
			{
				for (int i = 0; i < nn_bonds[bt].size(); ++i)
					multiply_from_right(fullUForward[bt], U, nn_bonds[bt][i].first, nn_bonds[bt][i].second);
				for (int i = 0; i < nn_bonds[bt].size(); ++i)
					multiply_from_right(fullUBackward[bt], invU, inv_nn_bonds[bt][i].first, inv_nn_bonds[bt][i].second);
				for (int i = 0; i < nn_bonds[bt].size(); ++i)
					multiply_from_right(fullInvUForward[bt], invU, nn_bonds[bt][i].first, nn_bonds[bt][i].second);
				for (int i = 0; i < nn_bonds[bt].size(); ++i)
					multiply_from_right(fullInvUBackward[bt], U, inv_nn_bonds[bt][i].first, inv_nn_bonds[bt][i].second);
			}
			
			if (param.use_projector)
			{
				dmatrix_t broken_H0 = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
				build_broken_dirac_H0(broken_H0);
				
				if (param.L % 3 == 0)
					P = symmetrize_EV(broken_H0);
				else
				{
					solver.compute(broken_H0);
					P = solver.eigenvectors().block(0, 0, n_matrix_size, n_matrix_size / 2);
				}
				
				//std::cout << solver.eigenvalues() << std::endl;
				//std::cout << P.adjoint() * P << std::endl;
				
				/*
				for (int i = 0; i < 360; ++i)
				{
					bool res = l.check_rotation_symmetry(i);
					if (res)
						std::cout << "Rotation " << i << std::endl;
				}
				*/
				
				Pt = P.adjoint();
				stabilizer.set_P(P, Pt);
			}
			stabilizer.set_method(param.use_projector);
		}
		
		dmatrix_t symmetrize_EV(const dmatrix_t& H)
		{
			Eigen::SelfAdjointEigenSolver<dmatrix_t> solver(H);
			auto& S = solver.eigenvectors();
			auto& en = solver.eigenvalues();
			dmatrix_t pm = dmatrix_t::Zero(n_matrix_size, n_matrix_size);
			for (int i = 0; i < n_matrix_size; ++i)
				pm(i, l.inverted_site(i)) = 1.;
			double epsilon = std::pow(10., -5.);

			dmatrix_t S_s = S + pm * S;
			dmatrix_t S_a = S - pm * S;
			dmatrix_t S_so(n_matrix_size, n_matrix_size);
			dmatrix_t S_ao(n_matrix_size, n_matrix_size);
			dmatrix_t S_f = dmatrix_t::Zero(n_matrix_size, 2*n_matrix_size);

			for (int i = 0; i < n_matrix_size; ++i)
			{
				if (S_s.col(i).norm() > epsilon)
					S_s.col(i) /= S_s.col(i).norm();
				else
					S_s.col(i) *= 0.;
				if (S_a.col(i).norm() > epsilon)
					S_a.col(i) /= S_a.col(i).norm();
				else
					S_a.col(i) *= 0.;
			}

			int cnt = 0;
			for (int i = 0; i < n_matrix_size; ++i)
			{
				int j;
				for (j = i; j < n_matrix_size && std::abs(en(j)-en(i)) < epsilon ; ++j)
				{
					S_so.col(j) = S_s.col(j);
					S_ao.col(j) = S_a.col(j);
					for (int k = i; k < j; ++k)
					{
						S_so.col(j) -= S_so.col(k) * (S_so.col(k).dot(S_s.col(j)));
						S_ao.col(j) -= S_ao.col(k) * (S_ao.col(k).dot(S_a.col(j)));
					}
					if (S_so.col(j).norm() > epsilon)
					{
						S_so.col(j) /= S_so.col(j).norm();
						S_f.col(cnt) = S_so.col(j);
						++cnt;
					}
					if (S_ao.col(j).norm() > epsilon)
					{
						S_ao.col(j) /= S_ao.col(j).norm();
						S_f.col(cnt) = S_ao.col(j);
						++cnt;
					}
				}
				i = j - 1;
			}
			if (cnt != n_matrix_size)
				std::cout << "Error! Found " << cnt << " out of " << 2*n_matrix_size << std::endl;
			if (param.inv_symmetry == 1)
				return S_f.block(0, 0, n_matrix_size, n_matrix_size / 2);
			else
				return S_f.block(0, n_matrix_size / 2, n_matrix_size, n_matrix_size / 2);
		}
		
		void build_dirac_H0(dmatrix_t& H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
				H0(a.first, a.second) = {-param.t, 0.};
			for (auto& a : l.bonds("d3_bonds"))
				H0(a.first, a.second) = {-param.tprime, 0.};
			for (int i = 0; i < l.n_sites(); ++i)
				H0(i, i) = l.parity(i) * param.stag_mu + param.mu;
		}
		
		void build_broken_dirac_H0(dmatrix_t& broken_H0)
		{
			for (auto& a : l.bonds("nearest neighbors"))
				broken_H0(a.first, a.second) = {-param.t, 0.};
			for (auto& a : l.bonds("d3_bonds"))
				broken_H0(a.first, a.second) = {-param.tprime, 0.};
			for (int i = 0; i < l.n_sites(); ++i)
				broken_H0(i, i) = l.parity(i) * param.stag_mu + param.mu;
			
			for (auto& a : l.bonds("chern"))
			{
				double tp = 0.00001;
				broken_H0(a.first, a.second) = {0., -tp};
				broken_H0(a.second, a.first) = {0., tp};
			}
			for (auto& a : l.bonds("chern_2"))
			{
				double tp = 0.00001;
				broken_H0(a.first, a.second) = {0., -tp};
				broken_H0(a.second, a.first) = {0., tp};
			}
		}
		
		void build_dirac_vertex(int cnt, double spin)
		{
			vertex_matrices[cnt] << std::exp(-param.lambda * spin), 0,
				0, std::exp(param.lambda * spin);
			inv_vertex_matrices[cnt] << std::exp(param.lambda * spin), 0,
				0, std::exp(-param.lambda * spin);
			delta_matrices[cnt] << std::exp(2.*param.lambda * spin) - 1.
				, 0, 0, std::exp(-2.*param.lambda * spin) - 1.;
		}
		
		void build_vertex_matrices()
		{
			vertex_matrices.resize(2, dmatrix_t(n_vertex_size, n_vertex_size));
			inv_vertex_matrices.resize(2, dmatrix_t(n_vertex_size, n_vertex_size));
			delta_matrices.resize(2, dmatrix_t(n_vertex_size, n_vertex_size));
			int cnt = 0;
			for (double spin : {1., -1.})
			{
				build_dirac_vertex(cnt, spin);
				++cnt;
			}
		}
		
		dmatrix_t& get_vertex_matrix(int i, int j, int s)
		{
			return vertex_matrices[static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_inv_vertex_matrix(int i, int j, int s)
		{
			return inv_vertex_matrices[static_cast<int>(s<0)];
		}
		
		dmatrix_t& get_delta_matrix(int i, int j, int s)
		{
			return delta_matrices[static_cast<int>(s<0)];
		}

		int get_bond_type(const std::pair<int, int>& bond) const
		{
			for (int i = 0; i < cb_bonds.size(); ++i)
				if (cb_bonds[i].at(bond.first) == bond.second)
					return i;
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
		
		void update_tau()
		{
			tau += param.direction;
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
		
		const std::vector<std::pair<int, int>>& get_nn_bonds(int b) const
		{
			return nn_bonds[b];
		}
		
		const std::vector<std::pair<int, int>>& get_inv_nn_bonds(int b) const
		{
			return inv_nn_bonds[b];
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
			gf_buffer_tau = tau;
			dir_buffer = param.direction;
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
			tau = gf_buffer_tau;
			param.direction = dir_buffer;
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
		
		void multiply_from_left(dmatrix_t& m, dmatrix_t& vm, int i, int j)
		{
			dmatrix_t old_i = m.row(i), old_j = m.row(j);
			m.row(i).noalias() = old_i * vm(0, 0) + old_j * vm(0, 1);
			m.row(j).noalias() = old_i * vm(1, 0) + old_j * vm(1, 1);
		}

		void multiply_from_right(dmatrix_t& m, dmatrix_t& vm, int i, int j)
		{
			dmatrix_t old_i = m.col(i), old_j = m.col(j);
			m.col(i).noalias() = old_i * vm(0, 0) + old_j * vm(1, 0);
			m.col(j).noalias() = old_i * vm(0, 1) + old_j * vm(1, 1);
		}

		void multiply_T_matrix()
		{
			if (param.use_projector)
			{
				if (param.direction == 1)
				{
					proj_W_l = proj_W_l * invT;
					proj_W_r = T * proj_W_r;
				}
				else if (param.direction == -1)
				{
					proj_W_l = proj_W_l * T;
					proj_W_r = invT * proj_W_r;
				}
			}
			else
			{
				if (param.direction == 1)
					equal_time_gf = T * equal_time_gf * invT;
				else if (param.direction == -1)
					equal_time_gf = invT * equal_time_gf * T;
			}
		}
		
		void multiply_U_matrices(int bt, int inv)
		{
			/*
			if (param.use_projector)
			{
				if (inv == 1 && param.direction == 1)
				{
					proj_W_l = proj_W_l * fullUBackward[bt];
					proj_W_r = fullUForward[bt] * proj_W_r;
				}
				else if (inv == 1  && param.direction == -1)
				{
					proj_W_l = proj_W_l * fullUForward[bt];
					proj_W_r = fullUBackward[bt] * proj_W_r;
				}
				else if (inv == -1 && param.direction == 1)
				{
					proj_W_l = proj_W_l * fullInvUBackward[bt];
					proj_W_r = fullInvUForward[bt] * proj_W_r;
				}
				else if (inv == -1 && param.direction == -1)
				{
					proj_W_l = proj_W_l * fullInvUForward[bt];
					proj_W_r = fullInvUBackward[bt] * proj_W_r;
				}
			}
			else
			{
				if (inv == 1 && param.direction == 1)
					equal_time_gf = fullUForward[bt] * equal_time_gf * fullUBackward[bt];
				else if (inv == 1 && param.direction == -1)
					equal_time_gf = fullUBackward[bt] * equal_time_gf * fullUForward[bt];
				else if (inv == -1 && param.direction == 1)
					equal_time_gf = fullInvUForward[bt] * equal_time_gf * fullInvUBackward[bt];
				else if (inv == -1 && param.direction == -1)
					equal_time_gf = fullInvUBackward[bt] * equal_time_gf * fullInvUForward[bt];
			}
			*/
			
			dmatrix_t& u = inv == 1 ? U : invU;
			dmatrix_t& iu = inv == 1 ? invU : U;
			for (int i = 0; i < nn_bonds[bt].size(); ++i)
			{
				int m, n;
				if (param.direction == 1)
				{
					m = inv_nn_bonds[bt][i].first;
					n = inv_nn_bonds[bt][i].second;
				}
				else if (param.direction == -1)
				{
					m = nn_bonds[bt][i].first;
					n = nn_bonds[bt][i].second;
				}
				
				if (param.use_projector)
				{
					if (param.direction == 1)
					{
						multiply_from_left(proj_W_r, u, m, n);
						multiply_from_right(proj_W_l, iu, m, n);
					}
					else if (param.direction == -1)
					{
						multiply_from_left(proj_W_r, iu, m, n);
						multiply_from_right(proj_W_l, u, m, n);
					}
				}
				else
				{
					if (param.direction == 1)
					{
						multiply_from_left(equal_time_gf, u, m, n);
						multiply_from_right(equal_time_gf, iu, m, n);
					}
					else if (param.direction == -1)
					{
						multiply_from_left(equal_time_gf, iu, m, n);
						multiply_from_right(equal_time_gf, u, m, n);
					}
				}
			}
			
		}
		
		void multiply_Gamma_matrix(int i, int j)
		{
			auto& vertex = aux_spins[tau-1];
			double sigma = vertex.get(bond_index(i, j));
			dmatrix_t& v = get_vertex_matrix(i, j, sigma);
			dmatrix_t& iv = get_inv_vertex_matrix(i, j, sigma);
			if (param.use_projector)
			{
				if (param.direction == 1)
				{
					multiply_from_left(proj_W_r, v, i, j);
					multiply_from_right(proj_W_l, iv, i, j);
				}
				else if (param.direction == -1)
				{
					multiply_from_left(proj_W_r, iv, i, j);
					multiply_from_right(proj_W_l, v, i, j);
				}
			}
			else
			{
				if (param.direction == 1)
				{
					multiply_from_left(equal_time_gf, v, i, j);
					multiply_from_right(equal_time_gf, iv, i, j);
				}	
				else if (param.direction == -1)
				{
					multiply_from_left(equal_time_gf, iv, i, j);
					multiply_from_right(equal_time_gf, v, i, j);
				}
			}
		}

		dmatrix_t propagator(int tau_n, int tau_m)
		{
			dmatrix_t b = id;
			for (int n = tau_n; n > tau_m; --n)
			{
				auto& vertex = aux_spins[n-1];
				
				for (int bt = 0; bt < nn_bonds.size(); ++bt)
				{
					//b *= fullUForward[bt];
					for (int i = 0; i < nn_bonds[bt].size(); ++i)
						multiply_from_right(b, U, nn_bonds[bt][i].first, nn_bonds[bt][i].second);

					for (int i = 0; i < nn_bonds[bt].size(); ++i)
					{
						double sigma = vertex.get(bond_index(nn_bonds[bt][i].first, nn_bonds[bt][i].second));
						dmatrix_t& v = get_vertex_matrix(nn_bonds[bt][i].first, nn_bonds[bt][i].second, sigma);
						multiply_from_right(b, v, nn_bonds[bt][i].first, nn_bonds[bt][i].second);
					}
					
					//b *= fullInvUForward[bt];
					for (int i = 0; i < nn_bonds[bt].size(); ++i)
						multiply_from_right(b, invU, nn_bonds[bt][i].first, nn_bonds[bt][i].second);
				}
				b *= T;
			}
			return b;
		}
		
		void advance_time_slice()
		{
			if (param.direction == 1)
			{
				update_tau();
				multiply_T_matrix();
				
				for (int bt = nn_bonds.size() - 1; bt >= 0; --bt)
				{
					multiply_U_matrices(bt, -1);
					
					for (int i = 0; i < nn_bonds[bt].size(); ++i)
					{
						int m = inv_nn_bonds[bt][inv_nn_bonds[bt].size() - 1 - i].first;
						int n = inv_nn_bonds[bt][inv_nn_bonds[bt].size() - 1 - i].second;
					
						multiply_Gamma_matrix(m, n);
					}
					
					multiply_U_matrices(bt, 1);
				}
			}
			else if (param.direction == -1)
			{
				for (int bt = 0; bt < nn_bonds.size(); ++bt)
				{
					multiply_U_matrices(bt, 1);
					
					for (int i = 0; i < nn_bonds[bt].size(); ++i)
						multiply_Gamma_matrix(nn_bonds[bt][i].first, nn_bonds[bt][i].second);
					
					multiply_U_matrices(bt, -1);
				}
				
				update_tau();
				multiply_T_matrix();
			}
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
			last_flip = {i, j};
			delta = get_delta_matrix(i, j, sigma);
	
			if (param.use_projector)
			{
				dmatrix_t b_l(P.cols(), n_vertex_size);
				b_l.col(0) = proj_W_l.col(i);
				b_l.col(1) = proj_W_l.col(j);
				W_W_l.noalias() = proj_W * b_l;
				dmatrix_t b_r(n_vertex_size, P.cols());
				b_r.row(0) = proj_W_r.row(i);
				b_r.row(1) = proj_W_r.row(j);
				delta_W_r.noalias() = delta * b_r;
				
				M = id_2;
				M.noalias() += delta_W_r * W_W_l;
				return M.determinant();
			}
			else
			{
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g(n_vertex_size, n_vertex_size);
				g << 1.-gf(i, i), -gf(i, j), -gf(j, i), 1.-gf(j, j);
				M = id_2; M.noalias() += g * delta;
				return M.determinant();
			}
		}

		void update_equal_time_gf_after_flip()
		{
			int indices[2] = {last_flip.first, last_flip.second};
			
			if (param.use_projector)
			{
				proj_W_r.row(indices[0]).noalias() += delta_W_r.row(0);
				proj_W_r.row(indices[1]).noalias() += delta_W_r.row(1);
				
				M = M.inverse().eval();
				dmatrix_t delta_W_r_W = delta_W_r * proj_W;
				dmatrix_t W_W_l_M = W_W_l * M;
				proj_W.noalias() -= W_W_l_M * delta_W_r_W;
			}
			else
			{
				
				M = M.inverse().eval();
				dmatrix_t& gf = equal_time_gf;
				dmatrix_t g_cols(n_matrix_size, n_vertex_size);
				g_cols.col(0) = gf.col(indices[0]);
				g_cols.col(1) = gf.col(indices[1]);
				dmatrix_t g_rows(n_vertex_size, n_matrix_size);
				g_rows.row(0) = gf.row(indices[0]);
				g_rows.row(1) = gf.row(indices[1]);
				g_rows(0, indices[0]) -= 1.;
				g_rows(1, indices[1]) -= 1.;
				dmatrix_t gd = g_cols * delta;
				dmatrix_t mg = M * g_rows;
				gf.noalias() += gd * mg;
				
				/*
				for (int z = 0; z < 2; ++z)
				{
					std::complex<double> denom = 1. + delta(z, z) * (1. - equal_time_gf(indices[z], indices[z]));
					dmatrix_t i_col = equal_time_gf.col(indices[z]) * delta(z, z) / denom;
					dmatrix_t j_row = equal_time_gf.row(indices[z]);
					j_row(0, indices[z]) -= 1.;
					equal_time_gf.noalias() -= i_col * j_row;
				}
				*/
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
			
			if (param.mu != 0 || param.stag_mu != 0)
			{
				complex_t n = {0., 0.};
				complex_t im = {0., 1.};
				for (int i = 0; i < l.n_sites(); ++i)
					n += equal_time_gf(i, i) / complex_t(l.n_sites());
				measure.add("n_re", std::real(n*param.sign_phase));
				measure.add("n_im", std::imag(n*param.sign_phase));
				measure.add("n", std::real(n));
			}
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
				param.direction = -1;
				
				
				for (int n = 0; n < param.n_discrete_tau; ++n)
				{
					for (int m = 0; m < param.n_dyn_tau; ++m)
					{
						advance_time_slice();
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
						advance_time_slice();
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
				
				dmatrix_t td_rev = id;
				for (int i = 0; i < dyn_tau.size(); ++i)
					dyn_tau[i][0] = obs[i].get_obs(et_gf_0, et_gf_0, et_gf_0, et_gf_0);
				for (int n = 1; n <= param.n_discrete_tau; ++n)
				{
					dmatrix_t g_l = propagator(max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau) * et_gf_L[et_gf_L.size() - n];
					dmatrix_t g_l_rev = -(id - et_gf_L[et_gf_L.size() - n]) * propagator(max_tau/2 + n*param.n_dyn_tau,
						max_tau/2 + (n-1)*param.n_dyn_tau).inverse();
					int n_r = max_tau/2 - n;
					dmatrix_t g_r = propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau) * et_gf_R[n-1];
					dmatrix_t g_r_rev = -(id - et_gf_R[n-1]) * propagator(max_tau/2 - (n-1)*param.n_dyn_tau,
						max_tau/2 - n*param.n_dyn_tau).inverse();
					time_displaced_gf = g_l * time_displaced_gf * g_r;
					td_rev = g_r_rev * td_rev * g_l_rev;
					for (int i = 0; i < dyn_tau.size(); ++i)
						dyn_tau[i][n] = obs[i].get_obs(et_gf_0, et_gf_T[n-1],
							time_displaced_gf, td_rev);
						
					//std::cout << "n = " << n << std::endl;
					//std::cout << "td_gf" << std::endl;
					//print_matrix(time_displaced_gf);
					//std::cout << "td_gf_rev" << std::endl;
					//print_matrix(td_rev);
				}
				
				reset_equal_time_gf_to_buffer();
				stabilizer.restore_buffer();
			}
			else
			{
				// 1 = forward, -1 = backward
				int direction = tau == 0 ? 1 : -1;
				dir_buffer = param.direction;
				param.direction = direction;
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
								time_displaced_gf, time_displaced_gf);
					}
					if (direction == 1 && tau < max_tau)
					{
						advance_time_slice();
						//stabilize_forward();
					}
					else if (direction == -1 && tau > 0)
					{
						advance_time_slice();
						//stabilize_backward();
					}
				}
				disable_time_displaced_gf();
				if (direction == 1)
					tau = 0;
				else if (direction == -1)
					tau = max_tau;
				param.direction = dir_buffer;
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
			
			nn_bonds.resize(3);
			for (int b = 0; b < 3; ++b)
				for (int i = 0; i < l.n_sites(); ++i)
				{
					int j = cb_bonds[b][i];
					if (i > j) continue;
					nn_bonds[b].push_back({i, j});
				}
			inv_nn_bonds.resize(3);
			for (int b = 0; b < 3; ++b)
				for (int i = nn_bonds[b].size() - 1; i >= 0; --i)
					inv_nn_bonds[b].push_back(nn_bonds[b][i]);
		}

		void print_matrix(const dmatrix_t& m)
		{
			Eigen::IOFormat clean(6, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
		
public:
		void print_gf()
		{
			std::cout << "tau = " << tau << std::endl;
			std::cout << "gf" << std::endl;
			equal_time_gf = id - proj_W_r * proj_W * proj_W_l;
			print_matrix(equal_time_gf);
			//std::cout << "correct" << std::endl;
			//dmatrix_t cgf = (id + propagator(tau, 0) * propagator(max_tau, tau)).inverse();
			//print_matrix(cgf);
		}
	private:
		Random& rng;
		const lattice& l;
		parameters& param;
		measurements& measure;
		int n_intervals;
		int tau;
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
		int gf_buffer_tau;
		int dir_buffer;
		dmatrix_t id;
		dmatrix_t id_2;
		dmatrix_t T;
		dmatrix_t invT;
		dmatrix_t U;
		dmatrix_t invU;
		std::vector<dmatrix_t> fullUForward;
		std::vector<dmatrix_t> fullInvUForward;
		std::vector<dmatrix_t> fullUBackward;
		std::vector<dmatrix_t> fullInvUBackward;
		dmatrix_t P;
		dmatrix_t Pt;
		dmatrix_t delta;
		dmatrix_t delta_W_r;
		dmatrix_t W_W_l;
		dmatrix_t M;
		std::pair<int, int> last_flip;
		std::vector<std::map<int, int>> cb_bonds;
		std::vector<std::vector<std::pair<int, int>>> nn_bonds;
		std::vector<std::vector<std::pair<int, int>>> inv_nn_bonds;
		stabilizer_t stabilizer;
};
