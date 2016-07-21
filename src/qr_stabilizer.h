#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/SVD>
#include "measurements.h"
#include "dump.h"
#include "lattice.h"
#include "parameters.h"
#include "Random.h"

class qr_stabilizer
{
	public:
		using complex_t = std::complex<double>;
		template<int n, int m>
		using matrix_t = Eigen::Matrix<complex_t, n, m>;
		using dmatrix_t = matrix_t<Eigen::Dynamic, Eigen::Dynamic>;

		qr_stabilizer(measurements& measure_,
			std::vector<dmatrix_t>& equal_time_gf_,
			std::vector<dmatrix_t>& time_displaced_gf_,
			std::vector<dmatrix_t>& proj_W_l_, std::vector<dmatrix_t>&
			proj_W_r_, std::vector<dmatrix_t>& proj_W_, int n_species_)
			: measure(measure_), update_time_displaced_gf(false),
			n_species(n_species_),
			equal_time_gf(equal_time_gf_), time_displaced_gf(time_displaced_gf_),
			proj_W_l(proj_W_l_), proj_W_r(proj_W_r_), proj_W(proj_W_)
		{}

		void set_method(bool use_projector_)
		{
			use_projector = use_projector_;
		}
		
		void enable_time_displaced_gf(int direction)
		{
			update_time_displaced_gf = true;
			sweep_direction = direction;
			U_buffer = U;
			D_buffer = D;
			V_buffer = V;
		}
		void disable_time_displaced_gf()
		{
			update_time_displaced_gf = false;
			U = U_buffer;
			D = D_buffer;
			V = V_buffer;
		}

		void resize(int n_intervals_, int dimension)
		{
			id_N = dmatrix_t::Identity(dimension, dimension);
			n_intervals = n_intervals_;
			if (use_projector)
			{
				proj_U_l.resize(boost::extents[n_species][n_intervals + 1]);
				proj_D_l.resize(boost::extents[n_species][n_intervals + 1]);
				proj_V_l.resize(boost::extents[n_species][n_intervals + 1]);
				proj_U_r.resize(boost::extents[n_species][n_intervals + 1]);
				proj_D_r.resize(boost::extents[n_species][n_intervals + 1]);
				proj_V_r.resize(boost::extents[n_species][n_intervals + 1]);
			}
			else
			{
				U.resize(boost::extents[n_species][n_intervals + 1]);
				D.resize(boost::extents[n_species][n_intervals + 1]);
				V.resize(boost::extents[n_species][n_intervals + 1]);
				U_buffer.resize(boost::extents[n_species][n_intervals + 1]);
				D_buffer.resize(boost::extents[n_species][n_intervals + 1]);
				V_buffer.resize(boost::extents[n_species][n_intervals + 1]);
				for (int s = 0; s < n_species; ++s)
					for (int n = 0; n < n_intervals + 1; ++n)
					{
						U[s][n] = id_N; D[s][n] = id_N; V[s][n] = id_N;
					}
			}
		}

		void set(int s, int n, const dmatrix_t& b)
		{
			qr_solver.compute((b * U[s][n-1]) * D[s][n-1]);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			U[s][n] = qr_solver.matrixQ();
			D[s][n] = qr_solver.matrixQR().diagonal().asDiagonal();
			V[s][n] = (D[s][n].inverse() * R) * (qr_solver.colsPermutation()
				.transpose() * V[s][n-1]);
			if (n == n_intervals)
			{
				recompute_equal_time_gf(s, id_N, id_N, id_N, U[s][n_intervals],
					D[s][n_intervals], V[s][n_intervals]);
				if (s == n_species - 1)
					init = true;
				U[s][n_intervals] = id_N;
				D[s][n_intervals] = id_N;
				V[s][n_intervals] = id_N;
			}
		}
		
		void set_proj_r(int s, int n, const dmatrix_t& b, const dmatrix_t& P)
		{
			/*
			if (n == 0)
				qr_solver.compute(P);
			else
				qr_solver.compute((b * proj_U_r[s][n-1]) * proj_D_r[s][n-1]);
			dmatrix_t p_q = dmatrix_t::Identity(P.rows(), P.cols());
			dmatrix_t p_r = dmatrix_t::Identity(P.cols(), P.rows());
			dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			proj_U_r[s][n] = qr_solver.matrixQ() * p_q;
			proj_D_r[s][n] = qr_solver.matrixQR().diagonal().asDiagonal();
			if (n == 0)
				proj_V_r[s][n] = proj_D_r[s][n].inverse() * (p_r * r) * qr_solver.colsPermutation().transpose();
			else
				proj_V_r[s][n] = proj_D_r[s][n].inverse() * (p_r * r) * qr_solver.colsPermutation().transpose()
				* proj_V_r[s][n-1];
			proj_W_r[s] = proj_U_r[s][n];
			*/
			if (n == 0)
				svd_solver.compute(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
			else
				svd_solver.compute((b * proj_U_r[s][n-1]) * proj_D_r[s][n-1]);
			proj_U_r[s][n] = svd_solver.matrixU();
			proj_D_r[s][n] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
			if (n == 0)
				proj_V_r[s][n] = svd_solver.matrixV().adjoint();
			else
				proj_V_r[s][n] = svd_solver.matrixV().adjoint() * proj_V_r[s][n-1];

			if (n == n_intervals)
			{
				proj_W_r[s] = proj_U_r[s][n];
				proj_W_l[s] = proj_U_l[s][n];
				proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				if (s == n_species - 1)
					init = true;
			}
		}
		
		void set_proj_l(int s, int n, const dmatrix_t& b, const dmatrix_t& Pt)
		{
			/*
			if (n == n_intervals)
				qr_solver.compute(Pt);
			else
				qr_solver.compute(proj_D_l[s][n+1] * (proj_U_l[s][n+1] * b));
			dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			if (n == n_intervals)
				proj_V_l[s][n] = qr_solver.matrixQ();
			else
				proj_V_l[s][n] = proj_V_l[s][n+1] * qr_solver.matrixQ();
			proj_D_l[s][n] = qr_solver.matrixQR().diagonal().asDiagonal();
			proj_U_l[s][n] = (proj_D_l[s][n].inverse() * r) * qr_solver.colsPermutation()
				.transpose();
			proj_W_l[s] = proj_U_l[s][n];
			*/
			if (n == n_intervals)
				svd_solver.compute(Pt, Eigen::ComputeThinU | Eigen::ComputeThinV);
			else
				svd_solver.compute(proj_D_l[s][n+1] * (proj_U_l[s][n+1] * b), Eigen::ComputeThinU | Eigen::ComputeThinV);
			if (n == n_intervals)
				proj_V_l[s][n] = svd_solver.matrixU();
			else
				proj_V_l[s][n] = proj_V_l[s][n+1] * svd_solver.matrixU();
			proj_D_l[s][n] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
			proj_U_l[s][n] = svd_solver.matrixV().adjoint();
		}

		// n = 0, ..., n_intervals - 1
		void stabilize_forward(int s, int n, const dmatrix_t& b)
		{
			if (use_projector)
			{
				/*
				qr_solver.compute((b * proj_U_r[s][n]) * proj_D_r[s][n]);
				dmatrix_t p_q = dmatrix_t::Identity(b.rows(), proj_U_r[s][n].cols());
				dmatrix_t p_r = dmatrix_t::Identity(proj_V_r[s][n].rows(), b.cols());
				dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				dmatrix_t d = qr_solver.matrixQR().diagonal().asDiagonal();
				proj_U_r[s][n+1] = qr_solver.matrixQ() * p_q;
				proj_D_r[s][n+1] = d;
				proj_V_r[s][n+1] = d.inverse() * (p_r * r) * qr_solver.colsPermutation().transpose()
					* proj_V_r[s][n];
				proj_W_r[s] = proj_U_r[s][n+1];
				proj_W_l[s] = proj_U_l[s][n+1];
				*/
				svd_solver.compute((b * proj_U_r[s][n]) * proj_D_r[s][n], Eigen::ComputeThinU | Eigen::ComputeThinV);
				proj_U_r[s][n+1] = svd_solver.matrixU();
				proj_D_r[s][n+1] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
				proj_V_r[s][n+1] = svd_solver.matrixV().adjoint() * proj_V_r[s][n];
				
				//proj_W_r[s] = proj_U_r[s][n+1] * proj_D_r[s][n+1] * proj_V_r[s][n+1];
				//proj_W_l[s] = proj_V_l[s][n+1] * proj_D_l[s][n+1] * proj_U_l[s][n+1];
				//proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				
				//std::cout << "stab forward, n = " << n << std::endl;
				//std::cout << "W_r" << std::endl;
				//print_matrix(proj_W_r[s]);
				proj_W_r[s] = proj_U_r[s][n+1];
				//print_matrix(proj_W_r[s]);
				//std::cout << "W_l" << std::endl;
				//print_matrix(proj_W_l[s]);
				proj_W_l[s] = proj_U_l[s][n+1];
				//print_matrix(proj_W_l[s]);
				//std::cout << "W" << std::endl;
				//print_matrix(proj_W[s]);
				proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				//print_matrix(proj_W[s]);
			}
			else
			{
				if (n == 0)
				{
					U[s][0] = id_N; D[s][0] = id_N; V[s][0] = id_N;
				}

				qr_solver.compute((b * U[s][n]) * D[s][n]);
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_l = U[s][n+1]; D_l = D[s][n+1]; V_l = V[s][n+1];
				U[s][n+1] = qr_solver.matrixQ();
				D[s][n+1] = qr_solver.matrixQR().diagonal().asDiagonal();
				V[s][n+1] = (D[s][n+1].inverse() * R) * (qr_solver.colsPermutation()
					.transpose() * V[s][n]);

				if (update_time_displaced_gf)
					recompute_time_displaced_gf(s, U_l, D_l, V_l, U[s][n+1], D[s][n+1],
						V[s][n+1]);
				else
					recompute_equal_time_gf(s, U_l, D_l, V_l, U[s][n+1], D[s][n+1],
						V[s][n+1]);
			}
			if (n == n_intervals - 1)
			{
				measure.add("norm_error", norm_error);
				norm_error = 0.;
				n_error = 0;
			}
		}

		//n = n_intervals, ..., 1
		void stabilize_backward(int s, int n, const dmatrix_t& b)
		{
			if (use_projector)
			{
				/*
				qr_solver.compute(proj_D_l[s][n] * (proj_U_l[s][n] * b));
				dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				proj_V_l[s][n-1] = proj_V_l[s][n] * qr_solver.matrixQ();
				proj_D_l[s][n-1] = qr_solver.matrixQR().diagonal().asDiagonal();
				proj_U_l[s][n-1] = proj_D_l[s][n-1].inverse() * r * qr_solver.colsPermutation().transpose();
				proj_W_r[s] = proj_U_r[s][n-1];
				proj_W_l[s] = proj_U_l[s][n-1];
				*/
				svd_solver.compute(proj_D_l[s][n] * (proj_U_l[s][n] * b), Eigen::ComputeThinU | Eigen::ComputeThinV);
				proj_V_l[s][n-1] = proj_V_l[s][n] * svd_solver.matrixU();
				proj_D_l[s][n-1] = svd_solver.singularValues().cast<complex_t>().asDiagonal();
				proj_U_l[s][n-1] = svd_solver.matrixV().adjoint();
				
				//std::cout << "stab backward, n = " << n << std::endl;
				//std::cout << "W_r" << std::endl;
				//print_matrix(proj_W_r[s]);
				proj_W_r[s] = proj_U_r[s][n-1];
				//print_matrix(proj_W_r[s]);
				//std::cout << "W_l" << std::endl;
				//print_matrix(proj_W_l[s]);
				proj_W_l[s] = proj_U_l[s][n-1];
				//print_matrix(proj_W_l[s]);
				//std::cout << "W" << std::endl;
				//print_matrix(proj_W[s]);
				proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				//print_matrix(proj_W[s]);
			}
			else
			{
				if (n == n_intervals)
				{
					U[s][n_intervals] = id_N;
					D[s][n_intervals] = id_N;
					V[s][n_intervals] = id_N;
				}

				qr_solver.compute(D[s][n] * (U[s][n] * b));
				dmatrix_t Q = qr_solver.matrixQ();
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_r = U[s][n-1]; D_r = D[s][n-1]; V_r = V[s][n-1];
				V[s][n-1] = V[s][n] * Q;
				D[s][n-1] = qr_solver.matrixQR().diagonal().asDiagonal();
				U[s][n-1] = D[s][n-1].inverse() * R * qr_solver.colsPermutation()
					.transpose();
				
				if (update_time_displaced_gf)
					recompute_time_displaced_gf(s, U[s][n-1], D[s][n-1], V[s][n-1], U_r,
						D_r, V_r);
				else
					recompute_equal_time_gf(s, U[s][n-1], D[s][n-1], V[s][n-1], U_r,
						D_r, V_r);
			}
			if (n == 1)
			{
				measure.add("norm_error", norm_error);
				norm_error = 0.;
				n_error = 0;
			}
		}

		void recompute_equal_time_gf(int s, const dmatrix_t& U_l_,
			const dmatrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const dmatrix_t& D_r_, const dmatrix_t& V_r_)
		{
			dmatrix_t old_gf = equal_time_gf[s];
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_U_r = U_r_.adjoint();

			qr_solver.compute(inv_U_r * inv_U_l + D_r_ * (V_r_ * V_l_) * D_l_);
			dmatrix_t invQ = qr_solver.matrixQ().adjoint();
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			equal_time_gf[s] = (inv_U_l * (qr_solver.colsPermutation()
				* R.inverse())) * (invQ * inv_U_r);

			if (init)
			{
				norm_error = (old_gf - equal_time_gf[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
//				measure.add("norm error", (old_gf - equal_time_gf).norm());
//				measure.add("max error", (old_gf - equal_time_gf).lpNorm<Eigen::
//					Infinity>());
//				measure.add("avg error", (old_gf - equal_time_gf).lpNorm<1>()
//					/ old_gf.rows() / old_gf.cols());
			}
		}

		void recompute_time_displaced_gf(int s, const dmatrix_t& U_l_,
			const dmatrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const dmatrix_t& D_r_, const dmatrix_t& V_r_)
		{
			int N = id_N.rows();
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_V_l = V_l_.adjoint();
			dmatrix_t inv_U_r = U_r_.adjoint();
			dmatrix_t inv_V_r = V_r_.inverse();

			dmatrix_t M(2 * N, 2 * N);
			M.topLeftCorner(N, N) = inv_V_l * inv_V_r;
			M.topRightCorner(N, N) = D_l_;
			M.bottomLeftCorner(N, N) = -D_r_;
			M.bottomRightCorner(N, N) = inv_U_r * inv_U_l;

			qr_solver.compute(M);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			dmatrix_t inv_V = qr_solver.colsPermutation() * R.inverse();
			dmatrix_t inv_U = qr_solver.matrixQ().adjoint();

			dmatrix_t lhs(2 * N, 2 * N);
			lhs.topLeftCorner(N, N) = inv_V_r * inv_V.topLeftCorner(N, N);
			lhs.topRightCorner(N, N) = inv_V_r * inv_V.topRightCorner(N, N);
			lhs.bottomLeftCorner(N, N) = inv_U_l * inv_V.bottomLeftCorner(N, N);
			lhs.bottomRightCorner(N, N) = inv_U_l * inv_V.bottomRightCorner(N, N);

			dmatrix_t rhs(2 * N, 2 * N);
			rhs.topLeftCorner(N, N) = inv_U.topLeftCorner(N, N) * inv_V_l;
			rhs.topRightCorner(N, N) = inv_U.topRightCorner(N, N) * inv_U_r;
			rhs.bottomLeftCorner(N, N) = inv_U.bottomLeftCorner(N, N) * inv_V_l;
			rhs.bottomRightCorner(N, N) = inv_U.bottomRightCorner(N, N) * inv_U_r;

			dmatrix_t old_td_gf = time_displaced_gf[s];
			if (sweep_direction == 1)
				time_displaced_gf[s] = lhs.bottomLeftCorner(N, N)
					* rhs.topLeftCorner(N, N) + lhs.bottomRightCorner(N, N)
					* rhs.bottomLeftCorner(N, N);
			else
				time_displaced_gf[s] = -lhs.topLeftCorner(N, N)
					* rhs.topRightCorner(N, N) - lhs.topRightCorner(N, N)
					* rhs.bottomRightCorner(N, N);

			dmatrix_t old_gf = equal_time_gf[s];
			equal_time_gf[s] = lhs.bottomLeftCorner(N, N) * rhs.topRightCorner(N, N)
				+ lhs.bottomRightCorner(N, N) * rhs.bottomRightCorner(N, N);
			if (init)
			{
				norm_error = (old_gf - equal_time_gf[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
				norm_error = (old_td_gf - time_displaced_gf[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
		}
	private:
		void print_matrix(const dmatrix_t& m)
		{
			Eigen::IOFormat clean(4, 0, ", ", "\n", "[", "]");
			std::cout << m.format(clean) << std::endl << std::endl;
		}
	private:
		measurements& measure;
		bool update_time_displaced_gf;
		int sweep_direction;
		int n_intervals;
		int n_species;
		std::vector<dmatrix_t>& equal_time_gf;
		std::vector<dmatrix_t>& time_displaced_gf;
		std::vector<dmatrix_t>& proj_W_l;
		std::vector<dmatrix_t>& proj_W_r;
		std::vector<dmatrix_t>& proj_W;
		dmatrix_t id_N;
		boost::multi_array<dmatrix_t, 2> U;
		boost::multi_array<dmatrix_t, 2> D;
		boost::multi_array<dmatrix_t, 2> V;
		boost::multi_array<dmatrix_t, 2> U_buffer;
		boost::multi_array<dmatrix_t, 2> D_buffer;
		boost::multi_array<dmatrix_t, 2> V_buffer;
		boost::multi_array<dmatrix_t, 2> proj_U_l;
		boost::multi_array<dmatrix_t, 2> proj_D_l;
		boost::multi_array<dmatrix_t, 2> proj_V_l;
		boost::multi_array<dmatrix_t, 2> proj_U_r;
		boost::multi_array<dmatrix_t, 2> proj_D_r;
		boost::multi_array<dmatrix_t, 2> proj_V_r;
		dmatrix_t U_l;
		dmatrix_t D_l;
		dmatrix_t V_l;
		dmatrix_t U_r;
		dmatrix_t D_r;
		dmatrix_t V_r;
		Eigen::ColPivHouseholderQR<dmatrix_t> qr_solver;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
		double norm_error = 0.;
		int n_error = 0;
		bool init = false;
		bool use_projector;
};
