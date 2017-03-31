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
		using diag_matrix_t = Eigen::DiagonalMatrix<complex_t, Eigen::Dynamic>;

		qr_stabilizer(measurements& measure_,
			dmatrix_t& equal_time_gf_, dmatrix_t& time_displaced_gf_,
			dmatrix_t& proj_W_l_, dmatrix_t& proj_W_r_, dmatrix_t& proj_W_)
			: measure(measure_), update_time_displaced_gf(false),
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
		
		void set_buffer()
		{
			U_l_buffer = proj_U_l;
			U_r_buffer = proj_U_r;
		}
		void restore_buffer()
		{
			proj_U_l = U_l_buffer;
			proj_U_r = U_r_buffer;
		}

		void resize(int n_intervals_, int dimension)
		{
			id_N = dmatrix_t::Identity(dimension, dimension);
			n_intervals = n_intervals_;
			if (use_projector)
			{
				proj_U_l.resize(n_intervals + 1);
				proj_U_r.resize(n_intervals + 1);
				U_l_buffer.resize(n_intervals + 1);
				U_r_buffer.resize(n_intervals + 1);
			}
			else
			{
				U.resize(n_intervals + 1);
				D.resize(n_intervals + 1);
				V.resize(n_intervals + 1);
				U_buffer.resize(n_intervals + 1);
				D_buffer.resize(n_intervals + 1);
				V_buffer.resize(n_intervals + 1);
				for (int n = 0; n < n_intervals + 1; ++n)
				{
					U[n] = id_N;
					D[n] = diag_matrix_t(dimension);
					D[n].setIdentity();
					V[n] = id_N;
				}
			}
		}

		void set(int n, const dmatrix_t& b)
		{
			qr_solver.compute((b * U[n-1]) * D[n-1]);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			U[n] = qr_solver.matrixQ();
			D[n] = qr_solver.matrixQR().diagonal().asDiagonal();
			V[n] = R * (qr_solver.colsPermutation()
				.transpose() * V[n-1]);
			for (int i = 0; i < V[n].rows(); ++i)
				V[n].row(i) = 1./D[n].diagonal()[i] * V[n].row(i);
			if (n == n_intervals)
			{
				diag_matrix_t d_id(id_N.rows()); d_id.setIdentity();
				recompute_equal_time_gf(id_N, d_id, id_N, U[n_intervals],
					D[n_intervals], V[n_intervals]);
				init = true;
				U[n_intervals] = id_N;
				D[n_intervals] = d_id;
				V[n_intervals] = id_N;
			}
		}
		
		void set_P(const dmatrix_t& P, const dmatrix_t& Pt)
		{
			this->P = P;
			this->Pt = Pt;
		}
		
		void set_proj_l(int n, const dmatrix_t& b)
		{
			if (n == n_intervals)
				qr_solver.compute(Pt);
			else
				qr_solver.compute(proj_U_l[n+1] * b);
			dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			proj_U_l[n] = r * qr_solver.colsPermutation().transpose();
			for (int i = 0; i < proj_U_l[n].rows(); ++i)
				proj_U_l[n].row(i) = 1./qr_solver.matrixQR()(i, i) * proj_U_l[n].row(i);
		}
		
		void set_proj_r(int n, const dmatrix_t& b)
		{
			if (n == 0)
				qr_solver.compute(P);
			else
				qr_solver.compute(b * proj_U_r[n-1]);
			dmatrix_t p_q = dmatrix_t::Identity(P.rows(), P.cols());
			proj_U_r[n] = qr_solver.matrixQ() * p_q;
			
			if (n == n_intervals)
			{
				proj_W_r = proj_U_r[n];
				proj_W_l = proj_U_l[n];
				proj_W = (proj_W_l * proj_W_r).inverse();
				//equal_time_gf = id_N - proj_U_r[n] * (proj_U_l[n]
				//	* proj_U_r[n]).inverse() * proj_U_l[n];
				
				init = true;
				equal_time_gf = id_N - proj_W_r * proj_W * proj_W_l;
				std::cout << equal_time_gf << std::endl;
			}
		}

		// n = 0, ..., n_intervals - 1
		void stabilize_forward(int n, const dmatrix_t& b)
		{
			if (use_projector)
			{
				qr_solver.compute(b * proj_U_r[n]);
				dmatrix_t p_q = dmatrix_t::Identity(b.rows(), proj_U_r[n].cols());
				proj_U_r[n+1] = qr_solver.matrixQ() * p_q;
				
				dmatrix_t old_gf = id_N;
				old_gf.noalias() -= proj_W_r * proj_W * proj_W_l;
				proj_W_r = proj_U_r[n+1];
				proj_W_l = proj_U_l[n+1];
				proj_W = (proj_W_l * proj_W_r).inverse();
				equal_time_gf = id_N;
				equal_time_gf.noalias() -= proj_W_r * proj_W * proj_W_l;
				
				//dmatrix_t old_gf = equal_time_gf;
				//equal_time_gf = id_N - proj_U_r[n+1] * (proj_U_l[n+1] * proj_U_r[n+1]).inverse() * proj_U_l[n+1];
				
				double ne = (old_gf - equal_time_gf).norm();
				if (ne > std::pow(10., -6.))
					std::cout << "Norm error: " << ne << std::endl;
				norm_error = ne / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
			else
			{
				if (n == 0)
				{
					U[0].setIdentity();
					D[0].setIdentity();
					V[0].setIdentity();
				}

				qr_solver.compute((b * U[n]) * D[n]);
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_l = U[n+1]; D_l = D[n+1]; V_l = V[n+1];
				U[n+1] = qr_solver.matrixQ();
				D[n+1] = qr_solver.matrixQR().diagonal().asDiagonal();
				V[n+1] = R * (qr_solver.colsPermutation()
					.transpose() * V[n]);
				for (int i = 0; i < V[n+1].rows(); ++i)
					V[n+1].row(i) = 1./D[n+1].diagonal()[i] * V[n+1].row(i);

				if (update_time_displaced_gf)
					recompute_time_displaced_gf(U_l, D_l, V_l, U[n+1], D[n+1],
						V[n+1]);
				else
					recompute_equal_time_gf(U_l, D_l, V_l, U[n+1], D[n+1],
						V[n+1]);
			}
			if (n == n_intervals - 1)
			{
				measure.add("norm_error", norm_error);
				norm_error = 0.;
				n_error = 0;
			}
		}

		//n = n_intervals, ..., 1
		void stabilize_backward(int n, const dmatrix_t& b)
		{
			if (use_projector)
			{
				qr_solver.compute(proj_U_l[n] * b);
				dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				proj_U_l[n-1] = r * qr_solver.colsPermutation().transpose();
				for (int i = 0; i < proj_U_l[n-1].rows(); ++i)
					proj_U_l[n-1].row(i) = 1./qr_solver.matrixQR()(i, i) * proj_U_l[n-1].row(i);
				
				dmatrix_t old_gf = id_N;
				old_gf.noalias() -= proj_W_r * proj_W * proj_W_l;
				proj_W_r = proj_U_r[n-1];
				proj_W_l = proj_U_l[n-1];
				proj_W = (proj_W_l * proj_W_r).inverse();
				equal_time_gf = id_N;
				equal_time_gf.noalias() -= proj_W_r * proj_W * proj_W_l;
				
				//dmatrix_t old_gf = equal_time_gf;
				//equal_time_gf = id_N - proj_U_r[n-1] * (proj_U_l[n-1] * proj_U_r[n-1]).inverse() * proj_U_l[n-1];
				
				double ne = (old_gf - equal_time_gf).norm();
				if (ne > std::pow(10., -6.))
					std::cout << "Norm error: " << ne << std::endl;
				norm_error = ne / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
			else
			{
				if (n == n_intervals)
				{
					U[n_intervals].setIdentity();
					D[n_intervals].setIdentity();
					V[n_intervals].setIdentity();
				}

				qr_solver.compute(D[n] * (U[n] * b));
				dmatrix_t Q = qr_solver.matrixQ();
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_r = U[n-1]; D_r = D[n-1]; V_r = V[n-1];
				V[n-1] = V[n] * Q;
				D[n-1] = qr_solver.matrixQR().diagonal().asDiagonal();
				U[n-1] = R * qr_solver.colsPermutation().transpose();
				for (int i = 0; i < U[n-1].rows(); ++i)
					U[n-1].row(i) = 1./D[n-1].diagonal()[i] * U[n-1].row(i);
				
				if (update_time_displaced_gf)
					recompute_time_displaced_gf(U[n-1], D[n-1], V[n-1], U_r,
						D_r, V_r);
				else
					recompute_equal_time_gf(U[n-1], D[n-1], V[n-1], U_r,
						D_r, V_r);
			}
			if (n == 1)
			{
				measure.add("norm_error", norm_error);
				norm_error = 0.;
				n_error = 0;
			}
		}

		void recompute_equal_time_gf(const dmatrix_t& U_l_,
			const diag_matrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const diag_matrix_t& D_r_, const dmatrix_t& V_r_)
		{
			dmatrix_t old_gf = equal_time_gf;
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_U_r = U_r_.adjoint();

			qr_solver.compute(inv_U_r * inv_U_l + D_r_ * (V_r_ * V_l_) * D_l_);
			dmatrix_t invQ = qr_solver.matrixQ().adjoint();
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			equal_time_gf = (inv_U_l * (qr_solver.colsPermutation()
				* R.inverse())) * (invQ * inv_U_r);

			if (init)
			{
				double ne = (old_gf - equal_time_gf).norm();
				if (ne > std::pow(10., -6.))
					std::cout << "Norm error: " << ne << std::endl;

				norm_error = ne / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
		}

		void recompute_time_displaced_gf(const dmatrix_t& U_l_,
			const diag_matrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const diag_matrix_t& D_r_, const dmatrix_t& V_r_)
		{
			int N = id_N.rows();
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_V_l = V_l_.adjoint();
			dmatrix_t inv_U_r = U_r_.adjoint();
			dmatrix_t inv_V_r = V_r_.inverse();

			dmatrix_t M(2 * N, 2 * N);
			M.topLeftCorner(N, N) = inv_V_l * inv_V_r;
			M.topRightCorner(N, N) = dmatrix_t(D_l_);
			M.bottomLeftCorner(N, N) = -dmatrix_t(D_r_);
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

			dmatrix_t old_td_gf = time_displaced_gf;
			if (sweep_direction == 1)
				time_displaced_gf = lhs.bottomLeftCorner(N, N)
					* rhs.topLeftCorner(N, N) + lhs.bottomRightCorner(N, N)
					* rhs.bottomLeftCorner(N, N);
			else
				time_displaced_gf = -lhs.topLeftCorner(N, N)
					* rhs.topRightCorner(N, N) - lhs.topRightCorner(N, N)
					* rhs.bottomRightCorner(N, N);

			dmatrix_t old_gf = equal_time_gf;
			equal_time_gf = lhs.bottomLeftCorner(N, N) * rhs.topRightCorner(N, N)
				+ lhs.bottomRightCorner(N, N) * rhs.bottomRightCorner(N, N);
			if (init)
			{			
				double ne = (old_gf - equal_time_gf).norm();
				if (ne > std::pow(10., -6.))
					std::cout << "Norm error: " << ne << std::endl;

				norm_error = ne / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
				ne = (old_td_gf - time_displaced_gf).norm();
				if (ne > std::pow(10., -6.))
					std::cout << "Norm error: " << ne << std::endl;

				norm_error = ne / (n_error + 1)
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
		dmatrix_t& equal_time_gf;
		dmatrix_t& time_displaced_gf;
		dmatrix_t& proj_W_l;
		dmatrix_t& proj_W_r;
		dmatrix_t& proj_W;
		dmatrix_t id_N;
		//Finite T
		std::vector<dmatrix_t> U;
		std::vector<diag_matrix_t> D;
		std::vector<dmatrix_t> V;
		std::vector<dmatrix_t> U_buffer;
		std::vector<diag_matrix_t> D_buffer;
		std::vector<dmatrix_t> V_buffer;
		//Projector
		std::vector<dmatrix_t> proj_U_l;
		std::vector<dmatrix_t> proj_U_r;
		std::vector<dmatrix_t> U_l_buffer;
		std::vector<dmatrix_t> U_r_buffer;
		dmatrix_t U_l;
		diag_matrix_t D_l;
		dmatrix_t V_l;
		dmatrix_t U_r;
		diag_matrix_t D_r;
		dmatrix_t V_r;
		dmatrix_t P;
		dmatrix_t Pt;
		Eigen::ColPivHouseholderQR<dmatrix_t> qr_solver;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
		double norm_error = 0.;
		int n_error = 0;
		bool init = false;
		bool use_projector;
};
