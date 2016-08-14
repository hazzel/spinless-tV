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
				proj_U_r.resize(boost::extents[n_species][n_intervals + 1]);
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
						U[s][n] = id_N;
						D[s][n] = diag_matrix_t(dimension);
						D[s][n].setIdentity();
						V[s][n] = id_N;
					}
			}
		}
		
		dmatrix_t stabilized_gf(int s, int n)
		{
			return id_N - proj_U_r[s][n] * (proj_U_l[s][n] * proj_U_r[s][n]).inverse() * proj_U_l[s][n];
		}

		void set(int s, int n, const dmatrix_t& b)
		{
			qr_solver.compute((b * U[s][n-1]) * D[s][n-1]);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			U[s][n] = qr_solver.matrixQ();
			D[s][n] = qr_solver.matrixQR().diagonal().asDiagonal();
			V[s][n] = R * (qr_solver.colsPermutation()
				.transpose() * V[s][n-1]);
			for (int i = 0; i < V[s][n].rows(); ++i)
				V[s][n].row(i) = 1./D[s][n].diagonal()[i] * V[s][n].row(i);
			if (n == n_intervals)
			{
				diag_matrix_t d_id(id_N.rows()); d_id.setIdentity();
				recompute_equal_time_gf(s, id_N, d_id, id_N, U[s][n_intervals],
					D[s][n_intervals], V[s][n_intervals]);
				if (s == n_species - 1)
					init = true;
				U[s][n_intervals] = id_N;
				D[s][n_intervals] = d_id;
				V[s][n_intervals] = id_N;
			}
		}
		
		void set_proj_l(int s, int n, const dmatrix_t& b, const dmatrix_t& Pt)
		{
			
			if (n == n_intervals)
				qr_solver.compute(Pt);
			else
				qr_solver.compute(proj_U_l[s][n+1] * b);
			dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			proj_U_l[s][n] = r * qr_solver.colsPermutation().transpose();
			for (int i = 0; i < proj_U_l[s][n].rows(); ++i)
				proj_U_l[s][n].row(i) = 1./qr_solver.matrixQR()(i, i) * proj_U_l[s][n].row(i);
			
			/*
			if (n == n_intervals)
				svd_solver.compute(Pt, Eigen::ComputeThinU | Eigen::ComputeThinV);
			else
				svd_solver.compute(proj_U_l[s][n+1] * b, Eigen::ComputeThinU | Eigen::ComputeThinV);
			proj_U_l[s][n] = svd_solver.matrixV().adjoint();
			*/
		}
		
		void set_proj_r(int s, int n, const dmatrix_t& b, const dmatrix_t& P)
		{
			
			if (n == 0)
				qr_solver.compute(P);
			else
				qr_solver.compute(b * proj_U_r[s][n-1]);
			dmatrix_t p_q = dmatrix_t::Identity(P.rows(), P.cols());
			proj_U_r[s][n] = qr_solver.matrixQ() * p_q;
			
			/*
			if (n == 0)
				svd_solver.compute(P, Eigen::ComputeThinU | Eigen::ComputeThinV);
			else
				svd_solver.compute(b * proj_U_r[s][n-1]);
			proj_U_r[s][n] = svd_solver.matrixU();
			*/
			
			//if (n == n_intervals)
			{
				//proj_W_r[s] = proj_U_r[s][n];
				//proj_W_l[s] = proj_U_l[s][n];
				//proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				equal_time_gf[s] = id_N - proj_U_r[s][n] * (proj_U_l[s][n]
					* proj_U_r[s][n]).inverse() * proj_U_l[s][n];
				
				std::cout << "g diag" << std::endl;
				std::cout << equal_time_gf[0].diagonal() << std::endl;
				Eigen::JacobiSVD<dmatrix_t> svd_solver(equal_time_gf[0]);
				std::cout << "g(0, 1) , g(1, 0)" << std::endl;
				std::cout << equal_time_gf[0](0, 1) << " "
					<< equal_time_gf[0](1, 0) << std::endl;
				std::cout << "g singular values" << std::endl;
				std::cout << svd_solver.singularValues() << std::endl;
				std::cout << "----" << std::endl << std::endl;
				
				if (s == n_species - 1)
					init = true;
			}
		}

		// n = 0, ..., n_intervals - 1
		void stabilize_forward(int s, int n, const dmatrix_t& b)
		{
			if (use_projector)
			{
				qr_solver.compute(b * proj_U_r[s][n]);
				dmatrix_t p_q = dmatrix_t::Identity(b.rows(), proj_U_r[s][n].cols());
				proj_U_r[s][n+1] = qr_solver.matrixQ() * p_q;
				
				/*
				svd_solver.compute(b * proj_U_r[s][n], Eigen::ComputeThinU | Eigen::ComputeThinV);
				proj_U_r[s][n+1] = svd_solver.matrixU();
				*/
				
				/*
				dmatrix_t old_gf = proj_W_r[s] * proj_W[s] * proj_W_l[s];
				proj_W_r[s] = proj_U_r[s][n+1];
				proj_W_l[s] = proj_U_l[s][n+1];
				proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				
				norm_error = (old_gf - proj_W_r[s] * proj_W[s] * proj_W_l[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
				*/
				
				//proj_W_r[s] = proj_U_r[s][n+1];
				//proj_W_l[s] = proj_U_l[s][n+1];
				//proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				dmatrix_t old_gf = equal_time_gf[s];
				equal_time_gf[s] = id_N - proj_U_r[s][n+1] * (proj_U_l[s][n+1] * proj_U_r[s][n+1]).inverse() * proj_U_l[s][n+1];
				norm_error = (old_gf - equal_time_gf[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
			else
			{
				if (n == 0)
				{
					U[s][0].setIdentity();
					D[s][0].setIdentity();
					V[s][0].setIdentity();
				}

				qr_solver.compute((b * U[s][n]) * D[s][n]);
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_l = U[s][n+1]; D_l = D[s][n+1]; V_l = V[s][n+1];
				U[s][n+1] = qr_solver.matrixQ();
				D[s][n+1] = qr_solver.matrixQR().diagonal().asDiagonal();
				V[s][n+1] = R * (qr_solver.colsPermutation()
					.transpose() * V[s][n]);
				for (int i = 0; i < V[s][n+1].rows(); ++i)
					V[s][n+1].row(i) = 1./D[s][n+1].diagonal()[i] * V[s][n+1].row(i);

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
				qr_solver.compute(proj_U_l[s][n] * b);
				dmatrix_t r = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				proj_U_l[s][n-1] = r * qr_solver.colsPermutation().transpose();
				for (int i = 0; i < proj_U_l[s][n-1].rows(); ++i)
					proj_U_l[s][n-1].row(i) = 1./qr_solver.matrixQR()(i, i) * proj_U_l[s][n-1].row(i);
				
				/*
				svd_solver.compute(proj_U_l[s][n] * b, Eigen::ComputeThinU | Eigen::ComputeThinV);
				proj_U_l[s][n-1] = svd_solver.matrixV().adjoint();
				*/
				
				/*
				dmatrix_t old_gf = proj_W_r[s] * proj_W[s] * proj_W_l[s];
				proj_W_r[s] = proj_U_r[s][n-1];
				proj_W_l[s] = proj_U_l[s][n-1];
				proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				
				norm_error = (old_gf - proj_W_r[s] * proj_W[s] * proj_W_l[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
				*/
				
				//proj_W_r[s] = proj_U_r[s][n-1];
				//proj_W_l[s] = proj_U_l[s][n-1];
				//proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
				dmatrix_t old_gf = equal_time_gf[s];
				equal_time_gf[s] = id_N - proj_U_r[s][n-1] * (proj_U_l[s][n-1] * proj_U_r[s][n-1]).inverse() * proj_U_l[s][n-1];
				norm_error = (old_gf - equal_time_gf[s]).norm() / (n_error + 1)
					+ n_error * norm_error / (n_error + 1);
				++n_error;
			}
			else
			{
				if (n == n_intervals)
				{
					U[s][n_intervals].setIdentity();
					D[s][n_intervals].setIdentity();
					V[s][n_intervals].setIdentity();
				}

				qr_solver.compute(D[s][n] * (U[s][n] * b));
				dmatrix_t Q = qr_solver.matrixQ();
				dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
				U_r = U[s][n-1]; D_r = D[s][n-1]; V_r = V[s][n-1];
				V[s][n-1] = V[s][n] * Q;
				D[s][n-1] = qr_solver.matrixQR().diagonal().asDiagonal();
				U[s][n-1] = R * qr_solver.colsPermutation().transpose();
				for (int i = 0; i < U[s][n-1].rows(); ++i)
					U[s][n-1].row(i) = 1./D[s][n-1].diagonal()[i] * U[s][n-1].row(i);
				
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
		
		void recompute_W(int s)
		{
			qr_solver.compute(proj_W_r[s]);
			dmatrix_t p_q = dmatrix_t::Identity(proj_W_r[s].rows(), proj_W_r[s].cols());
			proj_W_r[s] = qr_solver.matrixQ() * p_q;
			proj_W[s] = (proj_W_l[s] * proj_W_r[s]).inverse();
		}

		void recompute_equal_time_gf(int s, const dmatrix_t& U_l_,
			const diag_matrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const diag_matrix_t& D_r_, const dmatrix_t& V_r_)
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
		boost::multi_array<diag_matrix_t, 2> D;
		boost::multi_array<dmatrix_t, 2> V;
		boost::multi_array<dmatrix_t, 2> U_buffer;
		boost::multi_array<diag_matrix_t, 2> D_buffer;
		boost::multi_array<dmatrix_t, 2> V_buffer;
		boost::multi_array<dmatrix_t, 2> proj_U_l;
		boost::multi_array<dmatrix_t, 2> proj_U_r;
		dmatrix_t U_l;
		diag_matrix_t D_l;
		dmatrix_t V_l;
		dmatrix_t U_r;
		diag_matrix_t D_r;
		dmatrix_t V_r;
		Eigen::ColPivHouseholderQR<dmatrix_t> qr_solver;
		Eigen::JacobiSVD<dmatrix_t> svd_solver;
		double norm_error = 0.;
		int n_error = 0;
		bool init = false;
		bool use_projector;
};
