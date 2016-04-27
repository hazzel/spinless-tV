#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>
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

		qr_stabilizer(measurements& measure_, std::vector<dmatrix_t>& equal_time_gf_,
			std::vector<dmatrix_t>& time_displaced_gf_)
			: measure(measure_), update_time_displaced_gf(false),
			n_species(equal_time_gf.size()),
			equal_time_gf(equal_time_gf_), time_displaced_gf(time_displaced_gf_)
		{}

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
			U.resize(boost::extents[n_species][n_intervals + 1]);
			D.resize(boost::extents[n_species][n_intervals + 1]);
			V.resize(boost::extents[n_species][n_intervals + 1]);
			U_buffer.resize(boost::extents[n_species][n_intervals + 1]);
			D_buffer.resize(boost::extents[n_species][n_intervals + 1]);
			V_buffer.resize(boost::extents[n_species][n_intervals + 1]);
			for (int i = 0; i < n_species; ++i)
				for (int n = 0; n < n_intervals + 1; ++n)
				{
					U[i][n] = id_N; D[i][n] = id_N; V[i][n] = id_N;
				}
		}

		void set(int n, const dmatrix_t& b)
		{
			qr_solver.compute((b * U[n-1]) * D[n-1]);
			U[n] = qr_solver.matrixQ();
			D[n] = qr_solver.matrixQR().diagonal().asDiagonal();
			V[n] = (D[n].inverse() * qr_solver.matrixQR().triangularView<Eigen
				::Upper>()) * (qr_solver.colsPermutation().transpose() * V[n-1]);
			if (n == n_intervals)
			{
				recompute_equal_time_gf(id_N, id_N, id_N, U.back(), D.back(),
					V.back());
				U.back() = id_N; D.back() = id_N; V.back() = id_N;
			}
		}

		// n = 0, ..., n_intervals - 1
		void stabilize_forward(int n, const dmatrix_t& b)
		{
			if (n == 0)
			{
				U.front() = id_N; D.front() = id_N; V.front() = id_N;
			}

			qr_solver.compute((b * U[n]) * D[n]);
			U_l = U[n+1]; D_l = D[n+1]; V_l = V[n+1];
			U[n+1] = qr_solver.matrixQ();
			D[n+1] = qr_solver.matrixQR().diagonal().asDiagonal();
			V[n+1] = (D[n+1].inverse() * qr_solver.matrixQR().triangularView<Eigen
				::Upper>()) * (qr_solver.colsPermutation().transpose() * V[n]);

			if (update_time_displaced_gf)
				recompute_time_displaced_gf(U_l, D_l, V_l, U[n+1], D[n+1],
					V[n+1]);
			else
				recompute_equal_time_gf(U_l, D_l, V_l, U[n+1], D[n+1], V[n+1]);
		}

		//n = n_intervals, ..., 1
		void stabilize_backward(int n, const dmatrix_t& b)
		{
			if (n == n_intervals)
			{
				U.back() = id_N; D.back() = id_N; V.back() = id_N;
			}

			qr_solver.compute(D[n] * (U[n] * b));
			U_r = U[n-1]; D_r = D[n-1]; V_r = V[n-1];
			V[n-1] = V[n] * qr_solver.matrixQ();
			D[n-1] = qr_solver.matrixQR().diagonal().asDiagonal();
			U[n-1] = (D[n-1].inverse() * qr_solver.matrixQR().triangularView<Eigen
				::Upper>()) * qr_solver.colsPermutation().transpose();

			if (update_time_displaced_gf)
				recompute_time_displaced_gf(U[n-1], D[n-1], V[n-1], U_r, D_r,
					V_r);
			else
				recompute_equal_time_gf(U[n-1], D[n-1], V[n-1], U_r, D_r, V_r);
		}

		void recompute_equal_time_gf(const dmatrix_t& U_l_, const dmatrix_t& D_l_,
			const dmatrix_t& V_l_, const dmatrix_t& U_r_, const dmatrix_t& D_r_,
			const dmatrix_t& V_r_)
		{
			dmatrix_t old_gf = equal_time_gf;
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_U_r = U_r_.transpose();

			qr_solver.compute(inv_U_r * inv_U_l + D_r_ * (V_r_ * V_l_) * D_l_);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			dmatrix_t D = qr_solver.matrixQR().diagonal().asDiagonal();
			equal_time_gf = (inv_U_l * (qr_solver.colsPermutation()
				* R.inverse())) * (qr_solver.matrixQ().transpose() * inv_U_r);

			measure.add("norm error", (old_gf - equal_time_gf).norm());
			measure.add("max error", (old_gf - equal_time_gf).lpNorm<Eigen::
				Infinity>());
			measure.add("avg error", (old_gf - equal_time_gf).lpNorm<1>()
				/ old_gf.rows() / old_gf.cols());
		}

		void recompute_time_displaced_gf(const dmatrix_t& U_l_,
			const dmatrix_t& D_l_, const dmatrix_t& V_l_, const dmatrix_t& U_r_,
			const dmatrix_t& D_r_, const dmatrix_t& V_r_)
		{
			int N = id_N.rows();
			dmatrix_t inv_U_l = U_l_.inverse();
			dmatrix_t inv_V_l = V_l_.transpose();
			dmatrix_t inv_U_r = U_r_.transpose();
			dmatrix_t inv_V_r = V_r_.inverse();

			dmatrix_t M(2 * N, 2 * N);
			M.topLeftCorner(N, N) = inv_V_l * inv_V_r;
			M.topRightCorner(N, N) = D_l_;
			M.bottomLeftCorner(N, N) = -D_r_;
			M.bottomRightCorner(N, N) = inv_U_r * inv_U_l;

			qr_solver.compute(M);
			dmatrix_t R = qr_solver.matrixQR().triangularView<Eigen::Upper>();
			dmatrix_t inv_V = qr_solver.colsPermutation() * R.inverse();
			dmatrix_t inv_U = qr_solver.matrixQ().transpose();

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
			if ((old_gf - equal_time_gf).norm() > 0.0000001)
				std::cout << "error in stab: " << (old_gf - equal_time_gf).norm()
					<< std::endl;
			if ((old_td_gf - time_displaced_gf).norm() > 0.0000001)
				std::cout << "error in td stab: " << (old_td_gf - time_displaced_gf)
					.norm() << std::endl;
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
		dmatrix_t id_N;
		boost::multi_array<dmatrix_t, 2> U;
		boost::multi_array<dmatrix_t, 2> D;
		boost::multi_array<dmatrix_t, 2> V;
		boost::multi_array<dmatrix_t, 2> U_buffer;
		boost::multi_array<dmatrix_t, 2> D_buffer;
		boost::multi_array<dmatrix_t, 2> V_buffer;
		dmatrix_t U_l;
		dmatrix_t D_l;
		dmatrix_t V_l;
		dmatrix_t U_r;
		dmatrix_t D_r;
		dmatrix_t V_r;
		Eigen::ColPivHouseholderQR<dmatrix_t> qr_solver;
};
