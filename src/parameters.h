#pragma once
#include <complex>
#include <string>

struct parameters
{
	double beta, dtau, t, tprime, V, mu, stag_mu, gamma, lambda;
	int L, n_delta, n_tau_slices, n_discrete_tau, n_dyn_tau, direction;
	std::string method, geometry, decoupling;
	bool use_projector;
	std::vector<std::string> obs, static_obs;
	std::complex<double> sign_phase=1.;
};
