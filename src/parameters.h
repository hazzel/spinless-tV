#pragma once
#include <complex>
#include <string>

struct parameters
{
	double beta, dtau, t, V, lambda;
	int n_delta, n_tau_slices, n_discrete_tau;
	std::string method;
	bool use_projector;
};