#pragma once
#include <ostream>
#include <vector>
#include <cmath>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

void eval_B_cdw(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = (*o[1])[0] / ((*o[0])[0] * (*o[0])[0]);
}

void eval_B_chern(double& out,
	std::vector<std::valarray<double>*>& o)
{
	out = (*o[1])[0] / ((*o[0])[0] * (*o[0])[0]);
}

void eval_epsilon(std::valarray<double>& out,
	std::vector<std::valarray<double>*>& o)
{
	std::valarray<double>* ep_tau = o[0];
	double epsilon = (*o[1])[0];
	out.resize(ep_tau->size());
	for (int i = 0; i < ep_tau->size(); ++i)
		out[i] = (*ep_tau)[i] - epsilon * epsilon;
}

void eval_n(double& out,
	std::vector<std::valarray<double>*>& o)
{
	double sign_re = (*o[0])[0];
	double sign_im = (*o[1])[0];
	double n_re = (*o[2])[0];
	double n_im = (*o[3])[0];
	out = (n_re * sign_re + n_im * sign_im)
		/ (sign_re * sign_re + sign_im * sign_im);
}

void eval_sign(double& out,
	std::vector<std::valarray<double>*>& o)
{
	double sign_re = (*o[0])[0];
	double sign_im = (*o[1])[0];
	out = std::sqrt(sign_re * sign_re + sign_im * sign_im);
}

struct measure_M
{
	configuration& config;
	parser& pars;

	void perform()
	{
		std::vector<double> c(config.l.max_distance() + 1, 0.);
		std::complex<double> energy = 0., m2 = 0., ep = 0., chern = 0.;
		std::complex<double> n = 0.;
		config.M.static_measure(c, n, energy, m2, ep, chern);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		if (config.param.mu != 0 || config.param.stag_mu != 0)
		{
			config.measure.add("n_re", std::real(n*config.param.sign_phase));
			config.measure.add("n_im", std::imag(n*config.param.sign_phase));
			config.measure.add("n", std::real(n));
		}
		config.measure.add("sign_phase_re", std::real(config.param.sign_phase));
		config.measure.add("sign_phase_im", std::imag(config.param.sign_phase));
		config.measure.add("energy", std::real(energy));
		config.measure.add("M2", std::real(m2));
		config.measure.add("epsilon", std::real(ep));
		config.measure.add("chern", std::real(chern));
		config.measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		config.measure.add_evalable("B_cdw", "M2", "M4", eval_B_cdw);
		config.measure.add_evalable("B_chern", "chern2", "chern4", eval_B_chern);
		
		if (config.param.n_discrete_tau > 0)
			for (int i = 0; i < config.param.obs.size(); ++i)
				if (config.param.obs[i] == "epsilon")
					config.measure.add_vectorevalable("dyn_epjack_tau", "dyn_epsilon_tau", "epsilon", eval_epsilon);
				
		if (config.param.mu != 0 || config.param.stag_mu != 0)
		{
			//config.measure.add_evalable("n_jack", "sign_phase_re", "sign_phase_im", "n_re", "n_im", eval_n);
		}
		config.measure.add_evalable("sign_jack", "sign_phase_re", "sign_phase_im", eval_sign);
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		config.measure.get_statistics(os);
	}
};
