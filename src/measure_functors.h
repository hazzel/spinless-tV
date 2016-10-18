#pragma once
#include <ostream>
#include <vector>
#include <cmath>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

void eval_epsilon(std::valarray<double>& out,
	std::vector<std::valarray<double>*>& o)
{
	std::valarray<double>* ep_tau = o[0];
	double epsilon=(*o[1])[0];
	out.resize(ep_tau->size());
	for (int i = 0; i < ep_tau->size(); ++i)
		out[i] = (*ep_tau)[i] - epsilon * epsilon;
}

struct measure_M
{
	configuration& config;
	parser& pars;

	void perform()
	{
		std::vector<double> c(config.l.max_distance() + 1, 0.);
		double m2 = 0., ep = 0.;
		std::complex<double> n = 0.;
		config.M.static_measure(c, n, m2, ep);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		if (config.param.mu != 0)
		{
			config.measure.add("sign_phase_re", std::real(config.sign_phase));
			config.measure.add("sign_phase_im", std::imag(config.sign_phase));
			config.measure.add("n_re", std::real(n*config.sign_phase));
			config.measure.add("n_im", std::imag(n*config.sign_phase));
			config.measure.add("n", std::real(n));
		}
		config.measure.add("M2", m2);
		config.measure.add("epsilon", ep);
		config.measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		if (config.param.n_discrete_tau > 0)
			for (int i = 0; i < config.param.obs.size(); ++i)
				if (config.param.obs[i] == "epsilon")
					config.measure.add_vectorevalable("dyn_epjack_tau", "dyn_epsilon_tau", "epsilon", eval_epsilon);
		
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		config.measure.get_statistics(os);
	}
};
