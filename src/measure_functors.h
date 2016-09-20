#pragma once
#include <ostream>
#include <vector>
#include <cmath>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

struct measure_M
{
	configuration& config;
	measurements& measure;
	parser& pars;

	void perform()
	{
		std::vector<double> c(config.l.max_distance() + 1, 0.);
		double m2 = 0., ep = 0.;
		std::complex<double> n = 0.;
		config.M.static_measure(c, n, m2, ep);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		measure.add("sign_phase_re", std::real(config.sign_phase));
		measure.add("sign_phase_im", std::imag(config.sign_phase));
		measure.add("n_re", std::real(n*config.sign_phase));
		measure.add("n_im", std::imag(n*config.sign_phase));
		measure.add("M2", m2);
		measure.add("epsilon", ep);
		measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
