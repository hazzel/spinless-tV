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
		double n = 0., m2 = 0., ep = 0., kek = 0., chern = 0.;
		config.M.static_measure(c, n, m2, ep, kek, chern);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		measure.add("sign_phase", std::fmod(config.sign_phase, 2.*std::atan(1)));
		measure.add("n", n);
		measure.add("M2", m2);
		measure.add("epsilon", ep);
		measure.add("kekule", kek);
		measure.add("chern", chern);
		measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
