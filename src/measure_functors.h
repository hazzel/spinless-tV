#pragma once
#include <ostream>
#include <vector>
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
		double m2 = 0.;
		config.M.static_measure(c, m2);
		for (int i = 0; i < c.size(); ++i)
			c[i] /= config.shellsize[i];
		measure.add("M2", m2);
		measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
