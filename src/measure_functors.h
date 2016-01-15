#pragma once
#include <ostream>
#include <vector>
#include "measurements.h"
#include "parser.h"
#include "configuration.h"

struct measure_M
{
	configuration* config;
	measurements& measure;
	parser& pars;

	void perform()
	{
		std::vector<double> c = config->M.measure_M2();
		for (int i = 0; i < c.size()-1; ++i)
			c[i] /= config->shellsize[i];
		//measure.add("M2", config->M.measure_M2() / config->shellsize[1]);
		measure.add("M2", c.back());
		measure.add("corr", c);
	}

	void collect(std::ostream& os)
	{
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
