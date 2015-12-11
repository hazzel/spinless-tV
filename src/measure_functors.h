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
		measure.add("M2", config->M.measure_M2() / config->shellsize[1]);
	}

	void collect(std::ostream& os)
	{
		std::cout << "collect" << std::endl;
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
