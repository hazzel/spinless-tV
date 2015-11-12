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
	}

	void collect(std::ostream& os)
	{	
		os << "PARAMETERS" << std::endl;
		pars.get_all(os);
		measure.get_statistics(os);
	}
};
