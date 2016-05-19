#include <string>
#include <fstream>
#include <cmath>
#include "mc.h"
#include "move_functors.h"
#include "measure_functors.h"
#include "event_functors.h"

mc::mc(const std::string& dir)
	: rng(Random()), qmc(rng), config(measure)
{
	//Read parameters
	pars.read_file(dir);
	sweep = 0;
	measure_cnt = 0;
	n_cycles = pars.value_or_default<int>("cycles", 300);
	n_warmup = pars.value_or_default<int>("warmup", 100000);
	n_prebin = pars.value_or_default<int>("prebin", 500);
	hc.L = pars.value_or_default<int>("L", 9);
	config.param.beta = 1./pars.value_or_default<double>("T", 0.2);
	config.param.n_tau_slices = pars.value_or_default<double>("tau_slices", 500);
	config.param.dtau = config.param.beta / config.param.n_tau_slices;
	config.param.n_delta = pars.value_or_default<double>("stabilization", 10);
	config.param.t = pars.value_or_default<double>("t", 1.0);
	config.param.V = pars.value_or_default<double>("V", 1.355);
	config.param.lambda = std::acosh(std::exp(config.param.V*config.param.dtau
		/ 2.));
	if (pars.defined("seed"))
		rng.NewRng(pars.value_of<int>("seed"));

	//Initialize lattice
	config.l.generate_graph(hc);
	hc.generate_maps(config.l);

	//Set up measurements
	config.measure.add_observable("flip field", n_prebin * n_cycles);
	config.measure.add_observable("M2", n_prebin);
	config.measure.add_vectorobservable("corr", config.l.max_distance() + 1,
		n_prebin);
	config.measure.add_observable("norm error", n_prebin);
	config.measure.add_observable("max error", n_prebin);
	config.measure.add_observable("avg error", n_prebin);
	//measure.add_observable("sign", n_prebin * n_cycles);
	
	qmc.add_measure(measure_M{config, measure, pars}, "measurement");
	
	//Initialize configuration class
	config.initialize();
	
	//Set up events
	qmc.add_event(event_build{config, rng}, "initial build");
	qmc.add_event(event_flip_all{config, rng}, "flip all");

	//Initialize vertex list to reduce warm up time
	qmc.trigger_event("initial build");
}

mc::~mc()
{}

void mc::random_write(odump& d)
{
	rng.RngHandle()->write(d);
}
void mc::seed_write(const std::string& fn)
{
	std::ofstream s;
	s.open(fn.c_str());
	s << rng.Seed() << std::endl;
	s.close();
}
void mc::random_read(idump& d)
{
	rng.NewRng();
	rng.RngHandle()->read(d);
}
void mc::init() {}

void mc::write(const std::string& dir)
{
	odump d(dir+"dump");
	random_write(d);
	d.write(sweep);
	config.serialize(d);
	d.close();
	seed_write(dir+"seed");
	std::ofstream f(dir+"bins");
	if (is_thermalized())
	{
		f << "Thermalization: Done." << std::endl;
		f << "Sweeps: " << (sweep - n_warmup) << std::endl;
		f << "Bins: " << static_cast<int>((sweep - n_warmup) / n_prebin)
			<< std::endl;
	}
	else
	{
		f << "Thermalization: " << sweep << std::endl;
		f << "Sweeps: 0" << std::endl;
		f << "Bins: 0" << std::endl;
	}
	f.close();
}
bool mc::read(const std::string& dir)
{
	idump d(dir+"dump");
	if (!d)
	{
		std::cout << "read fail" << std::endl;
		return false;
	}
	else
	{
		random_read(d);
		d.read(sweep);
		config.serialize(d);
		d.close();
		return true;
	}
}

void mc::write_output(const std::string& dir)
{
	std::ofstream f(dir);
	qmc.collect_results(f);
	f.close();
	/*
	const std::vector<std::pair<std::string, double>>& acc =
		qmc.acceptance_rates();
	for (auto a : acc)
		std::cout << a.first << " : " << a.second << std::endl;
	std::cout << "Average sign: " << qmc.average_sign() << std::endl;
	*/
}

bool mc::is_thermalized()
{
	return sweep >= n_warmup;
}

void mc::do_update()
{
	if (!is_thermalized())
		double_sweep();
	else
		double_sweep();
	++sweep;
	status();
}

void mc::double_sweep()
{
	for (int n = 0; n < config.M.get_max_tau(); ++n)
	{
		qmc.trigger_event("flip all");
		if (is_thermalized())
		{
			if(measure_cnt == n_cycles)
			{
				qmc.do_measurement();
				measure_cnt = 0;
			}
			else
				++measure_cnt;
		}
		config.M.advance_backward();
		config.M.stabilize_backward();
	}
	for (int n = 0; n < config.M.get_max_tau(); ++n)
	{
		config.M.advance_forward();
		qmc.trigger_event("flip all");
		config.M.stabilize_forward();
		if (is_thermalized())
		{
			if(measure_cnt == n_cycles)
			{
				qmc.do_measurement();
				measure_cnt = 0;
			}
			else
				++measure_cnt;
		}
	}
}

void mc::do_measurement()
{
	//qmc.do_measurement();
}

void mc::status()
{
	if (sweep == n_warmup)
		std::cout << "Thermalization done." << std::endl;
	if (is_thermalized() && sweep % (1000) == 0)
		std::cout << "sweep: " << sweep << std::endl;
}
