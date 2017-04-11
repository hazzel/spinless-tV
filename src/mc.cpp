#include <string>
#include <fstream>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include "mc.h"
#include "move_functors.h"
#include "measure_functors.h"
#include "event_functors.h"
#include "hex_honeycomb.h"
#include "honeycomb.h"
#ifdef PROFILER
	#include "gperftools/profiler.h"
#endif

mc::mc(const std::string& dir)
	: rng(Random()), qmc(rng), config(rng, measure)
{
	//Read parameters
	pars.read_file(dir);
	sweep = 0;
	int n_sweeps = pars.value_or_default<int>("SWEEPS", 0);
	n_static_cycles = pars.value_or_default<int>("static_cycles", 300);
	n_dyn_cycles = pars.value_or_default<int>("dyn_cycles", 300);
	n_warmup = pars.value_or_default<int>("warmup", 100000);
	n_prebin = pars.value_or_default<int>("prebin", 500);
	config.param.geometry = pars.value_or_default<std::string>("geometry", "rhom");
	config.param.L = pars.value_or_default<int>("L", 9);
	config.param.beta = 1./pars.value_or_default<double>("T", 0.2);
	config.param.n_tau_slices = pars.value_or_default<double>("tau_slices", 500);
	config.param.n_discrete_tau = pars.value_or_default<double>("discrete_tau",
		500);
 	config.param.n_discrete_tau = std::min(config.param.n_discrete_tau, config.param.n_tau_slices / 4);
	config.param.n_dyn_tau = pars.value_or_default<double>("dyn_tau",
		1);
	config.param.n_rebuild = pars.value_or_default<double>("rebuild", 10);
	config.param.dtau = config.param.beta / config.param.n_tau_slices;
	config.param.n_delta = pars.value_or_default<double>("stabilization", 10);
	config.param.t = pars.value_or_default<double>("t", 1.0);
	config.param.tprime = pars.value_or_default<double>("tprime", 0.0);
	config.param.V = pars.value_or_default<double>("V", 1.355);
	config.param.mu = pars.value_or_default<double>("mu", 0.);
	config.param.stag_mu = pars.value_or_default<double>("stag_mu", 0.);
	config.param.method = pars.value_or_default<std::string>("method", "finiteT");
	config.param.use_projector = (config.param.method == "projective");
	config.param.decoupling = pars.value_or_default<std::string>("decoupling", "majorana");
	config.param.lambda = std::acosh(std::exp(config.param.V * config.param.dtau / 2.));
	config.param.inv_symmetry = pars.value_or_default<int>("inv_symmetry", 1);
		
	std::string static_obs_string = pars.value_or_default<std::string>("static_obs", "M2");
	boost::split(config.param.static_obs, static_obs_string, boost::is_any_of(","));
	std::string obs_string = pars.value_or_default<std::string>("obs", "M2");
	boost::split(config.param.obs, obs_string, boost::is_any_of(","));
	if (pars.defined("seed"))
		rng.NewRng(pars.value_of<int>("seed"));

	//Initialize lattice
	if (config.param.geometry == "hex")
	{
		hex_honeycomb hc(config.param.L);
		config.l.generate_graph(hc);
		hc.generate_maps(config.l);
	}
	else
	{
		honeycomb hc(config.param.L);
		config.l.generate_graph(hc);
		hc.generate_maps(config.l);
	}
	
	qmc.add_measure(measure_M{config, pars}, "measurement");
	
	//Initialize configuration class
	config.initialize();
	
	//Set up events
	qmc.add_event(event_build{config, rng}, "initial build");
	qmc.add_event(event_flip_all{config, rng}, "flip all");
	qmc.add_event(event_static_measurement{config, rng, n_prebin, config.param.static_obs},
		"static_measure");
	qmc.add_event(event_dynamic_measurement{config, rng, n_prebin, config.param.obs},
		"dyn_measure");
	
	#ifdef PROFILER
		ProfilerStart("/net/home/lxtsfs1/tpc/hesselmann/code/profiler/gperftools.prof");
	#endif
}

mc::~mc()
{
	#ifdef PROFILER
		ProfilerStop();
	#endif
}

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

void mc::init()
{
	//Set up measurements
	n_prebin *= 5;
	config.measure.add_observable("norm_error", n_prebin);
	if (config.param.mu != 0 || config.param.stag_mu != 0)
	{
		config.measure.add_observable("n_re", n_prebin*100);
		config.measure.add_observable("n_im", n_prebin*100);
		config.measure.add_observable("n", n_prebin*100);
	}
	config.measure.add_observable("sign_phase_re", n_prebin*100);
	config.measure.add_observable("sign_phase_im", n_prebin*100);
	config.measure.add_observable("energy", n_prebin);
	config.measure.add_observable("h_t", n_prebin);
	config.measure.add_observable("h_v", n_prebin);
	config.measure.add_observable("h_mu", n_prebin);
	config.measure.add_observable("M2", n_prebin);
	config.measure.add_observable("S_cdw_q", n_prebin);
	config.measure.add_observable("M4", n_prebin);
	config.measure.add_observable("epsilon", n_prebin);
	config.measure.add_observable("kekule", n_prebin);
	config.measure.add_observable("chern", n_prebin);
	config.measure.add_observable("chern2", n_prebin);
	config.measure.add_observable("S_chern_q", n_prebin);
	config.measure.add_observable("chern4", n_prebin);
	config.measure.add_vectorobservable("corr", config.l.max_distance() + 1,
		n_prebin);
	n_prebin /= 5;
	
	if (config.param.n_discrete_tau > 0)
		for (int i = 0; i < config.param.obs.size(); ++i)
		{
			config.measure.add_vectorobservable("dyn_"+config.param.obs[i]+"_tau",
				config.param.n_discrete_tau + 1, n_prebin);
		}
	
	//Initialize vertex list to reduce warm up time
	qmc.trigger_event("initial build");
}

void mc::write(const std::string& dir)
{
	odump d(dir+"dump");
	random_write(d);
	d.write(sweep);
	d.write(static_bin_cnt);
	d.write(dyn_bin_cnt);
	config.serialize(d);
	d.close();
	seed_write(dir+"seed");
	std::ofstream f(dir+"bins");
	if (is_thermalized())
	{
		f << "Thermalization: Done." << std::endl
			<< "Sweeps: " << (sweep - n_warmup) << std::endl
			<< "Static bins: " << static_cast<int>(static_bin_cnt / n_prebin)
			<< std::endl
			<< "Dynamic bins: " << static_cast<int>(dyn_bin_cnt / n_prebin)
			<< std::endl;
	}
	else
	{
		f << "Thermalization: " << sweep << std::endl
			<< "Sweeps: 0" << std::endl
			<< "Static bins: 0" << std::endl
			<< "Dynamic bins: 0" << std::endl;
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
		d.read(static_bin_cnt);
		d.read(dyn_bin_cnt);
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
	if (sweep % config.param.n_rebuild == 0)
		config.M.rebuild();
	for (int i = 0; i < n_dyn_cycles; ++i)
	{
		for (int n = 0; n < config.M.get_max_tau(); ++n)
		{
			config.param.direction = -1;
			if (is_thermalized())
			{
				if (!config.param.use_projector || (config.param.use_projector
					&& std::abs(config.M.get_tau() - config.M.get_max_tau()/2) < config.M.get_max_tau()/16))
				{
					++measure_static_cnt;
					if (measure_static_cnt % n_static_cycles == 0)
					{
						++static_bin_cnt;
						//qmc.do_measurement();
						qmc.trigger_event("static_measure");
						measure_static_cnt = 0;
					}
				}
				if (config.param.n_discrete_tau > 0)
				if (config.param.use_projector && config.M.get_tau() == config.M.get_max_tau()/2
					+ config.param.n_discrete_tau * config.param.n_dyn_tau)
				{
					++measure_dyn_cnt;
					if (measure_dyn_cnt % n_dyn_cycles == n_dyn_cycles / 2)
					{
						++dyn_bin_cnt;
						qmc.trigger_event("dyn_measure");
					}
				}
			}
			qmc.trigger_event("flip all");
			config.M.stabilize_backward();
			//if (n % 5 == 0)
			//	std::cout << "tau = " << n << std::endl;
		}
		if (config.param.n_discrete_tau > 0)
		if (!config.param.use_projector && is_thermalized())
		{
			++measure_dyn_cnt;
			if (measure_dyn_cnt % n_dyn_cycles == n_dyn_cycles / 2)
			{
				++dyn_bin_cnt;
				qmc.trigger_event("dyn_measure");
			}
		}
		for (int n = 0; n < config.M.get_max_tau(); ++n)
		{
			config.param.direction = 1;
			qmc.trigger_event("flip all");
			config.M.stabilize_forward();
			
			if (is_thermalized())
			{
				if (!config.param.use_projector || (config.param.use_projector
					&& std::abs(config.M.get_tau() - config.M.get_max_tau()/2) < config.M.get_max_tau()/16))
				{
					++measure_static_cnt;
					if (measure_static_cnt % n_static_cycles == 0)
					{
						++static_bin_cnt;
						//qmc.do_measurement();
						qmc.trigger_event("static_measure");
						measure_static_cnt = 0;
					}
				}
				if (config.param.n_discrete_tau > 0)
				if (config.param.use_projector && config.M.get_tau() == config.M.get_max_tau()/2
					+ config.param.n_discrete_tau * config.param.n_dyn_tau)
				{
					++measure_dyn_cnt;
					if (measure_dyn_cnt % n_dyn_cycles == n_dyn_cycles / 2)
					{
						++dyn_bin_cnt;
						qmc.trigger_event("dyn_measure");
					}
				}
			}
			//if (n % 5 == 0)
			//	std::cout << "tau = " << n << std::endl;
		}
		if (config.param.n_discrete_tau > 0)
		if (!config.param.use_projector && is_thermalized())
		{
			++measure_dyn_cnt;
			if (measure_dyn_cnt % n_dyn_cycles == n_dyn_cycles / 2)
			{
				++dyn_bin_cnt;
				qmc.trigger_event("dyn_measure");
			}
		}
	}
	++sweep;
	status();
}

void mc::do_measurement()
{}

void mc::status()
{
	//if (sweep == n_warmup)
	//	std::cout << "Thermalization done." << std::endl;
	//if (is_thermalized() && sweep % (1000) == 0)
	//	std::cout << "sweep: " << sweep << std::endl;
}
