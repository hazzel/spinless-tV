#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <string>
#include <map>
#include "Random.h"
#include "measurements.h"
#include "random.h"
#include "parser.h"
#include "types.h"
#include "dump.h"
#include "mctools.h"
#include "lattice.h"
#include "configuration.h"

class mc
{
	public:
		mc(const std::string& dir);
		~mc();

		void random_write(odump& d);
		void seed_write(const std::string& fn);
		void random_read(idump& d);
		void init();
		void write(const std::string& dir);
		bool read(const std::string& dir);
		void write_state(const std::string& dir);
		bool read_state(const std::string& dir);
		#ifdef MCL_PT
			void write_output(const std::string& dir, int para);
		#else
			void write_output(const std::string& dir);
		#endif
		bool is_thermalized();
		#ifdef MCL_PT
			vector<measurements> measure;
		#else
			measurements measure;
		#endif
		
		#ifdef MCL_PT
			bool request_global_update();
			void change_parameter(int);
			void change_to(int);
			double get_weight(int);
			int get_label();
		#endif
	public:
		void do_update();
		void do_measurement();
		void status();
	private:
		Random rng;
		mctools qmc;
		parser pars;
		int sweep;
		int n_static_cycles;
		int n_dyn_cycles;
		int n_warmup;
		int n_prebin;
		int n_tau_slices;
		
		int measure_dyn_cnt = 0;
		int measure_static_cnt = 0;
		int dyn_bin_cnt = 0;
		int static_bin_cnt = 0;

		configuration config;
};
