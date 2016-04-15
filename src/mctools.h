#pragma once
#include <vector>
#include <map>
#include <functional>
#include <utility>
#include <algorithm>
#include <string>
#include <ostream>
#include <iostream>
#include <initializer_list>
#include "Random.h"
#include "measurements.h"
#include "move_base.h"
#include "event_base.h"
#include "measure_base.h"

class mctools
{
	public:
		mctools(Random& rng_) : rng(rng_) {}
		~mctools()
		{}

		template<typename T>
		void add_move(T&& functor, const std::string& name, double prop_rate=1.0)
		{
			moves.push_back(move_base(std::forward<T>(functor), name, prop_rate));
			normalize_proposal_rates();
			acceptance.push_back(std::make_pair(name, 0.0));
		}

		template<typename T>
		void add_event(T&& functor, const std::string& name)
		{
			events.emplace(std::make_pair(name, event_base{std::forward<T>(
				functor), name}));
		}

		template<typename T>
		void add_measure(T&& functor, const std::string& name)
		{
			measures.push_back(measure_base(std::forward<T>(functor), name));
		}

		void do_update(measurements& measure_sign)
		{
			double r = rng();
			for (int i = 0; i < moves.size(); ++i)
			{
				if (r < proposal[i])
				{
					double q = moves[i].attempt();
					if (q < -std::pow(10., -16.) && verbose)
						std::cout << "Negative sign at " << moves[i].name()
							<< " with value " << q << "." << std::endl;
					if (q != 0.0)
						measure_sign.add("sign", (q >= 0.0) - (q < 0.0));
					if (rng() < std::abs(q))
						moves[i].accept();
					else
						moves[i].reject();
					break;
				}
			}
		}

		void trigger_event(const std::string& name)
		{
			//map[] operators requires constructor() without arguments
			events.find(name)->second.trigger();
		}

		void do_measurement()
		{
			for (measure_base& m : measures)
				m.perform();
		}
		
		void collect_results(std::ostream& os)
		{
			for (measure_base& m : measures)
				m.collect(os);
		}

		const std::vector<std::pair<std::string, double>>& acceptance_rates()
		{
			for (int i = 0; i < moves.size(); ++i)
				acceptance[i].second = moves[i].acceptance_rate();
			return acceptance;
		}

		void set_proposal_rates(std::initializer_list<double> list)
		{
			if (list.size() != moves.size())
				return;
			for (int i = 0; i < list.size(); ++i)
				moves[i].proposal_rate(list.begin()[i]);
			normalize_proposal_rates();
		}

		double average_sign()
		{
			double sign = 0.0;
			for (int i = 0; i < moves.size(); ++i)
				sign += moves[i].sign();
			return sign / static_cast<double>(moves.size());
		}
	private:
		void normalize_proposal_rates()
		{
			proposal.assign(moves.size(), 0.0);
			double sum = 0.0;
			for (move_base& m : moves)
				sum += m.proposal_rate();
			proposal[0] = moves[0].proposal_rate() / sum;
			for (int i = 1; i < moves.size(); ++i)
				proposal[i] = proposal[i-1] + moves[i].proposal_rate() / sum;
		}
	private:
		Random& rng;
		std::vector<move_base> moves;
		std::map<std::string, event_base> events;
		std::vector<measure_base> measures;
		std::vector<double> proposal;
		std::vector<std::pair<std::string, double>> acceptance;
		bool verbose = true;
};
