#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <iostream>

class move_base
{
	public:
		template<typename T>
		move_base(T&& functor, const std::string& name_, double prop_rate_=1.0)
			: name_str(name_), prop_rate(prop_rate_), avg_sign(1.0),
				n_attempted(0), n_accepted(0)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		//std::vector requires move constructor internally
		move_base(move_base&& rhs) { *this = std::move(rhs); }
		move_base& operator=(move_base&& rhs) = default;

		double attempt()
		{
			double p = attempt_fun();
			avg_sign *= static_cast<double>(n_attempted);
			avg_sign += (p >= 0.0) - (p < 0.0);
			++n_attempted;
			avg_sign /= static_cast<double>(n_attempted);
			return p;
		}
		double accept() { ++n_accepted; return accept_fun(); }
		void reject() { reject_fun(); }
		std::string name() { return name_str; }
		double proposal_rate() const { return prop_rate; }
		void proposal_rate(double prop_rate_) { prop_rate = prop_rate_; }
		double acceptance_rate() const { return static_cast<double>(n_accepted)
			/ static_cast<double>(n_attempted); }
		double sign() { return avg_sign; }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			attempt_fun = [functor]() { return functor->attempt(); };
			accept_fun = [functor]() { return functor->accept(); };
			reject_fun = [functor]() { functor->reject(); };
			clone_fun = [functor, this]() { return move_base(*functor, name_str,
				prop_rate); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<double()> attempt_fun;
		std::function<double()> accept_fun;
		std::function<void()> reject_fun;
		std::function<move_base()> clone_fun;
		std::string name_str;
		double prop_rate;
		double avg_sign;
		unsigned long n_attempted;
		unsigned long n_accepted;
};
