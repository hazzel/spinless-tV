#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>

template<typename matrix_t>
class wick_base
{
	public:
		template<typename T>
		wick_base(T&& functor)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		wick_base(wick_base&& rhs) {*this = std::move(rhs);}
		wick_base& operator=(wick_base&& rhs) = default;
		wick_base(const wick_base& rhs) { std::cout << "copy c" << std::endl;}

		double get_obs(const matrix_t& et_gf_0, const matrix_t& et_gf_t,
			const matrix_t& time_displaced_gf) const
		{ return get_obs_fun(et_gf_0, et_gf_t, time_displaced_gf); }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			get_obs_fun = [functor](const matrix_t& et_gf_0, const matrix_t& et_gf_t,
				const matrix_t& td_gf)
				{ return functor->get_obs(et_gf_0, et_gf_t, td_gf); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<double(const matrix_t&, const matrix_t&,
			const matrix_t&)> get_obs_fun;
};
