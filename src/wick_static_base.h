#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>

template<typename matrix_t>
class wick_static_base
{
	public:
		template<typename T>
		wick_static_base(T&& functor)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		wick_static_base(wick_static_base&& rhs) {*this = std::move(rhs);}
		wick_static_base& operator=(wick_static_base&& rhs) = default;
		wick_static_base(const wick_static_base& rhs) { std::cout << "copy c" << std::endl;}

		double get_obs(const matrix_t& et_gf) const
		{ return get_obs_fun(et_gf); }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			get_obs_fun = [functor](const matrix_t& et_gf)
				{ return functor->get_obs(et_gf); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<double(const matrix_t&)> get_obs_fun;
};
