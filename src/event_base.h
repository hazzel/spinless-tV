#pragma once
#include <vector>
#include <functional>
#include <utility>
#include <memory>
#include <ostream>
#include <iostream>

class event_base
{
	public:
		template<typename T>
		event_base(T&& functor, const std::string& name_)
			: name_str(name_)
		{
			construct_delegation(new typename std::remove_reference<T>::type(
				std::forward<T>(functor)));
		}
		
		event_base(event_base&& rhs) {*this = std::move(rhs);}
		event_base& operator=(event_base&& rhs) = default;

		void trigger() { trigger_fun(); }
		std::string name() { return name_str; }
	private:
		template<typename T>
		void construct_delegation (T* functor)
		{
			impl = std::shared_ptr<T>(functor);
			trigger_fun = [functor]() { functor->trigger(); };
			clone_fun = [functor, this]() { return event_base(*functor,
				name_str); };
		}
	private:
		std::shared_ptr<void> impl;
		std::function<void()> trigger_fun;
		std::function<event_base()> clone_fun;
		std::string name_str;
};
