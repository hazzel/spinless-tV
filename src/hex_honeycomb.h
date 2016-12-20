#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"


struct hex_honeycomb
{
	//typedef lattice::graph_t graph_t;
	typedef boost::adjacency_list<boost::setS, boost::vecS,
		boost::undirectedS> graph_t;

	int L;
	std::vector<Eigen::Vector2d> real_space_map;
	// Base vectors of Bravais lattice
	Eigen::Vector2d a1;
	Eigen::Vector2d a2;
	// Base vectors of reciprocal lattice
	Eigen::Vector2d b1;
	Eigen::Vector2d b2;
	// Vector to second sublattice point
	Eigen::Vector2d delta;
	double pi = 4. * std::atan(1.);

	hex_honeycomb(int L_ = 6)
		: L(L_),
			a1(3./2., std::sqrt(3.)/2.), a2(3./2., -std::sqrt(3.)/2.),
			delta(1./2., std::sqrt(3.)/2.)
	{
		b1 = Eigen::Vector2d(2.*pi/3., 2.*pi/std::sqrt(3.));
		b2 = Eigen::Vector2d(2.*pi/3., -2.*pi/std::sqrt(3.));
	}

	graph_t* graph()
	{
		int n_sites = 6;
		graph_t* g = new graph_t(n_sites);
		add_edges(g);
		return g;
	}

	void add_edges(graph_t* g)
	{
		typedef std::pair<int, int> edge_t;
		int n_vertices = boost::num_vertices(*g);
		
		boost::add_edge(0, 1, *g);
		boost::add_edge(0, 5, *g);
		real_space_map.push_back(Eigen::Vector2d{0., 0.});
		boost::add_edge(1, 0, *g);
		boost::add_edge(1, 2, *g);
		real_space_map.push_back(Eigen::Vector2d{delta});
		boost::add_edge(2, 1, *g);
		boost::add_edge(2, 3, *g);
		real_space_map.push_back(Eigen::Vector2d{a1});
		boost::add_edge(3, 2, *g);
		boost::add_edge(3, 4, *g);
		real_space_map.push_back(Eigen::Vector2d{a2 + delta});
		boost::add_edge(4, 3, *g);
		boost::add_edge(4, 5, *g);
		real_space_map.push_back(Eigen::Vector2d{a2});
		boost::add_edge(5, 4, *g);
		boost::add_edge(5, 0, *g);
		real_space_map.push_back(Eigen::Vector2d{a2 + delta - a1});
	}

	Eigen::Vector2d closest_k_point(const Eigen::Vector2d& K)
	{
		Eigen::Vector2d x = {0., 0.};
		double dist = (x - K).norm();
		for (int i = 0; i < L; ++i)
			for (int j = 0; j < L; ++j)
			{
				Eigen::Vector2d y = static_cast<double>(i) / static_cast<double>(L)
					* b1 + static_cast<double>(j) / static_cast<double>(L) * b2;
				double d = (y - K).norm();
				if (d < dist)
				{
					x = y;
					dist = d;
				}
			}
		return x;
	}

	void generate_maps(lattice& l)
	{
		//Symmetry points
		std::map<std::string, Eigen::Vector2d> points;

		points["K"] = closest_k_point({2.*pi/3., 2.*pi/3./std::sqrt(3.)});
		points["Kp"] = closest_k_point({2.*pi/3., -2.*pi/3./std::sqrt(3.)});
		points["Gamma"] = closest_k_point({0., 0.});
		points["M"] = closest_k_point({2.*pi/3., 0.});
		l.add_symmetry_points(points);

		//Site maps
		l.generate_neighbor_map("nearest neighbors", [&]
			(lattice::vertex_t i, lattice::vertex_t j) {
			return l.distance(i, j) == 1; });
		l.generate_bond_map("nearest neighbors", [&]
			(lattice::vertex_t i, lattice::vertex_t j)
			{ return l.distance(i, j) == 1; });
		l.generate_bond_map("d3_bonds", [&]
			(lattice::vertex_t i, lattice::vertex_t j)
			{ return l.distance(i, j) == 3; });
		l.generate_bond_map("kekule", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{0, 1}, {1, 0}, {2, 3}, {3, 2}, {4, 5}, {5, 4}};
		});
		
		l.generate_bond_map("kekule_2", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{1, 2}, {2, 1}, {3, 4}, {4, 3}, {5, 0}, {0, 5}};
		});
		
		l.generate_bond_map("chern", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{0, 4}, {4, 2}, {2, 0}};
		});
		
		l.generate_bond_map("chern_2", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{1, 5}, {5, 3}, {3, 1}};
		});
		
		l.generate_bond_map("nn_bond_1", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{1, 0}, {3, 4}};
		});
		
		l.generate_bond_map("nn_bond_2", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{5, 0}, {3, 2}};
		});
		
		l.generate_bond_map("nn_bond_3", [&]
			(lattice::pair_vector_t& list)
		{
			list = {{5, 4}, {1, 2}};
		});
	}
};
