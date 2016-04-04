#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"

struct honeycomb
{
	//typedef lattice::graph_t graph_t;
	typedef boost::adjacency_list<boost::setS, boost::vecS,
				boost::undirectedS> graph_t;

	int L;
	std::vector<Eigen::Vector2d> real_space_map;
	// Base vectors of Bravais lattice
	Eigen::Vector2d a1;
	Eigen::Vector2d a2;
	// Vector to second sublattice point
	Eigen::Vector2d delta;

	honeycomb(int L_ = 6)
		: L(L_), a1(std::sqrt(3.), 0), a2(std::sqrt(3.)/2., 3./2.), delta(0., 1.)
	{}

	graph_t* graph()
	{
		int n_sites = 2 * L * L;
		graph_t* g = new graph_t(n_sites);
		add_edges(g);
		return g;
	}

	void add_edges(graph_t* g)
	{
		typedef std::pair<int, int> edge_t;
		int n_vertices = boost::num_vertices(*g);
		for (int i = 0; i < n_vertices; ++i)
		{
			if (i % 2 == 1)
			{
				if ((i+1) % (2*L) == 0)
				{
					boost::add_edge(i, (i - 4*L + 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i - 2*L + 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i - 1 + n_vertices) % n_vertices, *g);
				}
				else
				{
					boost::add_edge(i, (i - 2*L + 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i + 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i - 1 + n_vertices) % n_vertices, *g);
				}
				double c1 = static_cast<int>((i-1)  / (2*L));
				double c2 = static_cast<int>((i-1)  % (2*L)) / 2;
				real_space_map.push_back({c1 * a1 + c2 * a2 + delta});
			}
			else
			{
				if (i % (2*L) == 0)
				{
					boost::add_edge(i, (i + 4*L - 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i + 2*L - 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i + 1 + n_vertices) % n_vertices, *g);
				}
				else
				{
					boost::add_edge(i, (i + 2*L - 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i - 1 + n_vertices) % n_vertices, *g);
					boost::add_edge(i, (i + 1 + n_vertices) % n_vertices, *g);
				}
				double c1 = static_cast<int>(i  / (2*L));
				double c2 = static_cast<int>(i  % (2*L)) / 2;
				real_space_map.push_back({c1 * a1 + c2 * a2});
			}
		}
	}
};
