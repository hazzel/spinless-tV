#pragma once
#include <iostream>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "lattice.h"

struct honeycomb
{
	//typedef lattice::graph_t graph_t;
	typedef boost::adjacency_list<boost::setS, boost::vecS,
				boost::undirectedS> graph_t;

	int L;

	honeycomb(int L_ = 6) : L(L_) {}

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
			}
		}
	}
};
