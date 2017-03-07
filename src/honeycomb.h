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
	std::vector<std::pair<int, int>> coord_map;
	// Base vectors of Bravais lattice
	Eigen::Vector2d a1;
	Eigen::Vector2d a2;
	// Base vectors of reciprocal lattice
	Eigen::Vector2d b1;
	Eigen::Vector2d b2;
	// Vector to second sublattice point
	Eigen::Vector2d delta;
	double pi = 4. * std::atan(1.);

	honeycomb(int L_ = 6)
		: L(L_),
			a1(3./2., std::sqrt(3.)/2.), a2(3./2., -std::sqrt(3.)/2.),
			delta(1./2., std::sqrt(3.)/2.)
	{
		b1 = Eigen::Vector2d(2.*pi/3., 2.*pi/std::sqrt(3.));
		b2 = Eigen::Vector2d(2.*pi/3., -2.*pi/std::sqrt(3.));
	}

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
				double c1 = static_cast<int>((i-1)  % (2*L)) / 2;
				double c2 = static_cast<int>((i-1)  / (2*L));
				real_space_map.push_back(Eigen::Vector2d{c1*a1 + c2*a2 + delta});
				coord_map.push_back({c1, c2});
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
				double c1 = static_cast<int>(i  % (2*L)) / 2;
				double c2 = static_cast<int>(i  / (2*L));
				real_space_map.push_back(Eigen::Vector2d{c1 * a1 + c2 * a2});
				coord_map.push_back({c1, c2});
			}
		}
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
		l.a1 = a1;
		l.a2 = a2;
		l.b1 = b1;
		l.b2 = b2;
		l.delta = delta;
		l.L = L;
	
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
			int N = l.n_sites();
			if (L == 2)
			{
				list = {{0, 1}, {1, 0}, {4, 7}, {7, 4}, {2, 5}, {5, 2}};
				return;
			}

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; j+=3)
				{
					int x0 = 2 * i + 2 * L * i;
					list.push_back({(x0 + 2*L*j) % N, (x0 + 2*L*j+1) % N});
					list.push_back({(x0 + 2*L*j+1) % N, (x0 + 2*L*j) % N});

					int x1 = 2 * i + 2 * L * i + 4*L;
					if (i == 0)
					{
						list.push_back({(x1 + 2*L*j) % N, (x1 + 2*L*j + 4*L-1) % N});
						list.push_back({(x1 + 2*L*j + 4*L-1) % N, (x1 + 2*L*j) % N});
					}
					else
					{
						list.push_back({(x1 + 2*L*j) % N, (x1 + 2*L*j + 2*L-1) % N});
						list.push_back({(x1 + 2*L*j + 2*L-1) % N, (x1 + 2*L*j) % N});
					}

					int x2 = 2 * i + 2 * L * i + 2*L;
					if (i == 0)
					{
						list.push_back({(x2 + 2*L*j) % N, (x2 + 2*L*j + 2*L-1) % N});
						list.push_back({(x2 + 2*L*j + 2*L-1) % N, (x2 + 2*L*j) % N});
					}
					else
					{
						list.push_back({(x2 + 2*L*j) % N, (x2 + 2*L*j - 1) % N});
						list.push_back({(x2 + 2*L*j - 1) % N, (x2 + 2*L*j) % N});
					}
				}
		});
		
		l.generate_bond_map("kekule_2", [&]
			(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (L == 2)
			{
				list = {{1, 2}, {2, 1}, {4, 5}, {5, 4}, {0, 7}, {7, 0}};
				return;
			}

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; j+=3)
				{
					int x0 = 2 * i + 2 * L * i;
					if (i == L - 1)
					{
						list.push_back({(x0+1 + 2*L*j) % N, (x0 + 2*L*j - 2*(L-1)) % N});
						list.push_back({(x0 + 2*L*j - 2*(L-1)) % N, (x0+1 + 2*L*j) % N});
					}
					else
					{
						list.push_back({(x0+1 + 2*L*j) % N, (x0+1 + 2*L*j+1) % N});
						list.push_back({(x0+1 + 2*L*j+1) % N, (x0+1 + 2*L*j) % N});
					}

					int x1 = 2 * i + 2 * L * i + 2*L;
					list.push_back({(x1 + 2*L*j) % N, (x1 + 2*L*j + 1) % N});
					list.push_back({(x1 + 2*L*j + 1) % N, (x1 + 2*L*j) % N});

					if (i == 0)
					{
						list.push_back({(x0 + 2*L*j) % N, (x0 + 2*L*j + 4*L-1) % N});
						list.push_back({(x0 + 2*L*j + 4*L-1) % N, (x0 + 2*L*j) % N});
					}
					else
					{
						list.push_back({(x0 + 2*L*j) % N, (x0 + 2*L*j + 2*L-1) % N});
						list.push_back({(x0 + 2*L*j + 2*L-1) % N, (x0 + 2*L*j) % N});
					}
				}
		});
		
		l.generate_bond_map("kekule_3", [&]
			(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (L == 2)
			{
				list = {{2, 3}, {3, 2}, {5, 6}, {6, 5}, {3, 4}, {4, 3}};
				return;
			}

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; j+=3)
				{
					int x0 = 2 * i + 2 * L * i;
					
					if (i == L - 1)
					{
						list.push_back({(x0+1 + 2*L*j + 2 - 2*L) % N, (x0 + 2*L*j - 2*(L-1)) % N});
						list.push_back({(x0 + 2*L*j - 2*(L-1)) % N, (x0+1 + 2*L*j + 2 - 2*L) % N});
					}
					else
					{
						list.push_back({(x0+1 + 2*L*j+2) % N, (x0+1 + 2*L*j+1) % N});
						list.push_back({(x0+1 + 2*L*j+1) % N, (x0+1 + 2*L*j+2) % N});
					}
					
					int x1 = 2 * i + 2 * L * i + 2*L;
					if (i == L - 1)
					{
						list.push_back({(x1 + 2*L*j + 2 - 2*L) % N, (x1 + 2*L*j + 1) % N});
						list.push_back({(x1 + 2*L*j + 1) % N, (x1 + 2*L*j + 2 - 2*L) % N});
					}
					else
					{
						list.push_back({(x1 + 2*L*j + 2) % N, (x1 + 2*L*j + 1) % N});
						list.push_back({(x1 + 2*L*j + 1) % N, (x1 + 2*L*j + 2) % N});
					}
					
					int x2 = 2 * i + 2 * L * i + 2 * L;
					if (i == 0)
					{
						list.push_back({(x2 + 2*L*j) % N, (x2 + 2*L*j + 4*L-1) % N});
						list.push_back({(x2 + 2*L*j + 4*L-1) % N, (x2 + 2*L*j) % N});
					}
					else
					{
						list.push_back({(x2 + 2*L*j) % N, (x2 + 2*L*j + 2*L-1) % N});
						list.push_back({(x2 + 2*L*j + 2*L-1) % N, (x2 + 2*L*j) % N});
					}
				}
		});
		
		l.generate_bond_map("chern", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (L == 2)
			{
				list = {{0, 4}, {4, 2}, {2, 0}, {2, 6}, {6, 4}, {0, 6}};
				return;
			}

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; ++j)
				{
					int x = 2 * i + 2 * L * j;
					int y = x + 2 * L;
					list.push_back({x % N, y % N});
	
					x = 2 * i + 2 * L * j;
					if (i == L - 1)
						y = x - 2 * (L - 1);
					else
						y = x + 2;
					list.push_back({(y+N) % N, x % N});
				
					x = 2 * i + 2 * L * j;
					if (i == 0)
						y = x + 4 * L - 2;
					else
						y = x + 2 * (L - 1);
					list.push_back({y % N, x % N});
				}
		});
		
		l.generate_bond_map("chern_2", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			if (L == 2)
			{
				list = {{1, 5}, {5, 3}, {3, 1}, {3, 7}, {7, 5}, {1, 7}};
				return;
			}

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; ++j)
				{
					int x = 2 * i + 1 + 2 * L * j;
					int y = x + 2 * L;
					list.push_back({y % N, x % N});
	
					x = 2 * i + 1 + 2 * L * j;
					if (i == L - 1)
						y = x - 2 * (L - 1);
					else
						y = x + 2;
					list.push_back({x % N, (y+N) % N});
				
					x = 2 * i + 1 + 2 * L * j;
					if (i == 0)
						y = x + 4 * L - 2;
					else
						y = x + 2 * (L - 1);
					list.push_back({x % N, y % N});
				}
		});
		
		/*
		l.generate_bond_map("chern_max_dist", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();
			for (auto& a : l.bonds("chern_1"))
			{
				
			}
		});
		*/
		
		l.generate_bond_map("nn_bond_1", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; ++j)
				{
					int x = 2 * i + 2 * L * j;
					int y = x + 1;
					list.push_back({y % N, x % N});
				}
		});
		
		l.generate_bond_map("nn_bond_2", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; ++j)
				{
					int x = 2 * i + 2 * L * j;
					int y;
					if (i == 0)
						y = x + 4 * L - 1;
					else
						y = x + 2 * L - 1;
					list.push_back({y % N, x % N});
				}
		});
		
		l.generate_bond_map("nn_bond_3", [&]
		(lattice::pair_vector_t& list)
		{
			int N = l.n_sites();

			for (int i = 0; i < L; ++i)
				for (int j = 0; j < L; ++j)
				{
					int x = 2 * i + 2 * L * j;
					int y;
					if (i == 0)
						y = x + 2 * L - 1;
					else
						y = x - 1;
					list.push_back({y % N, x % N});
				}
		});
	}
};
