#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <numeric>
#include <queue>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <pthread.h>
#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace boost::heap;

size_t n, m, n0, m0, k, l, T, d, B, threads, steps_est;
double epsilon1, epsilon2, delta1, delta2, R, p, elapsed;
unordered_set<size_t> nodes, seeds_return;
unordered_map<size_t,vector<double>> features;
vector<unordered_map<size_t,unordered_map<size_t,double>>> probs;
vector<bool> kreach, flag;
vector<double> score;
vector<vector<double>> hp;
unordered_map<size_t,unordered_set<size_t>> rev_graph;
vector<unordered_map<size_t,unordered_set<size_t>>> h_est, h_cv;
vector<unordered_set<size_t>> seeds_current, max_seeds;
vector<vector<unordered_set<size_t>>> seeds_current_cover, max_cover;
vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>> node_cover;
vector<unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>> rr_est, rr_cv;
unordered_map<size_t,size_t> seed_threads;
pthread_mutex_t m_lock;
template<>
struct std::hash<pair<size_t,size_t>>
{
	size_t operator()(const pair<size_t,size_t>& edge) const noexcept
	{
		size_t seed = 0;
		boost::hash_combine(seed, edge.first);
		boost::hash_combine(seed, edge.second);
		return seed;
	}
};
vector<unordered_set<pair<size_t,size_t>>> change;
unordered_map<pair<size_t,size_t>,unordered_set<size_t>> pres;
struct node
{
	size_t vertex;
	double gain;
	node(const size_t& v, double g) : vertex(v), gain(g) {}
};
struct compare_node
{
	bool operator()(const node& n1, const node& n2) const noexcept
	{
		return n1.gain > n2.gain;
	}
};
using handle_t = fibonacci_heap<node,compare<compare_node>>::handle_type;

double prob(size_t u, size_t v, size_t i)
{
	if (probs[i].find(u) == probs[i].end())
		probs[i][u] = unordered_map<size_t,double>();
	if (probs[i][u].find(v) == probs[i][u].end())
	{
		vector<double> x = vector<double>();
		x.insert(x.end(), features[u].begin(), features[u].end());
		x.insert(x.end(), features[v].begin(), features[v].end());
		double s = 0;
		for (size_t j = 0; j < d; j++)
			s += (hp[i][j] * x[j]);
		probs[i][u][v] = 1 / (1 + exp(-s));
	}
	return probs[i][u][v];
}

pair<unordered_map<size_t,unordered_set<size_t>>,size_t> gen_rrset(size_t v, size_t i, unsigned int& seed)
{
	unordered_map<size_t,unordered_set<size_t>> rr = unordered_map<size_t,unordered_set<size_t>>();
	unordered_set<size_t> visited = unordered_set<size_t>();
	size_t num = 0;
	queue<size_t> q = queue<size_t>();
	q.push(v);
	visited.insert(v);
	while (not q.empty())
	{
		size_t u = q.front();
		q.pop();
		rr[u] = unordered_set<size_t>();
		for (size_t x : rev_graph[u])
		{
			num++;
			if (visited.find(x) == visited.end() and (double)rand_r(&seed) / RAND_MAX < prob(x, u, i))
			{
				visited.insert(x);
				rr[u].insert(x);
				q.push(x);
			}
		}
	}
	return make_pair(rr, num);
}

void* greedy_insert_helper(void* args)
{
	auto& [j, u, w] = *((tuple<size_t,size_t,vector<double>>*)args);
	max_seeds[j] = seeds_current[j];
	max_cover[j] = seeds_current_cover[j];
	score[j] = 0;
	for (size_t i = 0; i < l; i++)
		score[j] += (w[i] * max_cover[j][i].size());
	if (max_seeds[j].size() < k and max_seeds[j].find(u) == max_seeds[j].end())
	{
		fibonacci_heap<node,compare<compare_node>> heap = fibonacci_heap<node,compare<compare_node>>();
		unordered_map<size_t,handle_t> handles = unordered_map<size_t,handle_t>();
		size_t vertex = u;
		double gain = 0;
		for (auto& e : node_cover[j])
		{
			double g = 0;
			for (auto& f : e.second)
				g += (w[f.first] * f.second.size());
			if (e.first == u)
				gain = g;
			else
				handles[e.first] = heap.push(node(e.first, g));
		}
		while (gain >= (pow(1 + epsilon1, j) - score[j]) / k and max_seeds[j].size() < k)
		{
			max_seeds[j].insert(vertex);
			score[j] += gain;
			handles.erase(vertex);
			if (heap.empty() or max_seeds[j].size() == k)
				break;
			if (node_cover[j].find(vertex) != node_cover[j].end())
			{
				for (auto& e : node_cover[j])
					if (handles.find(e.first) != handles.end())
					{
						double diff = 0;
						for (auto& f : e.second)
						{
							size_t num = 0, i = f.first;
							for (auto& x : f.second)
								if (node_cover[j][vertex].find(i) != node_cover[j][vertex].end() and node_cover[j][vertex][i].find(x) != node_cover[j][vertex][i].end() and max_cover[j][i].find(x) == max_cover[j][i].end())
									num++;
							diff += (w[i] * num);
						}
						heap.update(handles[e.first], node(e.first, (*handles[e.first]).gain - diff));
					}
				for (auto& f : node_cover[j][vertex])
					max_cover[j][f.first].insert(f.second.begin(), f.second.end());
			}
			node nd = heap.top();
			heap.pop();
			vertex = nd.vertex;
			gain = nd.gain;
		}
	}
	return NULL;
}

auto greedy_insert(pair<size_t,size_t>& edge, vector<double>& w)
{
	size_t u = edge.first, v = edge.second;
	vector<tuple<size_t,size_t,vector<double>>> t = vector<tuple<size_t,size_t,vector<double>>>(threads);
	vector<pthread_t> tid = vector<pthread_t>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		t[j] = make_tuple(j, u, w);
		int rc = pthread_create(&tid[j], NULL, &greedy_insert_helper, (void*)(&t[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	// return make_tuple(max_seeds, max_cover, pos);
	return distance(score.begin(), max_element(score.begin(), score.end()));
}

// auto greedy_insert(pair<size_t,size_t>& edge, vector<double>& w)
// {
// 	double mst = 0;
// 	size_t pos = 0, u = edge.first, v = edge.second;
// 	vector<unordered_set<size_t>> max_seeds = vector<unordered_set<size_t>>(threads);
// 	vector<vector<unordered_set<size_t>>> max_cover = vector<vector<unordered_set<size_t>>>(threads);
// 	for (size_t j = 0; j < threads; j++)
// 	{
// 		max_seeds[j] = seeds_current[j];
// 		max_cover[j] = seeds_current_cover[j];
// 		double score = 0;
// 		for (size_t i = 0; i < l; i++)
// 			score += (w[i] * seeds_current_cover[j][i].size());
// 		if (seeds_current[j].size() < k and seeds_current[j].find(u) == seeds_current[j].end())
// 		{
// 			fibonacci_heap<node,compare<compare_node>> heap = fibonacci_heap<node,compare<compare_node>>();
// 			unordered_map<size_t,handle_t> handles = unordered_map<size_t,handle_t>();
// 			size_t vertex = u;
// 			double gain = 0;
// 			for (auto& e : node_cover[j])
// 			{
// 				double g = 0;
// 				for (auto& f : e.second)
// 					g += (w[f.first] * f.second.size());
// 				if (e.first == u)
// 					gain = g;
// 				else
// 					handles[e.first] = heap.push(node(e.first, g));
// 			}
// 			while (gain >= (pow(1 + epsilon1, j) - score) / k and max_seeds[j].size() < k)
// 			{
// 				max_seeds[j].insert(vertex);
// 				score += gain;
// 				handles.erase(vertex);
// 				if (heap.empty() or max_seeds[j].size() == k)
// 					break;
// 				if (node_cover[j].find(vertex) != node_cover[j].end())
// 				{
// 					for (auto& e : node_cover[j])
// 						if (handles.find(e.first) != handles.end())
// 						{
// 							double diff = 0;
// 							for (auto& f : e.second)
// 							{
// 								size_t num = 0, i = f.first;
// 								for (auto& x : f.second)
// 									if (node_cover[j][vertex].find(i) != node_cover[j][vertex].end() and node_cover[j][vertex][i].find(x) != node_cover[j][vertex][i].end() and max_cover[j][i].find(x) == max_cover[j][i].end())
// 										num++;
// 								diff += (w[i] * num);
// 							}
// 							heap.update(handles[e.first], node(e.first, (*handles[e.first]).gain - diff));
// 						}
// 					for (auto& f : node_cover[j][vertex])
// 						max_cover[j][f.first].insert(f.second.begin(), f.second.end());
// 				}
// 				node nd = heap.top();
// 				heap.pop();
// 				vertex = nd.vertex;
// 				gain = nd.gain;
// 			}
// 		}
// 		if (score > mst)
// 		{
// 			mst = score;
// 			pos = j;
// 		}
// 	}
// 	return make_tuple(max_seeds, max_cover, pos);
// }

void* insert_edge_cov_helper(void* args)
{
	auto& [j, edge] = *((pair<size_t,pair<size_t,size_t>>*)args);
	flag[j] = true;
	size_t u = edge.first, v = edge.second;
	if (seeds_current[j].size() < k)
	{
		if (seeds_current[j].find(u) == seeds_current[j].end())
		{
			flag[j] = false;
			for (size_t h : pres[edge])
				if (seeds_current_cover[j][h].find(v) == seeds_current_cover[j][h].end())
				{
					if (node_cover[j].find(u) == node_cover[j].end())
						node_cover[j][u] = unordered_map<size_t,unordered_set<size_t>>();
					if (node_cover[j][u].find(h) == node_cover[j][u].end())
						node_cover[j][u][h] = unordered_set<size_t>();
					node_cover[j][u][h].insert(v);
				}
		}
		else
		{
			for (size_t h : pres[edge])
				if (seeds_current_cover[j][h].find(v) == seeds_current_cover[j][h].end())
				{
					seeds_current_cover[j][h].insert(v);
					for (auto& e : node_cover[j])
						if (e.second.find(h) != e.second.end())
							e.second[h].erase(v);
				}
		}
	}
	return NULL;
}

void* update_insert_helper(void* args)
{
	size_t j = (size_t)args;
	for (auto& e : node_cover[j])
		for (auto& f : e.second)
			for (auto& x : seeds_current_cover[j][f.first])
				f.second.erase(x);
	for (size_t x : seeds_current[j])
	{
		node_cover[j].erase(x);
		pthread_mutex_lock(&m_lock);
		if (seed_threads.find(x) == seed_threads.end())
			seed_threads[x] = 0;
		seed_threads[x]++;
		pthread_mutex_unlock(&m_lock);
	}
	kreach[j] = seeds_current[j].size() >= k;
	return NULL;
}

void insert_edge_cov(pair<size_t,size_t>& edge)
{
	vector<pthread_t> tid = vector<pthread_t>(threads);
	vector<pair<size_t,pair<size_t,size_t>>> t = vector<pair<size_t,pair<size_t,size_t>>>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		t[j] = make_pair(j, edge);
		int rc = pthread_create(&tid[j], NULL, &insert_edge_cov_helper, (void*)(&t[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	if (count(kreach.begin(), kreach.end(), true) >= threads / 2 or all_of(flag.begin(), flag.end(), [](bool i) { return i; }))
		return;
	seeds_return = unordered_set<size_t>();
	double eta = sqrt(log(l) / 2 / T);
	vector<double> sum = vector<double>(l, 0), w = vector<double>(l, 1.0 / l);
	vector<unordered_set<size_t>> seeds = vector<unordered_set<size_t>>(T), seeds_tmp;
	vector<vector<unordered_set<size_t>>> cover_tmp;
	for (size_t j = 0; j < T; j++)
	{
		auto res = greedy_insert(edge, w);
		// seeds[j] = get<0>(res)[get<2>(res)];
		seeds[j] = max_seeds[res];
		double s = 0;
		for (size_t i = 0; i < l; i++)
		{
			// sum[i] += get<1>(res)[get<2>(res)].size();
			sum[i] += max_cover[res].size();
			double wt = exp(- eta * sum[i]);
			w[i] = wt;
			s += wt;
		}
		for (size_t i = 0; i < l; i++)
			w[i] /= s;
		// for (auto& x : get<0>(res)[get<2>(res)])
		// 	seeds_return.insert(x);
		if (j == 0)
		{
			// seeds_tmp = get<0>(res);
			// cover_tmp = get<1>(res);
			seeds_tmp = max_seeds;
			cover_tmp = max_cover;
		}
	}
	seeds_return = seeds[rand() % T];
	seeds_current = seeds_tmp;
	seeds_current_cover = cover_tmp;
	seed_threads.clear();
	vector<size_t> tt = vector<size_t>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		tt[j] = j;
		int rc = pthread_create(&tid[j], NULL, &update_insert_helper, (void*)(tt[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	// for (size_t j = 0; j < threads; j++)
	// {
	// 	for (auto& e : node_cover[j])
	// 		for (auto& f : e.second)
	// 			for (auto& x : seeds_current_cover[j][f.first])
	// 				f.second.erase(x);
	// 	for (size_t x : seeds_current[j])
	// 	{
	// 		node_cover[j].erase(x);
	// 		if (seed_threads.find(x) == seed_threads.end())
	// 			seed_threads[x] = 0;
	// 		seed_threads[x]++;
	// 	}
	// 	if (seeds_current[j].size() < k)
	// 		num_k--;
	// }
}

void* greedy_remove_helper(void* args)
{
	auto& [j, u, w] = *((tuple<size_t,size_t,vector<double>>*)args);
	max_seeds[j] = seeds_current[j];
	max_cover[j] = seeds_current_cover[j];
	score[j] = 0;
	for (size_t i = 0; i < l; i++)
		score[j] += (w[i] * max_cover[j][i].size());
	vector<unordered_set<size_t>> freed = vector<unordered_set<size_t>>(l, unordered_set<size_t>());
	if (max_seeds[j].find(u) != max_seeds[j].end())
	{
		double r = 0;
		for (size_t i = 0; i < l; i++)
		{
			unordered_set<size_t> others = unordered_set<size_t>();
			for (size_t x : max_seeds[j])
				if (x != u)
					others.insert(h_cv[i][x].begin(), h_cv[i][x].end());
			for (size_t x : h_cv[i][u])
				if (others.find(x) == others.end())
					freed[i].insert(x);
			r += (w[i] * freed[i].size());
		}
		if (r >= (pow(1 + epsilon1, j) - score[j]) / k)
		{
			max_seeds[j].erase(u);
			for (size_t i = 0; i < l; i++)
				for (size_t x : freed[i])
					max_cover[j][i].erase(x);
			score[j] -= r;
			size_t vertex = n;
			double mg = 0;
			for (auto& e : node_cover[j])
				if (e.first != u)
				{
					double g = 0;
					for (auto& f : e.second)
						g += (w[f.first] * f.second.size());
					for (size_t i = 0; i < l; i++)
						for (size_t x : freed[i])
							if (h_cv[i][e.first].find(x) != h_cv[i][x].end())
								g += w[i];
					if (g >= mg and g >= (pow(1 + epsilon1, j) - score[j]) / k)
					{
						mg = g;
						vertex = e.first;
					}
				}
			if (mg)
			{
				score[j] += mg;
				max_seeds[j].insert(vertex);
				for (auto& f : node_cover[j][vertex])
					max_cover[j][f.first].insert(f.second.begin(), f.second.end());
			}
		}
	}
	return NULL;
}

auto greedy_remove(pair<size_t,size_t>& edge, vector<double>& w)
{
	size_t u = edge.first, v = edge.second;
	vector<tuple<size_t,size_t,vector<double>>> t = vector<tuple<size_t,size_t,vector<double>>>(threads);
	vector<pthread_t> tid = vector<pthread_t>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		t[j] = make_tuple(j, u, w);
		int rc = pthread_create(&tid[j], NULL, &greedy_remove_helper, (void*)(&t[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	// return make_tuple(max_seeds, max_cover, pos);
	return distance(score.begin(), max_element(score.begin(), score.end()));
}

// auto greedy_remove(pair<size_t,size_t>& edge, vector<double>& w)
// {
// 	double mst = 0;
// 	size_t pos = 0, u = edge.first, v = edge.second;
// 	vector<unordered_set<size_t>> max_seeds = vector<unordered_set<size_t>>(threads);
// 	vector<vector<unordered_set<size_t>>> max_cover = vector<vector<unordered_set<size_t>>>(threads);
// 	for (size_t j = 0; j < threads; j++)
// 	{
// 		max_seeds[j] = seeds_current[j];
// 		max_cover[j] = seeds_current_cover[j];
// 		double score = 0;
// 		for (size_t i = 0; i < l; i++)
// 			score += (w[i] * seeds_current_cover[j][i].size());
// 		vector<unordered_set<size_t>> freed = vector<unordered_set<size_t>>(l, unordered_set<size_t>());
// 		if (seeds_current[j].find(u) != seeds_current[j].end())
// 		{
// 			double r = 0;
// 			for (size_t i = 0; i < l; i++)
// 			{
// 				unordered_set<size_t> others = unordered_set<size_t>();
// 				for (size_t x : seeds_current[j])
// 					if (x != u)
// 						others.insert(h_cv[i][x].begin(), h_cv[i][x].end());
// 				for (size_t x : h_cv[i][u])
// 					if (others.find(x) == others.end())
// 						freed[i].insert(x);
// 				r += (w[i] * freed[i].size());
// 			}
// 			if (r >= (pow(1 + epsilon1, j) - score) / k)
// 			{
// 				max_seeds[j].erase(u);
// 				for (size_t i = 0; i < l; i++)
// 					for (size_t x : freed[i])
// 						max_cover[j][i].erase(x);
// 				score -= r;
// 				size_t vertex = n;
// 				double mg = 0;
// 				for (auto& e : node_cover[j])
// 					if (e.first != u)
// 					{
// 						double g = 0;
// 						for (auto& f : e.second)
// 							g += (w[f.first] * f.second.size());
// 						for (size_t i = 0; i < l; i++)
// 							for (size_t x : freed[i])
// 								if (h_cv[i][e.first].find(x) != h_cv[i][x].end())
// 									g += w[i];
// 						if (g >= mg and g >= (pow(1 + epsilon1, j) - score) / k)
// 						{
// 							mg = g;
// 							vertex = e.first;
// 						}
// 					}
// 				if (mg)
// 				{
// 					score += mg;
// 					max_seeds[j].insert(vertex);
// 					for (auto& f : node_cover[j][vertex])
// 						max_cover[j][f.first].insert(f.second.begin(), f.second.end());
// 				}
// 			}
// 		}
// 		if (score > mst)
// 		{
// 			mst = score;
// 			pos = j;
// 		}
// 	}
// 	return make_tuple(max_seeds, max_cover, pos);
// }

void* remove_edge_cov_helper(void* args)
{
	auto& [j, edge] = *((pair<size_t,pair<size_t,size_t>>*)args);
	flag[j] = true;
	size_t u = edge.first, v = edge.second;
	if (seeds_current[j].find(u) == seeds_current[j].end())
		for (size_t h : pres[edge])
			node_cover[j][u][h].erase(v);
	else
	{
		flag[j] = false;
		for (size_t h : pres[edge])
		{
			bool signal = true;
			for (size_t x : seeds_current[j])
				if (h_cv[h][x].find(v) != h_cv[h][x].end())
				{
					signal = false;
					break;
				}
			if (signal)
			{
				seeds_current_cover[j][h].erase(v);
				for (auto& e : h_cv[h])
				{
					size_t x = e.first;
					if (seeds_current[j].find(x) == seeds_current[j].end() and e.second.find(v) != e.second.end())
					{
						if (node_cover[j].find(x) == node_cover[j].end())
							node_cover[j][x] = unordered_map<size_t,unordered_set<size_t>>();
						if (node_cover[j][x].find(h) == node_cover[j][x].end())
							node_cover[j][x][h] = unordered_set<size_t>();
						node_cover[j][x][h].insert(v);
					}
				}
			}
		}
	}
	return NULL;
}

void* update_remove_helper(void* args)
{
	auto& [j, v, seeds_previous] = *((tuple<size_t,size_t,unordered_set<size_t>>*)args);
	for (auto& e : node_cover[j])
		for (auto& f : e.second)
			for (auto& x : seeds_current_cover[j][f.first])
				f.second.erase(x);
	for (size_t x : seeds_current[j])
	{
		node_cover[j].erase(v);
		pthread_mutex_lock(&m_lock);
		if (seed_threads.find(x) == seed_threads.end())
			seed_threads[x] = 0;
		seed_threads[x]++;
		pthread_mutex_unlock(&m_lock);
	}
	for (size_t x : seeds_previous)
		if (seeds_current[j].find(x) == seeds_current[j].end())
			for (size_t i = 0; i < l; i++)
				if (h_cv[i].find(x) != h_cv[i].end())
				{
					if (node_cover[j].find(x) == node_cover[j].end())
						node_cover[j][x] = unordered_map<size_t,unordered_set<size_t>>();
					if (node_cover[j][x].find(i) == node_cover[j][x].end())
						node_cover[j][x][i] = unordered_set<size_t>();
					for (size_t y : h_cv[i][x])
						if (seeds_current_cover[j][i].find(y) == seeds_current_cover[j][i].end())
							node_cover[j][x][i].insert(y);
				}
	kreach[j] = seeds_current[j].size() >= k;
	return NULL;
}

void remove_edge_cov(pair<size_t,size_t>& edge)
{
	vector<pthread_t> tid = vector<pthread_t>(threads);
	vector<pair<size_t,pair<size_t,size_t>>> t = vector<pair<size_t,pair<size_t,size_t>>>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		t[j] = make_pair(j, edge);
		int rc = pthread_create(&tid[j], NULL, &remove_edge_cov_helper, (void*)(&t[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	if (all_of(flag.begin(), flag.end(), [](bool i) { return i; }))
		return;
	seeds_return = unordered_set<size_t>();
	double eta = sqrt(log(l) / 2 / T);
	vector<double> sum = vector<double>(l, 0), w = vector<double>(l, 1.0 / l);
	vector<unordered_set<size_t>> seeds = vector<unordered_set<size_t>>(T), seeds_tmp, seeds_previous = seeds_current; 
	vector<vector<unordered_set<size_t>>> cover_tmp;
	for (size_t j = 0; j < T; j++)
	{
		auto res = greedy_remove(edge, w);
		// seeds[j] = get<0>(res)[get<2>(res)];
		seeds[j] = max_seeds[res];
		double s = 0;
		for (size_t i = 0; i < l; i++)
		{
			// sum[i] += get<1>(res)[get<2>(res)].size();
			sum[i] += max_cover[res].size();
			double wt = exp(- eta * sum[i]);
			w[i] = wt;
			s += wt;
		}
		for (size_t i = 0; i < l; i++)
			w[i] /= s;
		// for (auto& x : get<0>(res)[get<2>(res)])
		// 	seeds_return.insert(x);
		if (j == 0)
		{
			// seeds_tmp = get<0>(res);
			// cover_tmp = get<1>(res);
			seeds_tmp = max_seeds;
			cover_tmp = max_cover;
		}
	}
	seeds_return = seeds[rand() % T];
	seeds_current = seeds_tmp;
	seeds_current_cover = cover_tmp;
	seed_threads.clear();
	vector<tuple<size_t,size_t,unordered_set<size_t>>> tt = vector<tuple<size_t,size_t,unordered_set<size_t>>>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		tt[j] = make_tuple(j, edge.second, seeds_previous[j]);
		int rc = pthread_create(&tid[j], NULL, &update_remove_helper, (void*)(&tt[j]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t j = 0; j < threads; j++)
		pthread_join(tid[j], NULL);
	// for (size_t j = 0; j < threads; j++)
	// {
	// 	for (auto& e : node_cover[j])
	// 		for (auto& f : e.second)
	// 			for (auto& x : seeds_current_cover[j][f.first])
	// 				f.second.erase(x);
	// 	for (size_t x : seeds_current[j])
	// 	{
	// 		node_cover[j].erase(edge.second);
	// 		if (seed_threads.find(x) == seed_threads.end())
	// 			seed_threads[x] = 0;
	// 		seed_threads[x]++;
	// 	}
	// 	for (size_t x : seeds_previous[j])
	// 		if (seeds_current[j].find(x) == seeds_current[j].end())
	// 			for (size_t i = 0; i < l; i++)
	// 				if (h_cv[i].find(x) != h_cv[i].end())
	// 				{
	// 					if (node_cover[j].find(x) == node_cover[j].end())
	// 						node_cover[j][x] = unordered_map<size_t,unordered_set<size_t>>();
	// 					if (node_cover[j][x].find(i) == node_cover[j][x].end())
	// 						node_cover[j][x][i] = unordered_set<size_t>();
	// 					for (size_t y : h_cv[i][x])
	// 						if (seeds_current_cover[j][i].find(y) == seeds_current_cover[j][i].end())
	// 							node_cover[j][x][i].insert(y);
	// 				}
	// 	if (seeds_current[j].size() < k)
	// 		num_k--;
	// }
}

void* gen_est_helper(void* args)
{
	auto& [i, v] = *((pair<size_t,size_t>*)args);
	unsigned int seed = time(NULL)^i;
	auto res = gen_rrset(v, i, seed);
	if (rr_est[i].find(v) == rr_est[i].end())
		rr_est[i][v] = vector<unordered_map<size_t,unordered_set<size_t>>>();
	rr_est[i][v].push_back(res.first);
	// steps_est += res.second;
	for (auto& e : res.first)
	{
		size_t u = e.first;
		if (h_est[i].find(u) == h_est[i].end())
			h_est[i][u] = unordered_set<size_t>();
		h_est[i][u].insert(v);
	}
	return NULL;
}

void gen_est(size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<pair<size_t,size_t>> t = vector<pair<size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_pair(i, v);
		int rc = pthread_create(&tid[i], NULL, &gen_est_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
}

void gen_est()
{
	for (size_t v : nodes)
	{
		for (size_t j = 0; j < floor(p); j++)
			gen_est(v);
		if ((double)rand() / RAND_MAX < p - floor(p))
			gen_est(v);
	}
}

void* gen_cv_helper(void* args)
{
	auto& [i, v] = *((pair<size_t,size_t>*)args);
	unsigned int seed = time(NULL)^i;
	auto res = gen_rrset(v, i, seed);
	if (rr_cv[i].find(v) == rr_cv[i].end())
		rr_cv[i][v] = vector<unordered_map<size_t,unordered_set<size_t>>>();
	rr_cv[i][v].push_back(res.first);
	for (auto& e : res.first)
	{
		size_t u = e.first;
		change[i].insert(make_pair(u, v));
	}
	return NULL;
}

void gen_cv(size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<pair<size_t,size_t>> t = vector<pair<size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_pair(i, v);
		int rc = pthread_create(&tid[i], NULL, &gen_cv_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
}

void gen_cv()
{
	for (size_t v : nodes)
	{
		for (size_t j = 0; j < floor(p); j++)
			gen_cv(v);
		if ((double)rand() / RAND_MAX < p - floor(p))
			gen_cv(v);
	}
	pres.clear();
	for (size_t i = 0; i < l; i++)
		for (auto& edge : change[i])
		{
			if (pres.find(edge) == pres.end())
				pres[edge] = unordered_set<size_t>();
			pres[edge].insert(i);
		}
	vector<vector<pair<size_t,size_t>>> freq = vector<vector<pair<size_t,size_t>>>(l + 1, vector<pair<size_t,size_t>>());
	for (auto& e : pres)
		freq[e.second.size()].push_back(e.first);
	for (size_t j = l; j > 0; j--)
	{
		for (auto& edge : freq[j])
		{
			for (size_t i : pres[edge])
			{
				if (h_cv[i].find(edge.first) == h_cv[i].end())
					h_cv[i][edge.first] = unordered_set<size_t>();
				h_cv[i][edge.first].insert(edge.second);
			}
			insert_edge_cov(edge);
		}
	}
}

void* aug_est_helper(void* args)
{
	auto& [i, u, v] = *((tuple<size_t,size_t,size_t>*)args);
	unsigned int seed = time(NULL)^i;
	if (h_est[i].find(v) != h_est[i].end())
		for (auto& x : h_est[i][v])
			for (auto& rr : rr_est[i][x])
			{
				// steps_est++;
				if (rr.find(v) != rr.end() and rr.find(u) == rr.end() and (double)rand_r(&seed) / RAND_MAX < prob(u, v, i))
				{
					queue<size_t> q = queue<size_t>();
					unordered_set<size_t> visited = unordered_set<size_t>();
					q.push(u);
					visited.insert(u);
					rr[v].insert(u);
					while (not q.empty())
					{
						size_t a = q.front();
						q.pop();
						rr[a] = unordered_set<size_t>();
						h_est[i][a].insert(x);
						for (auto& b : rev_graph[a])
						{
							// steps_est++;
							if (visited.find(b) == visited.end() and rr.find(b) == rr.end() and (double)rand_r(&seed) / RAND_MAX < prob(b, a, i))
							{
								visited.insert(b);
								rr[a].insert(b);
								q.push(b);
							}
						}
					}
				}
			}
	return NULL;
}

void aug_est(size_t	u, size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<tuple<size_t,size_t,size_t>> t = vector<tuple<size_t,size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_tuple(i, u, v);
		int rc = pthread_create(&tid[i], NULL, &aug_est_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
}

void* aug_cv_helper(void* args)
{
	auto& [i, u, v] = *((tuple<size_t,size_t,size_t>*)args);
	unsigned int seed = time(NULL)^i;
	if (h_cv[i].find(v) != h_cv[i].end())
		for (auto& x : h_cv[i][v])
			for (auto& rr : rr_cv[i][x])
			{
				if (rr.find(v) != rr.end() and rr.find(u) == rr.end())
				{
					if ((double)rand_r(&seed) / RAND_MAX < prob(u, v, i))
					{
						queue<size_t> q = queue<size_t>();
						unordered_set<size_t> visited = unordered_set<size_t>();
						q.push(u);
						visited.insert(u);
						rr[v].insert(u);
						while (not q.empty())
						{
							size_t a = q.front();
							q.pop();
							if (seed_threads.find(a) == seed_threads.end() or seed_threads[a] <= threads * 0.1)
								change[i].insert(make_pair(a, x));
							rr[a] = unordered_set<size_t>();
							for (auto& b : rev_graph[a])
								if (visited.find(b) == visited.end() and rr.find(b) == rr.end() and (double)rand_r(&seed) / RAND_MAX < prob(b, a, i))
								{
									visited.insert(b);
									rr[a].insert(b);
									q.push(b);
								}
						}
					}
				}
			}
	return NULL;
}

void aug_cv(size_t u, size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<tuple<size_t,size_t,size_t>> t = vector<tuple<size_t,size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_tuple(i, u, v);
		int rc = pthread_create(&tid[i], NULL, &aug_cv_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
	for (size_t i = 0; i < l; i++)
		for (auto& edge : change[i])
		{
			if (pres.find(edge) == pres.end())
				pres[edge] = unordered_set<size_t>();
			pres[edge].insert(i);
		}
	vector<vector<pair<size_t,size_t>>> freq = vector<vector<pair<size_t,size_t>>>(l + 1, vector<pair<size_t,size_t>>());
	for (auto& e : pres)
		freq[e.second.size()].push_back(e.first);
	for (size_t j = l; j > 0; j--)
		for (auto& edge : freq[j])
		{
			for (size_t i : pres[edge])
			{
				if (h_cv[i].find(edge.first) == h_cv[i].end())
					h_cv[i][edge.first] = unordered_set<size_t>();
				h_cv[i][edge.first].insert(edge.second);
			}
			insert_edge_cov(edge);
		}
}

void dim(unordered_map<size_t,unordered_set<size_t>>& rr, size_t u, unordered_set<size_t>& removed)
{
	// for (size_t x : rr[u])
	// 	dim(rr, x, removed);
	// rr.erase(u);
	// removed.insert(u);
	stack<size_t> s = stack<size_t>();
	s.push(u);
	while (not s.empty())
	{
		size_t v = s.top();
		if (rr[v].empty())
		{
			rr.erase(v);
			removed.insert(v);
			s.pop();
		}
		else
		{
			for (size_t x : rr[v])
				s.push(x);
			rr[v].clear();
		}
	}
}

void* dim_est_helper(void* args)
{
	auto& [i, u, v] = *((tuple<size_t,size_t,size_t>*)args);
	if (h_est[i].find(v) != h_est[i].end())
		for (size_t x : h_est[i][v])
		{
			unordered_set<size_t> removed = unordered_set<size_t>(), remained = unordered_set<size_t>();
			for (auto& rr : rr_est[i][x])
			{
				if (rr.find(v) != rr.end() and rr[v].find(u) != rr[v].end())
				{
					dim(rr, u, removed);
					rr[v].erase(u);
				}
				for (auto& e : rr)
					remained.insert(e.first);
			}
			for (size_t y : removed)
				if (remained.find(y) == remained.end())
					h_est[i][y].erase(x);
		}
		return NULL;
}

void dim_est(size_t u, size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<tuple<size_t,size_t,size_t>> t = vector<tuple<size_t,size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_tuple(i, u, v);
		int rc = pthread_create(&tid[i], NULL, &dim_est_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
}

void* dim_cv_helper(void* args)
{
	auto& [i, u, v] = *((tuple<size_t,size_t,size_t>*)args);
	if (h_cv[i].find(v) != h_cv[i].end())
		for (size_t x : h_cv[i][v])
		{
			unordered_set<size_t> removed = unordered_set<size_t>(), remained = unordered_set<size_t>();
			for (auto& rr : rr_cv[i][x])
			{
				if (rr.find(v) != rr.end() and rr[v].find(u) != rr[v].end())
				{
					dim(rr, u, removed);
					rr[v].erase(u);
				}
				for (auto& e : rr)
					remained.insert(e.first);
			}
			for (size_t y : removed)
				if (remained.find(y) == remained.end() and seed_threads.find(y) != seed_threads.end() and seed_threads[y] >= threads * 0.9)
					change[i].insert(make_pair(y, x));
		}
	return NULL;
}

void dim_cv(size_t u, size_t v)
{
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<tuple<size_t,size_t,size_t>> t = vector<tuple<size_t,size_t,size_t>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_tuple(i, u, v);
		int rc = pthread_create(&tid[i], NULL, &dim_cv_helper, (void*)(&t[i]));
		if (rc)
			cerr << "ERROR : pthread_create, rc : " << rc << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
	for (size_t i = 0; i < l; i++)
		for (auto& edge : change[i])
		{
			if (pres.find(edge) == pres.end())
				pres[edge] = unordered_set<size_t>();
			pres[edge].insert(i);
		}
	vector<vector<pair<size_t,size_t>>> freq = vector<vector<pair<size_t,size_t>>>(l + 1, vector<pair<size_t,size_t>>());
	for (auto& e : pres)
		freq[e.second.size()].push_back(e.first);
	for (size_t j = l; j > 0; j--)
		for (auto& edge : freq[j])
		{
			for (size_t i : pres[edge])
				h_cv[i][edge.first].erase(edge.second);
			remove_edge_cov(edge);
		}
}

void restart()
{
	if (m0 == 0 or n0 == 0)
		return;
	cout << "Restarting" << endl;
	// double tmp = 8 * B * d * m0 * n0 / epsilon2;
	// l = ceil(d * pow(tmp, d) * log(tmp / delta2));
	// T = ceil(2 * log(l) / pow(epsilon2, 2));
	double c = 1;
	// double delta = n0 * pow(delta1 / 8 / pow(2 * n0, T * k), 12 * pow(n0, 2) / c / pow(k, 3));
	R = c * k * log(n0 / delta1) / pow(epsilon1, 2);
	threads = ceil(log(n0) / epsilon1);
	h_est.clear();
	h_est = vector<unordered_map<size_t,unordered_set<size_t>>>(l, unordered_map<size_t,unordered_set<size_t>>());
	h_cv.clear();
	h_cv = vector<unordered_map<size_t,unordered_set<size_t>>>(l, unordered_map<size_t,unordered_set<size_t>>());
	rr_est = vector<unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>>(l, unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>());
	rr_cv = vector<unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>>(l, unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>());
	seeds_current.clear();
	seeds_current = vector<unordered_set<size_t>>(threads, unordered_set<size_t>());
	seeds_current_cover.clear();
	seeds_current_cover = vector<vector<unordered_set<size_t>>>(threads, vector<unordered_set<size_t>>(l, unordered_set<size_t>()));
	max_seeds.clear();
	max_seeds = vector<unordered_set<size_t>>(threads);
	max_cover.clear();
	max_cover = vector<vector<unordered_set<size_t>>>(threads);
	node_cover.clear();
	node_cover = vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>>(threads, unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>());
	steps_est = 0;
	kreach.clear();
	kreach = vector<bool>(threads, false);
	flag.clear();
	flag = vector<bool>(threads, true);
	score.clear();
	score = vector<double>(threads, 0);
	change.clear();
	change = vector<unordered_set<pair<size_t,size_t>>>(l, unordered_set<pair<size_t,size_t>>());
	probs.clear();
	probs = vector<unordered_map<size_t,unordered_map<size_t,double>>>(l, unordered_map<size_t,unordered_map<size_t,double>>());
	vector<size_t> vec = vector<size_t>(nodes.begin(), nodes.end());
	unsigned int seed = time(NULL);
	timespec begin, end;
	pthread_mutex_init(&m_lock, NULL);
	clock_gettime(CLOCK_MONOTONIC, &begin);
	size_t num = 0, K = 0;
	do
	{
		size_t v = vec[rand() % nodes.size()];
		for (size_t i = 0; i < l; i++)
			num += gen_rrset(v, i, seed).second;
		K++;
	}
	while (num < R * m0 * l);
	p = (double)K / n0;
	// gen_est();
	gen_cv();
	clock_gettime(CLOCK_MONOTONIC, &end);
	pthread_mutex_destroy(&m_lock);
	elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
	cout << "Current seed set : ";
	for (size_t x : seeds_return)
		cout << x << " ";
	cout << endl << "Running time : " << elapsed << " seconds" << endl;
	elapsed = 0;
}

void insert_node(size_t v)
{
	nodes.insert(v);
	n++;
	if (n >= 2 * n0)
	{
		m0 = m;
		n0 = n;
		restart();
	}
	else
		for (size_t i = 0; i < l; i++)
		{
			h_est[i][v] = unordered_set<size_t>();
			h_cv[i][v] = unordered_set<size_t>();
			if ((double)rand() / RAND_MAX < p - floor(p))
			{
				h_est[i][v].insert(v);
				h_cv[i][v].insert(v);
			}
		}
}

void insert_edge(size_t u, size_t v)
{
	if (rev_graph.find(v) == rev_graph.end())
		rev_graph[v] = unordered_set<size_t>();
	rev_graph[v].insert(u);
	m++;
	if (m >= 2 * m0)
	{
		n0 = n;
		m0 = m;
		restart();
	}
	else
	{
		for (size_t i = 0; i < l; i++)
			change[i].clear();
		pres.clear();
		timespec begin, end;
		pthread_mutex_init(&m_lock, NULL);
		// clock_gettime(CLOCK_MONOTONIC, &begin);
		// aug_est(u, v);
		// clock_gettime(CLOCK_MONOTONIC, &end);
		// elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		// if (steps_est >= 16 * R * m0 * l)
		// {
		// 	n0 = n;
		// 	m0 = m;
		// 	restart();
		// 	return;
		// }
		clock_gettime(CLOCK_MONOTONIC, &begin);
		aug_cv(u, v);
		clock_gettime(CLOCK_MONOTONIC, &end);
		pthread_mutex_destroy(&m_lock);
		elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		cout << "Current seed set : ";
		for (size_t x : seeds_return)
			cout << x << " ";
		cout << endl << "Running time : " << elapsed << " seconds" << endl;
		elapsed = 0;
	}
}

void remove_node(size_t v)
{
	nodes.erase(v);
	n--;
	if (n <= 0.5 * n0)
	{
		m0 = m;
		n0 = n;
		restart();
	}
	else
		for (size_t i = 0; i < l; i++)
		{
			h_est[i].erase(v);
			h_cv[i].erase(v);
		}
}

void remove_edge(size_t u, size_t v)
{
	if (rev_graph.find(v) == rev_graph.end())
		return;
	rev_graph[v].erase(u);
	m--;
	if (m <= 0.5 * m0)
	{
		n0 = n;
		m0 = m;
		restart();
	}
	else
	{
		for (size_t i = 0; i < l; i++)
			change[i].clear();
		pres.clear();
		timespec begin, end;
		pthread_mutex_init(&m_lock, NULL);
		// clock_gettime(CLOCK_MONOTONIC, &begin);
		// dim_est(u, v);
		// clock_gettime(CLOCK_MONOTONIC, &end);
		// elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		// if (steps_est >= 16 * R * m0 * l)
		// {
		// 	n0 = n;
		// 	m0 = m;
		// 	restart();
		// 	return;
		// }
		clock_gettime(CLOCK_MONOTONIC, &begin);
		dim_cv(u, v);
		clock_gettime(CLOCK_MONOTONIC, &end);
		pthread_mutex_destroy(&m_lock);
		elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		cout << "Current seed set : ";
		for (size_t x : seeds_return)
			cout << x << " ";
		cout << endl << "Running time : " << elapsed << " seconds" << endl;
		elapsed = 0;
	}
}

int main(int argc, char* argv[])
{
	if (argc < 12)
	{
		cerr << "Usage : ./<executable> <path-to-graph> <k> <d> <B> <epsilon1> <epsilon2> <delta1> <delta2> <l> <T> <path-to-hyperparameters>" << endl;
		return EXIT_FAILURE;
	}
	srand(time(NULL));
	string graph_path = string(argv[1]) + "graph.txt";
	string features_path = string(argv[1]) + "features.txt";
	n = 0;
	m = 0;
	n0 = 0;
	m0 = 0;
	k = atoi(argv[2]);
	d = atoi(argv[3]);
	B = atoi(argv[4]);
	epsilon1 = atof(argv[5]);
	epsilon2 = atof(argv[6]);
	delta1 = atof(argv[7]);
	delta2 = atof(argv[8]);
	l = atoi(argv[9]);
	T = atoi(argv[10]);
	elapsed = 0;
	nodes = unordered_set<size_t>();
	features = unordered_map<size_t,vector<double>>();
	rev_graph = unordered_map<size_t,unordered_set<size_t>>();
	h_est = vector<unordered_map<size_t,unordered_set<size_t>>>();
	h_cv = vector<unordered_map<size_t,unordered_set<size_t>>>();
	rr_est = vector<unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>>();
	rr_cv = vector<unordered_map<size_t,vector<unordered_map<size_t,unordered_set<size_t>>>>>();
	kreach = vector<bool>();
	flag = vector<bool>();
	score = vector<double>();
	hp = vector<vector<double>>(l, vector<double>(d));
	probs = vector<unordered_map<size_t,unordered_map<size_t,double>>>();
	seeds_current = vector<unordered_set<size_t>>();
	max_seeds = vector<unordered_set<size_t>>();
	seeds_current_cover = vector<vector<unordered_set<size_t>>>();
	max_cover = vector<vector<unordered_set<size_t>>>();
	node_cover = vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>>();
	change = vector<unordered_set<pair<size_t,size_t>>>();
	pres = unordered_map<pair<size_t,size_t>,unordered_set<size_t>>();
	seed_threads = unordered_map<size_t,size_t>();
	size_t u, v;
	ifstream ff(features_path);
	while (ff >> v)
	{
		features[v] = vector<double>(d / 2);
		for (size_t i = 0; i < d / 2; i++)
			ff >> features[v][i];
	}
	ff.close();
	ifstream fg(graph_path);
	while (fg >> u >> v)
	{
		if (nodes.find(u) == nodes.end())
		{
			nodes.insert(u);
			n++;
			n0++;
		}
		if (nodes.find(v) == nodes.end())
		{
			nodes.insert(v);
			n++;
			n0++;
		}
		if (rev_graph.find(v) == rev_graph.end())
			rev_graph[v] = unordered_set<size_t>();
		rev_graph[v].insert(u);
		m++;
		m0++;
	}
	fg.close();
	ifstream fh(argv[11]);
	for (size_t i = 0; i < l; i++)
		for (size_t j = 0; j < d; j++)
			fh >> hp[i][j];
		// generate(hp[i].begin(), hp[i].end(), [](){ return pow(-1, rand() % 2) * B * (double)rand() / RAND_MAX; });
	fh.close();
	restart();
	vector<size_t> vec = vector<size_t>(nodes.begin(), nodes.end());
	size_t num = m0 * 2, ins = 0, rem = 0;
	for (int i = 0; i < num; i++)
		if ((double)rand() / RAND_MAX < 0.5)
		{
			size_t u, v;
			do
			{
				u = vec[rand() % nodes.size()];
				v = vec[rand() % nodes.size()];
			}
			while (u == v or rev_graph[v].find(u) != rev_graph[v].end());
			cout << i << " " << u << " " << v << " insert" << endl;
			insert_edge(u, v);
			ins++;
		}
		else
		{
			size_t u, v;
			do
			{
				u = vec[rand() % nodes.size()];
				v = vec[rand() % nodes.size()];
			}
			while (u == v or rev_graph[v].find(u) == rev_graph[v].end());
			cout << i << " " << u << " " << v << " remove" << endl;
			remove_edge(u, v);
			rem++;
		}
	cout << "Inserted : " << ins << endl << "Removed : " << rem << endl;
	return EXIT_SUCCESS;
}
