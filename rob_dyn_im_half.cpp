#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/functional/hash.hpp>
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace boost::heap;

size_t n, m, n0, m0, k, l, T, d, B, threads, num_k, steps_est;
double epsilon1, epsilon2, delta1, delta2, R, p, elapsed;
unordered_set<size_t> nodes, seeds_return;
unordered_map<size_t,vector<double>> features;
unordered_map<size_t,unordered_map<size_t,unordered_map<size_t,double>>> probs;
vector<double> scores_current;
vector<vector<double>> hp;
unordered_map<size_t,unordered_set<size_t>> rev_graph;
vector<unordered_map<size_t,unordered_set<size_t>>> h_est;
vector<unordered_map<size_t,unordered_set<size_t>>> h_cv;
vector<unordered_set<size_t>> seeds_current;
vector<vector<unordered_set<size_t>>> seeds_current_cover;
vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>> node_cover;
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
	if (probs.find(u) == probs.end())
		probs[u] = unordered_map<size_t,unordered_map<size_t,double>>();
	if (probs[u].find(v) == probs[u].end())
		probs[u][v] = unordered_map<size_t,double>();
	if (probs[u][v].find(i) == probs[u][v].end())
	{
		vector<double> x = vector<double>();
		x.insert(x.end(), features[u].begin(), features[u].end());
		x.insert(x.end(), features[v].begin(), features[v].end());
		double s = 0;
		for (size_t j = 0; j < d; j++)
			s += (hp[i][j] * x[j]);
		probs[u][v][i] = 1 / (1 + exp(-s));
	}
	return probs[u][v][i];
}

pair<unordered_set<size_t>,size_t> gen_rrset(size_t v, size_t i)
{
	unordered_set<size_t> rr = unordered_set<size_t>();
	size_t num = 0;
	queue<size_t> q = queue<size_t>();
	q.push(v);
	while (not q.empty())
	{
		size_t u = q.front();
		q.pop();
		rr.insert(u);
		for (size_t x : rev_graph[u])
		{
			num++;
			if (rr.find(x) == rr.end() and (double)rand() / RAND_MAX < prob(x, u, i))
				q.push(x);
		}
	}
	return make_pair(rr, num);
}

auto greedy(pair<size_t,size_t>& edge, vector<double>& w)
{
	double mst = 0;
	size_t pos = 0, u = edge.first, v = edge.second;
	vector<unordered_set<size_t>> max_seeds = vector<unordered_set<size_t>>(threads);
	vector<vector<unordered_set<size_t>>> max_cover = vector<vector<unordered_set<size_t>>>(threads);
	for (size_t j = 0; j < threads; j++)
	{
		max_seeds[j] = seeds_current[j];
		max_cover[j] = seeds_current_cover[j];
		double score = 0;
		for (size_t i = 0; i < l; i++)
			score += (w[i] * seeds_current_cover[j][i].size());
		if (seeds_current[j].size() < k and seeds_current[j].find(u) == seeds_current[j].end())
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
			while (gain >= (pow(1 + epsilon1, j) - score) / k and max_seeds[j].size() < k)
			{
				max_seeds[j].insert(vertex);
				score += gain;
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
		if (score > mst)
		{
			mst = score;
			pos = j;
		}
	}
	return make_tuple(max_seeds, max_cover, pos);
}

void insert_edge_cov(pair<size_t,size_t>& edge, unordered_map<pair<size_t,size_t>,unordered_set<size_t>>& pres)
{
	if (num_k >= threads / 2)
		return;
	size_t u = edge.first, v = edge.second;
	bool flag = true;
	for (size_t j = 0; j < threads; j++)
		if (seeds_current[j].size() < k)
		{
			if (seeds_current[j].find(u) == seeds_current[j].end())
			{
				flag = false;
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
	if (flag)
		return;
	// seeds_return = unordered_set<size_t>();
	double eta = sqrt(log(l) / 2 / T);
	vector<double> sum = vector<double>(l, 0), w = vector<double>(l, 1.0 / l);
	vector<unordered_set<size_t>> seeds = vector<unordered_set<size_t>>(T), seeds_tmp;
	vector<vector<unordered_set<size_t>>> cover_tmp;
	for (size_t j = 0; j < T; j++)
	{
		auto res = greedy(edge, w);
		seeds[j] = get<0>(res)[get<2>(res)];
		double s = 0;
		for (size_t i = 0; i < l; i++)
		{
			sum[i] += get<1>(res)[get<2>(res)].size();
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
			seeds_tmp = get<0>(res);
			cover_tmp = get<1>(res);
		}
	}
	seeds_return = seeds[rand() % T];
	seeds_current = seeds_tmp;
	seeds_current_cover = cover_tmp;
	num_k = threads;
	for (size_t j = 0; j < threads; j++)
	{
		for (auto& e : node_cover[j])
			for (auto& f : e.second)
				for (auto& x : seeds_current_cover[j][f.first])
					f.second.erase(x);
		for (size_t v : seeds_current[j])
			node_cover[j].erase(v);
		if (seeds_current[j].size() < k)
			num_k--;
	}
}

void gen_est(size_t v)
{
	for (size_t i = 0; i < l; i++)
	{
		pair<unordered_set<size_t>,size_t> res = gen_rrset(v, i);
		steps_est += res.second;
		for (size_t u : res.first)
		{
			if (h_est[i].find(u) == h_est[i].end())
				h_est[i][u] = unordered_set<size_t>();
			h_est[i][u].insert(v);
		}
	}
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

void gen_cv(size_t v, unordered_map<pair<size_t,size_t>,unordered_set<size_t>>& pres)
{
	for (size_t i = 0; i < l; i++)
	{
		pair<unordered_set<size_t>,size_t> res = gen_rrset(v, i);
		for (size_t u : res.first)
		{
			auto edge = make_pair(u, v);
			if (pres.find(edge) == pres.end())
				pres[edge] = unordered_set<size_t>();
			pres[edge].insert(i);
		}
	}
}

void gen_cv()
{
	unordered_map<pair<size_t,size_t>,unordered_set<size_t>> pres = unordered_map<pair<size_t,size_t>,unordered_set<size_t>>();
	for (size_t v : nodes)
	{
		for (size_t j = 0; j < floor(p); j++)
			gen_cv(v, pres);
		if ((double)rand() / RAND_MAX < p - floor(p))
			gen_cv(v, pres);
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
			insert_edge_cov(edge, pres);
		}
	}
}

void aug_est(size_t	u, size_t v)
{
	for (size_t i = 0; i < l; i++)
		for (auto& x : h_est[i][v])
		{
			steps_est++;
			if ((double)rand() / RAND_MAX < prob(u, v, i))
			{
				queue<size_t> q = queue<size_t>();
				q.push(u);
				while (not q.empty())
				{
					size_t a = q.front();
					q.pop();
					if (h_est[i][a].find(x) == h_est[i][a].end())
					{
						h_est[i][a].insert(x);
						for (auto& b : rev_graph[a])
						{
							steps_est++;
							if ((double)rand() / RAND_MAX < prob(b, a, i))
								q.push(b);
						}
					}
				}
			}
		}
}

void aug_cv(size_t u, size_t v)
{
	if (num_k >= threads / 2)
		return;
	unordered_map<pair<size_t,size_t>,unordered_set<size_t>> pres = unordered_map<pair<size_t,size_t>,unordered_set<size_t>>();
	for (size_t i = 0; i < l; i++)
		for (auto& x : h_cv[i][v])
			if ((double)rand() / RAND_MAX < prob(u, v, i))
			{
				queue<size_t> q = queue<size_t>();
				q.push(u);
				while (not q.empty())
				{
					size_t a = q.front();
					q.pop();
					auto edge = make_pair(a, x);
					if (pres.find(edge) == pres.end())
						pres[edge] = unordered_set<size_t>();
					if (pres[edge].find(i) == pres[edge].end())
					{
						pres[edge].insert(i);
						for (auto& b : rev_graph[a])
							if ((double)rand() / RAND_MAX < prob(b, a, i))
								q.push(b);
					}
				}
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
			insert_edge_cov(edge, pres);
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
	num_k = 0;
	h_est.clear();
	h_est = vector<unordered_map<size_t,unordered_set<size_t>>>(l, unordered_map<size_t,unordered_set<size_t>>());
	h_cv.clear();
	h_cv = vector<unordered_map<size_t,unordered_set<size_t>>>(l, unordered_map<size_t,unordered_set<size_t>>());
	seeds_current.clear();
	seeds_current = vector<unordered_set<size_t>>(threads, unordered_set<size_t>());
	seeds_current_cover.clear();
	seeds_current_cover = vector<vector<unordered_set<size_t>>>(threads, vector<unordered_set<size_t>>(l, unordered_set<size_t>()));
	node_cover.clear();
	node_cover = vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>>(threads, unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>());
	steps_est = 0;
	vector<size_t> vec = vector<size_t>(nodes.begin(), nodes.end());
	timespec begin, end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	size_t num = 0, K = 0;
	do
	{
		size_t v = vec[rand() % nodes.size()];
		for (size_t i = 0; i < l; i++)
			num += gen_rrset(v, i).second;
		K++;
	}
	while (num < R * m0 * l);
	p = (double)K / n0;
	// cout << p << endl;
	gen_est();
	gen_cv();
	clock_gettime(CLOCK_MONOTONIC, &end);
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
	features[v] = vector<double>(d / 2);
	cout << "Features : ";
	for (size_t i = 0; i < d / 2; i++)
	{
		features[v][i] = (double)rand() / RAND_MAX * B * pow(-1, rand() % 2);
		cout << features[v][i] << " ";
	}
	cout << endl;
	if (n >= 2 * n0)
	{
		m0 = m;
		n0 = n;
		restart();
	}
	else
	{
		timespec begin, end;
		clock_gettime(CLOCK_MONOTONIC, &begin);
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
		clock_gettime(CLOCK_MONOTONIC, &end);
		elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		cout << "Current seed set : ";
		for (size_t x : seeds_return)
			cout << x << " ";
		cout << endl << "Running time : " << elapsed << " seconds" << endl;
		elapsed = 0;
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
		timespec begin, end;
		clock_gettime(CLOCK_MONOTONIC, &begin);
		aug_est(u, v);
		clock_gettime(CLOCK_MONOTONIC, &end);
		elapsed += ((end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
		if (steps_est >= 16 * R * m0 * l)
		{
			n0 = n;
			m0 = m;
			restart();
			return;
		}
		clock_gettime(CLOCK_MONOTONIC, &begin);
		aug_cv(u, v);
		clock_gettime(CLOCK_MONOTONIC, &end);
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
	if (argc < 8)
	{
		cerr << "Usage : ./<executable> <path-to-graph> <k> <d> <B> <l> <T> <path-to-hyperparameters>" << endl;
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
	epsilon1 = 0.9;
	epsilon2 = 0.9;
	delta1 = 0.9;
	delta2 = 0.9;
	l = atoi(argv[5]);
	T = atoi(argv[6]);
	elapsed = 0;
	nodes = unordered_set<size_t>();
	features = unordered_map<size_t,vector<double>>();
	rev_graph = unordered_map<size_t,unordered_set<size_t>>();
	h_est = vector<unordered_map<size_t,unordered_set<size_t>>>();
	h_cv = vector<unordered_map<size_t,unordered_set<size_t>>>();
	hp = vector<vector<double>>(l, vector<double>(d));
	probs = unordered_map<size_t,unordered_map<size_t,unordered_map<size_t,double>>>();
	seeds_current = vector<unordered_set<size_t>>();
	seeds_current_cover = vector<vector<unordered_set<size_t>>>();
	node_cover = vector<unordered_map<size_t,unordered_map<size_t,unordered_set<size_t>>>>();
	size_t u, v, vm = 0;
	ifstream ff(features_path);
	while (ff >> v)
	{
		if (vm < v)
			vm = v;
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
	ifstream fh(argv[7]);
	for (size_t i = 0; i < l; i++)
		for (size_t j = 0; j < d; j++)
			fh >> hp[i][j];
	fh.close();
	restart();
	size_t num = m0 * 2;
	for (int i = 0; i < num; i++)
		if ((double)rand() / RAND_MAX < 0.5)
		{
			vector<size_t> vec = vector<size_t>(nodes.begin(), nodes.end());
			size_t u = vec[rand() % nodes.size()], v = vec[rand() % nodes.size()];
			if (u == v or rev_graph[v].find(u) != rev_graph[v].end())
			{
				i--;
				continue;
			}
			cout << i << " " << u << " " << v << " insert" << endl;
			insert_edge(u, v);
		}
		else
		{
			cout << i << " " << vm + 1 << " insert" << endl;
			insert_node(vm + 1);
			vm++;
		}
	return EXIT_SUCCESS;
}
