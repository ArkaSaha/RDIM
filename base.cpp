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
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace boost::heap;

size_t k, l, T, d, B;
unordered_set<size_t> nodes;
unordered_map<size_t,vector<double>> features;
unordered_map<size_t,unordered_map<size_t,unordered_map<size_t,double>>> probs;
vector<vector<double>> hp;
unordered_map<size_t,unordered_set<size_t>> rev_graph;
struct node
{
	size_t vertex;
	double gain;
	node(const size_t& v, double g) : vertex(v), gain(g) {}
};
struct compare_node
{
	bool operator()(const node& n1, const node& n2) const
	{
		return n1.gain > n2.gain;
	}
};
using handle_t = fibonacci_heap<node,compare<compare_node>>::handle_type;

double prob(size_t u, size_t v, size_t i)
{
	if (probs.find(u) == probs.end() or probs[u].find(v) == probs[u].end() or probs[u][v].find(i) == probs[u][v].end())
	{
		vector<double> x = vector<double>();
		x.insert(x.end(), features[u].begin(), features[u].end());
		x.insert(x.end(), features[v].begin(), features[v].end());
		double p = 0;
		for (size_t j = 0; j < d; j++)
			p += (hp[i][j] * x[j]);
		probs[u][v][i] = 1 / (1 + exp(-p));
	}
	return probs[u][v][i];
}

unordered_set<size_t> gen_rrset(size_t v, size_t i)
{
	unordered_set<size_t> rr = unordered_set<size_t>();
	queue<size_t> q = queue<size_t>();
	q.push(v);
	while (not q.empty())
	{
		size_t u = q.front();
		q.pop();
		rr.insert(u);
		for (size_t x : rev_graph[u])
			if (rr.find(x) == rr.end() and (double)rand() / RAND_MAX < prob(x, u, i))
				q.push(x);
	}
	return rr;
}

pair<unordered_set<size_t>,vector<double>> greedy(vector<double>& w)
{
	fibonacci_heap<node,compare<compare_node>> heap = fibonacci_heap<node,compare<compare_node>>();
	unordered_map<size_t,handle_t> handles = unordered_map<size_t,handle_t>();
	unordered_set<size_t> seeds = unordered_set<size_t>();
	vector<double> scores = vector<double>(l, 0);
	unordered_map<size_t,vector<unordered_set<size_t>>> cover = unordered_map<size_t,vector<unordered_set<size_t>>>();
	for (size_t i = 0; i < l; i++)
		for (size_t u : nodes)
		{
			unordered_set<size_t> rr = gen_rrset(u, i);
			for (size_t v : rr)
			{
				if (cover.find(v) == cover.end())
					cover[v] = vector<unordered_set<size_t>>(l, unordered_set<size_t>());
				cover[v][i].insert(u);
			}
		}
	for (auto& e : cover)
	{
		double g = 0;
		for (size_t i = 0; i < l; i++)
			g += (w[i] * e.second[i].size());
		handles[e.first] = heap.push(node(e.first, g));
	}
	for (size_t j = 0; j < k; j++)
	{
		node nd = heap.top();
		heap.pop();
		size_t v = nd.vertex;
		handles.erase(v);
		seeds.insert(v);
		auto c = cover[v];
		cover.erase(v);
		for (size_t i = 0; i < l; i++)
			scores[i] += c[i].size();
		if (heap.empty())
			break;
		for (auto& e : cover)
		{
			double diff = 0;
			for (size_t i = 0; i < l; i++)
			{
				size_t num = 0;
				for (size_t x : c[i])
					if (e.second[i].find(x) != e.second[i].end())
					{
						e.second[i].erase(x);
						num++;
					}
				diff += (w[i] * num);
			}
			heap.update(handles[e.first], node(e.first, (*handles[e.first]).gain - diff));
		}
	}
	return make_pair(seeds, scores);
}

pair<unordered_set<size_t>,double> hiro()
{
	// unordered_set<size_t> seeds_return = unordered_set<size_t>();
	vector<unordered_set<size_t>> seeds = vector<unordered_set<size_t>>(T);
	double eta = sqrt(log(l) / 2 / T);
	vector<double> sum = vector<double>(l, 0), w = vector<double>(l, 1.0 / l);
	timespec begin, end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	for (size_t j = 0; j < T; j++)
	{
		auto res = greedy(w);
		double s = 0;
		for (size_t i = 0; i < l; i++)
		{
			sum[i] += res.second[i];
			double wt = exp(- eta * sum[i]);
			w[i] = wt;
			s += wt;
		}
		for (size_t i = 0; i < l; i++)
			w[i] /= s;
		// for (auto& x : res.first)
		// 	seeds_return.insert(x);
		seeds[j] = res.first;
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	// return seeds_return;
	return make_pair(seeds[rand() % T], (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
}

int main(int argc, char* argv[])
{
	if (argc < 8)
	{
		cerr << "Usage : ./<executable> <path-to-graph> <k> <d> <B> <l> <T> <path-to-hyperparameters>" << endl;
		return EXIT_FAILURE;
	}
	srand(time(NULL));
	string graph_path = string(argv[1]) + "graph_tmp.txt";
	string features_path = string(argv[1]) + "features_tmp.txt";
	k = atoi(argv[2]);
	d = atoi(argv[3]);
	B = atoi(argv[4]);
	l = atoi(argv[5]);
	T = atoi(argv[6]);
	nodes = unordered_set<size_t>();
	features = unordered_map<size_t,vector<double>>();
	rev_graph = unordered_map<size_t,unordered_set<size_t>>();
	hp = vector<vector<double>>(l, vector<double>(d));
	probs = unordered_map<size_t,unordered_map<size_t,unordered_map<size_t,double>>>();
	size_t u, v;
	ifstream ff(features_path);
	while (ff >> v)
	{
		nodes.insert(v);
		features[v] = vector<double>(d / 2);
		for (size_t i = 0; i < d / 2; i++)
			ff >> features[v][i];
	}
	ff.close();
	ifstream fg(graph_path);
	while (fg >> u >> v)
	{
		if (rev_graph.find(v) == rev_graph.end())
			rev_graph[v] = unordered_set<size_t>();
		rev_graph[v].insert(u);
	}
	fg.close();
	ifstream fh(argv[5]);
	for (size_t i = 0; i < l; i++)
		for (size_t j = 0; j < d; j++)
			fh >> hp[i][j];
		// generate(hp[i].begin(), hp[i].end(), [](){ return pow(-1, rand() % 2) * B * (double)rand() / RAND_MAX; });
	fh.close();
	auto res = hiro();
	for (size_t v : res.first)
		cout << v << endl;
	cout << "Running time : " << res.second << " seconds" << endl;
	return EXIT_SUCCESS;
}
