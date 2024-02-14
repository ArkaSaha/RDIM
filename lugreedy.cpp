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
#include <pthread.h>
#include <boost/heap/fibonacci_heap.hpp>

#define threads 20
#define N 10000

using namespace std;
using namespace boost::heap;

size_t k, d, B;
vector<size_t> vec, nbrs;
unordered_set<size_t> nodes;
unordered_map<size_t,vector<double>> features;
unordered_map<size_t,unordered_map<size_t,pair<double,double>>> probs;
unordered_map<size_t,unordered_set<size_t>> graph, rev_graph;
pthread_mutex_t m_lock;
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

double prob(size_t u, size_t v, bool low)
{
	pthread_mutex_lock(&m_lock);
	if (probs.find(u) == probs.end() or probs[u].find(v) == probs[u].end())
	{
		double p = 0;
		for (size_t i = 0; i < d / 2; i++)
			p += (B * (abs(features[u][i]) + abs(features[v][i])));
		probs[u][v] = make_pair(1 / (1 + exp(p)), 1 / (1 + exp(-p)));
	}
	pthread_mutex_unlock(&m_lock);
	return low ? probs[u][v].first : probs[u][v].second;
}

unordered_set<size_t> gen_rrset(size_t v, bool low)
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
			if (rr.find(x) == rr.end() and (double)rand() / RAND_MAX < prob(x, u, low))
				q.push(x);
	}
	return rr;
}

double logcnk(int n, int k)
{
	double ans = 0;
	for (int i = n - k + 1; i <= n; i++)
		ans += log(i);
	for (int i = 1; i <= k; i++)
		ans -= log(i);
	return ans;
}

pair<unordered_set<size_t>,double> cov(size_t theta, size_t n, bool low)
{
	unordered_map<size_t,unordered_set<size_t>> pres = unordered_map<size_t,unordered_set<size_t>>();
	for (size_t j = 0; j < theta; j++)
	{
		size_t u = vec[rand() % n];
		auto rr = gen_rrset(u, low);
		for (size_t v : rr)
		{
			if (pres.find(v) == pres.end())
				pres[v] = unordered_set<size_t>();
			pres[v].insert(j);
		}
	}
	fibonacci_heap<node,compare<compare_node>> heap = fibonacci_heap<node,compare<compare_node>>();
	unordered_map<size_t,handle_t> handles = unordered_map<size_t,handle_t>();
	for (auto& e : pres)
		handles[e.first] = heap.push(node(e.first, e.second.size()));
	unordered_set<size_t> seeds = unordered_set<size_t>();
	double score = 0;
	for (size_t j = 0; j < k; j++)
	{
		node nd = heap.top();
		heap.pop();
		size_t v = nd.vertex;
		handles.erase(v);
		seeds.insert(v);
		score += nd.gain;
		auto c = pres[v];
		pres.erase(v);
		if (heap.empty())
			break;
		for (auto& e : pres)
		{
			double diff = 0;
			for (size_t x : c)
				if (e.second.find(x) != e.second.end())
				{
					e.second.erase(x);
					diff++;
				}
			heap.update(handles[e.first], node(e.first, (*handles[e.first]).gain - diff));
		}
	}
	return make_pair(seeds, n * score / theta);
}

double opt(size_t n, double epsilon, bool low)
{
	double eps = epsilon * sqrt(2);
	for (size_t i = 0; i < log2((double)n / k); i++)
	{
		double x = n / pow(2, i);
		size_t theta = (2 + 2 * eps / 3) * n * (log(n) + logcnk(n, k) + log(log2((double)n / k))) / (eps * eps * x);
		double est = cov(theta, n, low).second;
		if (est >= x * (1 + eps))
			return est / (1 + eps);
	}
	return k;
}

unordered_set<size_t> greedy(bool low)
{
	double epsilon = 0.1;
	size_t n = nodes.size(), theta = 2 * n * pow((1 - exp(-1)) * sqrt(log(n) + log(2)) + sqrt((1 - exp(-1)) * (log(n) + log(2) + logcnk(n, k))), 2) / (pow(epsilon, 2) * opt(n, epsilon, low));
	return cov(theta, n, low).first;
}

void bfs(size_t u, unordered_map<size_t,unordered_set<size_t>>& adj, unordered_set<size_t>& reached)
{
	queue<size_t> q = queue<size_t>();
	q.push(u);
	while (not q.empty())
	{
		size_t v = q.front();
		q.pop();
		reached.insert(v);
		for (size_t w : adj[v])
			if (reached.find(w) == reached.end())
				q.push(w);
	}
}

void* spread_helper(void* args)
{
	auto& [i, seeds] = *((pair<size_t,unordered_set<size_t>>*)args);
	size_t samples = N / threads;
	unsigned int seed = time(NULL)^i;
	nbrs[i] = 0;
	for (size_t j = 0; j < samples; j++)
	{
		unordered_map<size_t,unordered_set<size_t>> adj = unordered_map<size_t,unordered_set<size_t>>();
		for (auto& e : graph)
		{
			size_t u = e.first;
			if (adj.find(u) == adj.end())
				adj[u] = unordered_set<size_t>();
			for (size_t v : e.second)
				if ((double)rand_r(&seed) / RAND_MAX < prob(u, v, true))
					adj[u].insert(v);
		}
		unordered_set<size_t> reached = unordered_set<size_t>();
		for (size_t v : seeds)
			bfs(v, adj, reached);
		nbrs[i] += reached.size();
	}
	return NULL;
}

double spread(unordered_set<size_t>& seeds)
{
	vector<pthread_t> tid = vector<pthread_t>(threads);
	vector<pair<size_t,unordered_set<size_t>>> t = vector<pair<size_t,unordered_set<size_t>>>(threads);
	for (size_t i = 0; i < threads; i++)
	{
		t[i] = make_pair(i, seeds);
		if (pthread_create(&tid[i], NULL, &spread_helper, (void*)(&t[i])))
			cerr << "Thread creation error" << endl;
	}
	for (size_t i = 0; i < threads; i++)
		pthread_join(tid[i], NULL);
	double s = 0;
	for (size_t i = 0; i < threads; i++)
		s += nbrs[i];
	return s / N;
}

pair<unordered_set<size_t>,double> lugreedy()
{
	vec = vector<size_t>(nodes.begin(), nodes.end());
	nbrs = vector<size_t>(threads);
	pthread_mutex_init(&m_lock, NULL);
	timespec begin, end;
	clock_gettime(CLOCK_MONOTONIC, &begin);
	unordered_set<size_t> lower = greedy(true), upper = greedy(false);
	double sl = spread(lower), su = spread(upper);
	clock_gettime(CLOCK_MONOTONIC, &end);
	pthread_mutex_destroy(&m_lock);
	return make_pair(sl < su ? upper : lower, (end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec) / pow(10, 9));
}

int main(int argc, char* argv[])
{
	if (argc < 5)
	{
		cerr << "Usage : ./<executable> <path-to-graph> <k> <d> <B>" << endl;
		return EXIT_FAILURE;
	}
	srand(time(NULL));
	string graph_path = string(argv[1]) + "graph_tmp.txt";
	string features_path = string(argv[1]) + "features_tmp.txt";
	k = atoi(argv[2]);
	d = atoi(argv[3]);
	B = atoi(argv[4]);
	nodes = unordered_set<size_t>();
	features = unordered_map<size_t,vector<double>>();
	graph = unordered_map<size_t,unordered_set<size_t>>();
	rev_graph = unordered_map<size_t,unordered_set<size_t>>();
	probs = unordered_map<size_t,unordered_map<size_t,pair<double,double>>>();
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
		if (graph.find(u) == graph.end())
			graph[u] = unordered_set<size_t>();
		graph[u].insert(v);
		if (rev_graph.find(v) == rev_graph.end())
			rev_graph[v] = unordered_set<size_t>();
		rev_graph[v].insert(u);
	}
	fg.close();
	auto res = lugreedy();
	for (size_t v : res.first)
		cout << v << endl;
	cout << "Running time : " << res.second << " seconds" << endl;
	return EXIT_SUCCESS;
}
