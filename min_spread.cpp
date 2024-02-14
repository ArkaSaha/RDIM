#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

size_t d, B, l;
unordered_map<size_t,vector<double>> features;
vector<unordered_map<size_t,unordered_map<size_t,double>>> probs;
vector<vector<double>> hp;
vector<double> score;
unordered_map<size_t,unordered_set<size_t>> graph;

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
	size_t N = 10000, nbrs = 0;
	unsigned int seed = time(NULL)^i;
	for (size_t j = 0; j < N; j++)
	{
		unordered_map<size_t,unordered_set<size_t>> adj = unordered_map<size_t,unordered_set<size_t>>();
		for (auto& e : graph)
		{
			size_t u = e.first;
			if (adj.find(u) == adj.end())
				adj[u] = unordered_set<size_t>();
			for (size_t v : e.second)
				if ((double)rand_r(&seed) / RAND_MAX < prob(u, v, i))
					adj[u].insert(v);
		}
		unordered_set<size_t> reached = unordered_set<size_t>();
		for (size_t v : seeds)
			bfs(v, adj, reached);
		nbrs += reached.size();
	}
	score[i] = (double) nbrs / N;
	return NULL;
}

double spread(unordered_set<size_t>& seeds)
{
	double min_spread = numeric_limits<double>::max();
	score = vector<double>(l);
	vector<pthread_t> tid = vector<pthread_t>(l);
	vector<pair<size_t,unordered_set<size_t>>> t = vector<pair<size_t,unordered_set<size_t>>>(l);
	for (size_t i = 0; i < l; i++)
	{
		t[i] = make_pair(i, seeds);
		if (pthread_create(&tid[i], NULL, &spread_helper, (void*)(&t[i])))
			cerr << "Thread creation error" << endl;
	}
	for (size_t i = 0; i < l; i++)
		pthread_join(tid[i], NULL);
	for (size_t i = 0; i < l; i++)
		if (min_spread > score[i])
			min_spread = score[i];
	return min_spread;
}

int main(int argc, char* argv[])
{
	if (argc < 7)
	{
		cerr << "Usage : ./<executable> <path-to-graph> <d> <B> <l> <path-to-hyperparameters> <paths-to-seeds>" << endl;
		return EXIT_FAILURE;
	}
	srand(time(NULL));
	string graph_path = string(argv[1]) + "graph_tmp.txt";
	string features_path = string(argv[1]) + "features_tmp.txt";
	d = atoi(argv[2]);
	B = atoi(argv[3]);
	l = atoi(argv[4]);
	features = unordered_map<size_t,vector<double>>();
	graph = unordered_map<size_t,unordered_set<size_t>>();
	hp = vector<vector<double>>(l, vector<double>(d));
	probs = vector<unordered_map<size_t,unordered_map<size_t,double>>>(l, unordered_map<size_t,unordered_map<size_t,double>>());
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
		if (graph.find(u) == graph.end())
			graph[u] = unordered_set<size_t>();
		graph[u].insert(v);
	}
	fg.close();
	ifstream fh(argv[5]);
	for (size_t i = 0; i < l; i++)
		for (size_t j = 0; j < d; j++)
			fh >> hp[i][j];
		// generate(hp[i].begin(), hp[i].end(), [](){ return pow(-1, rand() % 2) * B * (double)rand() / RAND_MAX; });
	fh.close();
	for (size_t i = 6; i < argc; i++)
	{
		unordered_set<size_t> seeds = unordered_set<size_t>();
		ifstream fs(argv[i]);
		while (fs >> v)
			seeds.insert(v);
		fs.close();
		cout << spread(seeds) << "\t";
	}
	cout << endl;
	return EXIT_SUCCESS;
}
