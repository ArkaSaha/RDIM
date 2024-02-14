from os import system, path
import random, sys

data = sys.argv[1]
k = int(sys.argv[2])
d = int(sys.argv[3])
B = int(sys.argv[4])
l = int(sys.argv[5])
T = int(sys.argv[6])

theta = []
file_theta = path.join(data, 'theta.txt')
if path.isFile(file_theta):
	with open(file_theta, 'r') as f:
		for line in f:
			theta.append(float(line[:-1]))
if not theta:
	theta = [0 for _ in range(d)]

random.seed()
with open('hp.txt', 'w') as f:
	for _ in range(l):
		for i in range(d):
			f.write('{}\n'.format(theta[i] + random.random() * B * (-1) ** random.randint(0, 1)))

system('rm -f comp_* rob_dyn_im output.log && g++ -O3 -std=c++20 -lpthread -o rob_dyn_im rob_dyn_im_{}.cpp && ./rob_dyn_im {}/ {} {} {} {} {} hp.txt > output.log'.format(sys.argv[7], data, k, d, B, l, T))
system('cp {var}/graph.txt {var}/graph_tmp.txt && cp {var}/features.txt {var}/features_tmp.txt'.format(var=data))
with open('output.log', 'r') as fin:
	times = [0, 0, 0, 0]
	res = [0, 0, 0]
	for line in fin:
		if line[0].isdigit():
			ll = line.strip().split()
			if len(ll) == 4:
				if ll[-1] == 'insert':
					with open('{}/graph_tmp.txt'.format(data), 'a') as fout:
						fout.write('{} {}\n'.format(ll[1], ll[2]))
				else:
					with open('{}/graph_tmp.txt'.format(data), 'r') as f1, open('{}/graph_new.txt'.format(data), 'w') as f2:
						for line in f1:
							if line[:-1] != ll[1] + ' ' + ll[2]:
								f2.write(line)
					system('mv {}/graph_new.txt {}/graph_tmp.txt'.format(data, data))
			elif len(ll) == 3:
				if ll[-1] == 'insert':
					u = ll[1]
				else:
					with open('{}/features_tmp.txt'.format(data), 'r') as f1, open('{}/features_new.txt'.format(data), 'w') as f2:
						for line in f1:
							if line.split()[0] != ll[1]:
								f2.write(line)
					system('mv {}/features_new.txt {}/features_tmp.txt'.format(data, data))
		elif line.startswith('Features'):
			with open('{}/features_tmp.txt'.format(data), 'a') as fout:
				fout.write('{}\t{}'.format(u, line[len('Features : '):]))
		elif line.startswith('Current seed set'):
			seeds = line[len('Current seed set : '):].strip().split()
			with open('seeds_robdyn.txt', 'w') as fout:
				for v in seeds:
					fout.write('{}\n'.format(v))
			system('rm -f lugreedy && g++ -O3 -std=c++20 -lpthread -o lugreedy lugreedy.cpp && ./lugreedy {}/ {} {} {} > seeds_lugreedy.txt'.format(data, k, d, B))
			with open('seeds_lugreedy.txt', 'r') as f:
				for li in f:
					if li.startswith('Running time'):
						res[0] = float(li[len('Running time : '):-len(' seconds ')])
			system('rm -f base && g++ -O3 -std=c++20 -o base base.cpp && ./base {}/ {} {} {} {} {} hp.txt > seeds_base.txt'.format(data, k, d, B, l, T))
			with open('seeds_base.txt', 'r') as f:
				for li in f:
					if li.startswith('Running time'):
						res[1] = float(li[len('Running time : '):-len(' seconds ')])
			system('rm -f hiro && g++ -O3 -std=c++20 -o hiro hiro.cpp && ./hiro {}/ {} {} {} {} {} hp.txt > seeds_hiro.txt'.format(data, k, d, B, l, T))
			with open('seeds_hiro.txt', 'r') as f:
				for li in f:
					if li.startswith('Running time'):
						res[2] = float(li[len('Running time : '):-len(' seconds ')])
			for i in range(3):
				times[i + 1] += res[i]
		elif line.startswith('Running time'):
			times[0] += float(line[len('Running time : '):-len(' seconds ')])
			with open('comp_time.txt', 'a') as fout:
				fout.write('\t'.join(map(str, times)) + '\n')
			system('rm -f min_spread && g++ -O3 -std=c++20 -lpthread -o min_spread min_spread.cpp && ./min_spread {}/ {} {} {} hp.txt seeds_robdyn.txt seeds_lugreedy.txt seeds_base.txt seeds_hiro.txt >> comp_score.txt && rm -f seeds_*'.format(data, d, B, l))
system('rm -f {var}/graph_tmp.txt {var}/features_tmp.txt'.format(var=data))
