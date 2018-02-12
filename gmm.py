"""
@filename: gmm.py
@author: james_rolfe
@updated: 20180212
@about: Generates random points, then using a gaussian mixture model to cluster them. 
		The clusters are translated to semi-connected graphs. x,y coors and cluster 
		membership and nodes/edges are writen to a csv.
"""

import networkx as nx
from numpy.random import normal, uniform
import pylab as py
from math import floor, sqrt
import matplotlib.pyplot as plt
import csv
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def gen_random_points(x_low, x_high, y_low, y_high, num_points):
	ret_group = []
	for i in range(num_points):
		x = uniform(x_low, x_high)
		y = uniform(y_low, y_high)
		ret_group.append((x,y))
	return ret_group

def gen_normal_points(x_mean, y_mean, x_std, y_std, num_points):
	ret_group = []
	for i in range(num_points):
		x = normal(x_mean, x_std)
		y = normal(y_mean, y_std)
		ret_group.append((x,y))
	return ret_group

def cull_data(data):
	r = [[x,y] for (x,y) in data 
			if x >= 0 and x <= 100 
			and y >= 0 and y <= 100]
	return r

def gen_normal_groups(num_groups, num_points):
	ret_data = []
	for i in range(num_groups):
		x_mean = uniform(0, 100)
		x_std = uniform(0, 25)
		y_mean = uniform(0, 100)
		y_std = uniform(0, 25)
		ret_data += gen_normal_points(x_mean, y_mean, x_std, y_std, num_points)
	return cull_data(ret_data)

def gen_random_groups(num_groups, num_points):
	ret_data = []
	for i in range(num_groups):
		x_low = uniform(0, 50)
		x_high = uniform(50, 100)
		y_low = uniform(0, 50)
		y_high = uniform(50, 100)
		ret_data += gen_random_points(x_low, x_high, y_low, y_high, num_points)
	return cull_data(ret_data)

def gmm_det(type, num_groups, num_points):
	gmm = GaussianMixture(n_components=num_groups)
	data = []
	if type == "normal":
		data = gen_normal_groups(num_groups, num_points)
	else:
		data = gen_random_groups(num_groups, num_points)
	gmm.fit(data)
	labels = gmm.predict(data)

	xs = [x for x,y in data]
	ys = [y for x,y in data]

	df = pd.DataFrame(dict(x=xs, y=ys, label=labels))
	groups = df.groupby('label')

	# Plot
	fig, ax = plt.subplots()
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=9, label=name)
	ax.legend()

	plt.show()
	return zip(xs, ys, labels)

def distance(p0, p1):
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def populate_edges(nx_G, dist_thres):
	for n_1, attr_1 in nx_G.nodes(data=True):
		nx_G.add_edges_from([(n_1, n_2) for n_2, attr_2 in nx_G.nodes(data=True)
			              if n_1 != n_2 
			              and (distance(attr_1['pos'], attr_2['pos']) <= dist_thres)])

def make_graph(data_tuples, dist_thres):
	ret_g = nx.Graph()
	for i,(x,y,label) in enumerate(data_tuples):
		ret_g.add_node(i, pos=(x,y), res=uniform(0,100), mem=label)
	populate_edges(ret_g, dist_thres)
	return ret_g

def build_data(nx_G):
	ret_data = []
	header = ["node", "x", "y", "res_val", "connections", "cluster_num"]
	ret_data.append(header)
	for n, attr in nx_G.nodes(data=True):
		tmp = [n, attr['pos'][0], attr['pos'][1], attr['res'], nx_G.edges(n), attr['mem']]
		ret_data.append(tmp)
	return ret_data

def write_to_csv(fname, data):
	tmp_f = open(fname, 'w')
	with tmp_f:
		writer = csv.writer(tmp_f)
		writer.writerows(data)

def save_g(nx_G, fname):
    json.dump(dict(nodes=[[n, nx_G.node[n]] for n in nx_G.nodes()],
                   edges=[[u, v, nx_G.edge[u][v]] for u,v in nx_G.edges()],
    			   attrs=[[n, attr['pos'], attr['res']] for n, attr in nx_G.nodes(data=True)]),
              open(fname, 'w'), indent=2)

def load_g(fname):
    nx_G = nx.Graph()
    d = json.load(open(fname))
    nx_G.add_nodes_from(d['nodes'])
    nx_G.add_edges_from(d['edges'])
    attr = d['attrs']
    return nx_G, attr

gmm_data = gmm_det("rand", 3, 100)
gmm_nxgraph = make_graph(gmm_data, 5)
gmm_graph_data = build_data(gmm_nxgraph)
write_to_csv("gmm_rand.csv", gmm_graph_data)

res_list = [attr['res'] for _,attr in gmm_nxgraph.nodes(data=True)]
nx.draw(gmm_nxgraph, cmap=plt.get_cmap('Blues'), node_color=res_list)
plt.show()
# print labels









