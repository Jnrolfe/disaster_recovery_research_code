"""
@filename: disaster_research_v1.py
@author: james_rolfe
@updated: 20180306
@about: 
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
from random import sample
import sys

class house():
	def __init__(self, name, x_mean, x_std, y_mean, y_std):
		self.name = name
		self.x = normal(x_mean, x_std)
		self.y = normal(y_mean, y_std)
                self.x_mean = x_mean
                self.y_mean = y_mean
                self.x_std = x_std
                self.y_std = y_std
		self.util = 0
                self.soc = 0
                self.dem = 0
		self.dam = 0
		self.group = 0

def gmm_det(graph, num_communitites):
	gmm = GaussianMixture(n_components=num_communitites)
	hs = graph.nodes()
	data = [(h.x, h.y) for h in hs]
	gmm.fit(data)
	labels = gmm.predict(data)

	for i,h in enumerate(hs):
	    h.group = int(labels[i]) + 1

	# Plotting
	xs = [x for x,y in data]
	ys = [y for x,y in data]

	df = pd.DataFrame(dict(x=xs, y=ys, label=labels))
	groups = df.groupby('label')
	print df

	fig, ax = plt.subplots()
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
	    ax.plot(group.x, group.y, marker='o', linestyle='', ms=9, label=name)
	ax.legend()

	plt.show()

	return

def make_attr_list(node):
	a1 = abs(normal(node.group, 1)) * 10
	return [a1]

def get_dem_resource(attr_list):
	return sum(attr_list)/len(attr_list)

def get_soc_resource(g, node, max_deg):
	n_deg = g.degree(node)
	return (float(n_deg) / max_deg) * 100

def set_utilities(graph, num_comm):
	max_deg = max([y for x,y in nx.degree(graph)])
	for h in graph.nodes():
		attr_list = [uniform((float(h.group)/num_comm), 1)*100]
		dem = get_dem_resource(attr_list)
		soc = get_soc_resource(graph, h, max_deg)
                h.soc = soc
                h.dem = dem
		h.util = (soc + dem)/2
	return 

def set_damage(graph):
	for h in graph.nodes():
		h.dam = (h.x + h.y)/2
	return

def distance(p0, p1):
	return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def norm_dist(p0, p1, max_dist):
	# normalized distance (observed/max)
    return distance(p0, p1)/max_dist

def connect_houses(graph, dist_thres, max_dist, num_connects):
	houses = graph.nodes()
	max_iters = len(houses)
	iters = 0
	conns = 0
	
	while(conns < num_connects):
		samp = sample(houses, 2)
		h1, h2 = samp[0], samp[1]
		if norm_dist((h1.x, h1.y), (h2.x, h2.y), max_dist) < dist_thres:
			graph.add_edge(h1, h2)
			conns += 1
		if iters == max_iters:
			print "WARNING: max iterations reached when adding connections"
			print "# connections requested: " + str(num_connects)
			print "# connections reached: " + str(conns)
			print "Doubling distance threshold to: " + str(dist_thres*2)
			iters = 0

	return

def build_data(graph):
	ret_data = []
	header = ["house", "x", "y", "dem", "soc", "dam", "group", "connections", "x_mean", "x_std", "y_mean", "y_std"]
	hs = graph.nodes()
	for h in hs:
		tmp = [h.name, h.x, h.y, h.dem, h.soc, h.dam, h.group, 
			  [(a.name, b.name) for a, b in graph.edges(h)],
                          h.x_mean, h.x_std, h.y_mean, h.y_std]
		ret_data.append(tmp)
        ret_data = sorted(ret_data, key=lambda x: x[0])
        ret_data.insert(0, header)
        return ret_data 

def write_to_csv(fname, data):
	tmp_f = open(fname, 'w')
	with tmp_f:
		writer = csv.writer(tmp_f)
		writer.writerows(data)

def make_communities(num_communitites, num_houses, num_connects, dist_thres):
	houses = []
        hcount = 1

	for _ in xrange(num_communitites):
		x_mean = uniform(0, 100)
		x_std = uniform(0, x_mean)
		y_mean = uniform(0, 100)
		y_std = uniform(0, y_mean)
		for _ in xrange(num_houses):
			h = house(hcount, x_mean, x_std, y_mean, y_std)
			if (h.x > 100) or (h.x < 0) or (h.y > 100) or (h.y < 0):
				continue # exclude from data
			houses.append(h)
                        hcount += 1

	n = len(houses)
	if num_connects > (0.5*n*(n-1)):
		print "ERROR: Number of connections is too large"
		print "# connections requested: " + str(num_connects)
		print "# connections possible: " + str(0.5*n*(n-1))
		sys.exit(1)

	g = nx.Graph()
	g.add_nodes_from(houses)

	max_dist = max([distance((0, 0), (h.x, h.y)) for h in houses])
	connect_houses(g, dist_thres, max_dist, num_connects)

	gmm_det(g, num_communitites)

	set_utilities(g, num_communitites)
	set_damage(g)

	return g

NUM_GROUPS = 5
NUM_HOUSES_PER_GROUP = 100
NUM_CONNS = 10000
DIST_THRES = 0.05
FNAME = "data_v1_20180306.csv"

g = make_communities(NUM_GROUPS, NUM_HOUSES_PER_GROUP, NUM_CONNS, DIST_THRES)
write_to_csv(FNAME, build_data(g))

# util_list = [h.util for h in g.nodes()]
# nx.draw(g, cmap=plt.get_cmap('Blues'), node_color=util_list)
# plt.show()
