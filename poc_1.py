import networkx as nx
from numpy.random import normal, uniform
import pylab as py
from math import floor, sqrt
import matplotlib.pyplot as plt
import csv
from random import randint

NODE_COUNT = 0

def distance(p0, p1):
    return sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def fix_point(x):
	if x > 1:
		x = 2 - x
	elif x < -1:
		x = -2 - x
	return x

def gen_normal_points(x_mean, y_mean, x_std, y_std, num_points):
	ret_group = []
	for i in range(num_points):
		x = fix_point(normal(x_mean, x_std))
		y = fix_point(normal(y_mean, y_std))
		ret_group.append((x,y))
	return ret_group

def gen_random_points(x_low, x_high, y_low, y_high, num_points):
	ret_group = []
	for i in range(num_points):
		x = fix_point(uniform(x_low, x_high))
		y = fix_point(uniform(y_low, y_high))
		ret_group.append((x,y))
	return ret_group

def populate_graph(nx_G, coor_list, nc, res_tup, num_c):
	for coor in coor_list:
		nx_G.add_node(nc, pos=coor, res=uniform(res_tup[0], res_tup[1]),
                        mem=num_c)
		nc += 1
	return nc

def populate_graph_random(nx_G, coor_list, nc, res_tup, num_groups):
	for coor in coor_list:
		num_c = randint(1, num_groups)
		nx_G.add_node(nc, pos=coor, res=uniform(res_tup[0], res_tup[1]),
                        mem=num_c)
		nc += 1
	return nc

def populate_edges(nx_G, dist_thres):
	for n_1, attr_1 in nx_G.nodes(data=True):
		G.add_edges_from([(n_1, n_2) for n_2, attr_2 in nx_G.nodes(data=True)
			              if n_1 != n_2 
			              and (distance(attr_1['pos'], attr_2['pos']) <= dist_thres)])

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

####################################

G = nx.Graph()

group_1 = gen_random_points(-1, 1, -1, 1, 250)
group_2 = gen_random_points(-1, 1, -1, 1, 250)
group_3 = gen_random_points(-1, 1, -1, 1, 250)
group_4 = gen_random_points(-1, 1, -1, 1, 250)

NODE_COUNT = populate_graph_random(G, group_1, NODE_COUNT, (0.0, 0.25), 4)
NODE_COUNT = populate_graph_random(G, group_2, NODE_COUNT, (0.25, 0.5), 4)
NODE_COUNT = populate_graph_random(G, group_3, NODE_COUNT, (0.5, 0.75), 4)
NODE_COUNT = populate_graph_random(G, group_4, NODE_COUNT, (0.75, 1), 4)

# group_11 = gen_normal_points(1, 1, 0.3, 0.25, 25)
# group_01 = gen_normal_points(-1, 1, 0.3, 0.25, 25)
# group_10 = gen_normal_points(1, -1, 0.3, 0.25, 25)
# group_00 = gen_normal_points(-1, -1, 0.3, 0.25, 25)

# NODE_COUNT = populate_graph(G, group_11, NODE_COUNT, (0.75, 1.0), 0)
# NODE_COUNT = populate_graph(G, group_01, NODE_COUNT, (0.5, 0.75), 1)
# NODE_COUNT = populate_graph(G, group_10, NODE_COUNT, (0.25, 0.5), 2)
# NODE_COUNT = populate_graph(G, group_00, NODE_COUNT, (0.0, 0.25), 3)



populate_edges(G, 0.10)

res_list = [attr['res'] for _,attr in G.nodes(data=True)]

nx.draw(G, cmap=plt.get_cmap('Blues'), node_color=res_list)
plt.show()

write_to_csv("poc_2.csv", build_data(G))

# save_g(G, "temp.json")
