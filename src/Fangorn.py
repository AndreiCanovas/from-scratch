import queue
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Ginasty import gini_impurity, gini_impurity_avg, \
                    information_gain, plot_gini_grid 


# *TODO: create hyperparams options

# *? plots: plot_node_info, hierarchy_pos, plot_decision_tree
# *? utils: get_col_sample, get_col_possible_splits, make_tree_split
# *?        get_best_split, create_node, create_leaf, make_tree
# *?        make_tree_as_graph, make_prediciton

# tree plots:
def plot_node_info(tree, node_id, edge_list):
    '''
    n as tree[node_id]
    '''

    try:        
        n   = tree[node_id]
        yes = n['chils_branches'][[v for u, v in edge_list if u == node_id][0]]
        no  = n['chils_branches'][[v for u, v in edge_list if u == node_id][1]]

        print(f'''
        ~ node {node_id}:
        
        input_sample_size: {n["input_sample"].shape}   | gini_impurity = {round(n["input_gini"], 3)}
        |
        •
        question: dim {n["split_info"]["col_index"]} >= {round(n["split_info"]["col_value"], 2)}
        |
        ├── yes: sample size={yes["sample"].shape} | gini_impurity = {round(yes["gini"], 3)} 
        └── no:  sample size={no["sample"].shape}  | gini_impurity = {round(no["gini"], 3)}

        avg-gini: {round(n["split_info"]["gini"], 3)} | information-gain: {round(n["information_gain"], 3)}
        ''')
        
    except:
        print(f'''
        node {node_id} is a leaf node!
        ''')
    
    return

def hierarchy_pos(G, root=None, width=1., 
                  vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_decision_tree(G):
    
    plt.figure(figsize=(15, 8))
    plt.axis("off")
    plt.tight_layout()

    pos = hierarchy_pos(G)

    branchs = [n[0] for n in G.nodes(data=True) if n[1]['node_type'] == 'branch']
    nx.draw_networkx_nodes(G, pos, nodelist=branchs)

    leafs   = [n[0] for n in G.nodes(data=True) if n[1]['node_type'] != 'branch']
    nx.draw_networkx_nodes(G, pos, nodelist=leafs, node_color='C2', node_shape='s')

    nx.draw_networkx_labels(G, pos)
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        if d['node_type'] == 'branch':
            text = f"{d['split_info']['col_index']} >= {d['split_info']['col_value']}"
            plt.text(x, y+0.05, s=text, bbox=dict(facecolor='white', alpha=0.5), horizontalalignment='center')
        elif d['node_type'] == 'leaf':
            text = f"{d['decision']['class']}"
            plt.text(x, y-0.05, s=text, bbox=dict(facecolor='C2', alpha=0.5), horizontalalignment='center')
            

    nx.draw_networkx_edges(G, pos)
    edge_gini = {(u, v): round(G.nodes[u]['chils_branches'][v]['gini'], 2) for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_gini)

    plt.show()
    
    return 

# tree utils
def sample_preprocess(X, y):
    pass

def get_col_sample(sample, drop_cols):
    
    # select how many columns of X will be used by the tree:
    cols_size  = round((sample.shape[1] - 1) * (1 - drop_cols))

    # get a random dimension array from available X variables:
    remaining_cols = np.random.choice(range(sample.shape[1] - 1), 
                                      cols_size, 
                                      replace=False)
    return remaining_cols
    
def get_col_possible_splits(sample, column_index):
    
    # create n possible splits for a node question:
    split_values_list = []
    unique_col_values = np.unique(sample[:, column_index])

    col_type = str(type(sample[:, column_index][0]))

    if 'float' in col_type or 'int' in col_type:

        for i, v in enumerate(unique_col_values):
            if i != len(unique_col_values) - 1:
                split_values_list.append((v + unique_col_values[i + 1]) / 2)    

    else:
        unique_col_values = split_values_list 
        
    return unique_col_values

def make_tree_split(sample, y_classes, column_index, split_value):
    
    yes_branch = sample[np.where((sample[:, column_index] >= split_value))]
    no_branch  = sample[np.where(~(sample[:, column_index] >= split_value))]

    yes_branch_gini = gini_impurity(yes_branch, y_classes)
    no_branch_gini  = gini_impurity(no_branch, y_classes)

    return yes_branch, yes_branch_gini, no_branch, no_branch_gini

def get_best_split(sample, y_classes, columns_index, plot_output=True):
    
    gini_list = {}

    for col_index in columns_index:

        split_values_list = get_col_possible_splits(sample, col_index)

        gini_range = []

        for split_value in split_values_list:

            yes_set, yes_set_gini, no_set, no_set_gini = make_tree_split(sample, 
                                                                         y_classes,
                                                                         col_index, 
                                                                         split_value)
            gini = gini_impurity_avg(sample, 
                                     yes_set.shape[0], yes_set_gini, 
                                     no_set.shape[0], no_set_gini)
            

            gini_range.append(gini)

        gini_list[col_index] = np.concatenate((split_values_list[:, np.newaxis], 
                                               np.array(gini_range)[:, np.newaxis]), axis=1)
        
    best_col = None
    best_split_value = None
    best_gini = 1

    for k, v in gini_list.items():

        if v[v[:, 1] == v[:, 1].min()][0][1] < best_gini:

            best_col         = k
            best_split_value = v[v[:, 1] == v[:, 1].min()][0][0]
            best_gini        = v[v[:, 1] == v[:, 1].min()][0][1]
        
    if plot_output:
        plot_gini_grid(gini_list)
    
    return {'col_index': best_col, 
            'col_value': best_split_value, 
            'gini': best_gini}, gini_list

def create_node(node_reference, sample, y_classes, available_columns, plot_output):
    
    # calculate gini impurity before split - with the input sample: 
    input_gini = gini_impurity(sample, y_classes)

    # get from all available columns the best column to split and it's value:    
    split_info, _ = get_best_split(sample, y_classes, available_columns, plot_output=plot_output)

    # get the result datasets from each branch after split
    yes_set, yes_set_gini, no_set, no_set_gini = make_tree_split(sample, 
                                                                 y_classes,
                                                                 split_info['col_index'], 
                                                                 split_info['col_value'])
    
    # average gini from both branches
    output_gini = gini_impurity_avg(sample, 
                                    yes_set.shape[0], yes_set_gini, 
                                    no_set.shape[0], no_set_gini)
    
    # calculate the information gain by the difference between the 
    #gini before vs gini after split
    tree_gain = information_gain(input_gini, output_gini, percentage=True)
    
    # create tree dict structure
    tree_dict = {'input_sample': sample, 
                 'input_gini': input_gini,
                 'split_info': split_info, 
                 'information_gain': tree_gain,
                 'node_type': 'branch',
                 'chils_branches': {(node_reference + 1): {'sample': yes_set,
                                                           'gini': yes_set_gini},
                                    (node_reference + 2): {'sample': no_set,
                                                           'gini': no_set_gini}
                                   }
                }
    
    return tree_dict

def create_leaf(sample, y_classes):
    
    # calculate gini impurity before split - with the input sample: 
    leaf_gini = gini_impurity(sample, y_classes)
    
    # decision into a leaf node:
    leaf_sample = sample
    decision = sorted([(c, leaf_sample[leaf_sample[:, -1] == c, :].shape[0] / leaf_sample.shape[0]) for c in y_classes], 
                      key=lambda x: x[1], 
                      reverse=True)[0]

    decision = {'class': decision[0], 
                'proba': decision[1]}
    
    # create tree dict structure
    tree_dict = {'input_sample': sample, 
                 'input_gini': leaf_gini,
                 'node_type': 'leaf',
                 'decision': decision}
    
    return tree_dict
    
def make_tree(data, y_classes, available_columns, plot_output):
    
    tree = {}
    root_node = 0
    node_ref = root_node

    nodes_to_run = queue.Queue()
    nodes_to_run.put(node_ref)

    while True:
    
        if nodes_to_run.empty():
            break
        else:
            node = nodes_to_run.get()    
        
        if node == root_node:
            sample = data
        else:
            sample = [v['chils_branches'][k_]['sample'] for k, v in tree.items() 
                                                            if v['node_type'] == 'branch'
                                                                for k_ in v['chils_branches'] 
                                                                    if k_ == node][0]    


        # jump empty samples / nodes:
        if sample.size:
            
            node_tmp = create_node(node_ref, sample, y_classes, available_columns, plot_output)
                
            if node_tmp["information_gain"] > .0:
                
                tree[node] = node_tmp.copy()
                
                if plot_output:
                    print(f'node: {node} | inform. gain: {tree[node]["information_gain"]}')
                    for k, v in tree[node]['chils_branches'].items():
                        print(np.unique(v['sample'][:, -1], return_counts=True))

                for k in tree[node]['chils_branches']:
                    nodes_to_run.put(k)

                tree_nodes = [root_node] + [k_ for k, v in tree.items() if v['node_type'] == 'branch' 
                                                                            for k_ in v['chils_branches']]
                node_ref = max(tree_nodes)
                
            else:
                
                tree[node] = create_leaf(sample, y_classes)
                            
    return tree

def make_tree_as_graph(tree):
    
    tre_edge_list = [(k, k_) for k, parent_node in tree.items() 
                                     if parent_node['node_type'] == 'branch'
                                         for k_, child_node in parent_node['chils_branches'].items()
                                             if child_node['sample'].size]

    g = nx.DiGraph(tre_edge_list)

    for node in g.nodes:
        nx.set_node_attributes(g, tree)    
    
    plot_decision_tree(g)

    # for n in g.nodes:
    #     plot_node_info(n, g.edges)
    
    return g

def make_prediciton(tree, new_samples):
    
    def _make_one_prediction(tree, new_sample, node):

        if tree[node]['node_type'] == 'branch':

            col_to_ask   = tree[node]['split_info']['col_index']
            value_to_ask = tree[node]['split_info']['col_value']

            if new_sample[col_to_ask] >= value_to_ask:
                next_node = list(tree[node]['chils_branches'])[0]

            else:
                next_node = list(tree[node]['chils_branches'])[1]

            return _make_one_prediction(tree, new_sample, next_node)

        else:
            return tree[node]['decision']['class'], \
                   tree[node]['decision']['proba']
        
    
    predictions = []
    for row in range(new_samples.shape[0]):

        new_sample = new_samples[row, :]
        
        prediciton = _make_one_prediction(tree, new_sample, 0)
        
        predictions.append(prediciton)
        
    return np.array(predictions)