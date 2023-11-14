import numpy as np
import matplotlib.pyplot as plt


# *? plots: plot_gini_grid
# *? utils: gini_impurity, gini_impurity_avg, information_gain

def plot_gini_grid(gini_list):
        
        plt.figure(figsize=(10, 5))
        plt.suptitle('gini mapp')

        for k, v in gini_list.items():
            plt.plot(v[:, 0], v[:, 1], label=f'col index: {k}')

        plt.legend()
        plt.show();
        
        return
    
def gini_impurity(sub_set, y_classes):
    '''
                    k  
    gini(sample): 1 -  Σ [ (pi)**2 ]
                    (i=1)
    '''
    
    gini = 1
    sub_set_size = sub_set.shape[0]
    
    for cl in y_classes:
        
        cl_rows = sub_set[np.where(sub_set[:, -1] == cl), -1].shape[1]
        if cl_rows:
            gini -= (cl_rows / sub_set_size) ** 2
    
    return gini

def gini_impurity_avg(original_set, 
                        left_set_size, gini_left, 
                        right_set_size, gini_righ):
    '''
    gini(node): n1/n * ( gini_1 ) + n2/n * ( gini_2 )
    '''
    
    set_size = original_set.shape[0]
    gini_avg = (left_set_size/set_size) * (gini_left) + \
                (right_set_size/set_size) * (gini_righ)
    
    return gini_avg

def information_gain(parent_gini, child_gini, percentage=False):
    '''
    Δ gini(A) = gini(D) − giniA(D)
    '''
    delta_gini = parent_gini - child_gini
    
    if delta_gini == 0:
        return delta_gini
    
    else:
        if percentage:
            return (parent_gini - child_gini) / parent_gini
        else:
            return parent_gini - child_gini