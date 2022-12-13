# Packages
from collections import defaultdict
from itertools import product, combinations, groupby
import networkx as nx
import pandas as pd
import numpy as np
import math
import copy
import random
import itertools
from copy import deepcopy
from BayesNet import BayesNet
from typing import List, Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt
from pgmpy.readwrite import XMLBIFReader

# BNReasoner Class
class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    def network_pruning(self, Query: List[str], Evidence: pd.Series) -> bool:
        """
        Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated.
        """
        # # First create a deepcopy of the structure of BN
        # self.bn = BayesNet()
        bn_copy = deepcopy(self.bn) 

        # Combine the Query and Evidence states
        combined_states = set(Query) | set(Evidence.index)

        # Node-Pruning: Only keep the leaves that are in either the Query or Evidence & repeat as often as possible
        while True:
            count = 0

            # Prune leaf node for every node that is not in combined_states
            for leafnode in set(bn_copy.get_all_variables()) - combined_states:
                if len(bn_copy.get_children(leafnode)) == 0:
                    bn_copy.del_var(leafnode)
                    count += 1

            if count == 0:
                break

        # Edge-pruning: Only keep the outgoing edges that are not in the Evidence 
        for edgenode in Evidence.index:
            children = bn_copy.get_children(edgenode)
            for child in children:
                bn_copy.del_edge((edgenode, child))

                # Simplify CPTs
                oldcpt = bn_copy.get_cpt(child)
                newcpt = bn_copy.reduce_factor(instantiation = Evidence, cpt = oldcpt)
                bn_copy.update_cpt(child, newcpt)

        # Return the pruned BN
        return bn_copy


    def d_separation(self, X: List[str], Z: List[str], Y: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z.
        """
        # Copy the graph
        graph_copy = deepcopy(self)

        # Node-Pruning: Delete all nodes that are not in the Query: X, Y, or in the Evidence: Z
        while True:
            count = 0

            # Prune leaf node for every node that is not in Query or Evidence
            for leaf in set(graph_copy.bn.get_all_variables()) - set(X + Y + Z):
                if len(graph_copy.bn.get_children(leaf)) == 0:
                    graph_copy.bn.del_var(leaf)
                    count += 1

            if count == 0:
                break

        # Edge-Pruning: Delete all outgoing edges from Evidence: Z
        for edgenode in Z:
            children = graph_copy.bn.get_children(edgenode)
            for child in children:
                graph_copy.bn.del_edge((edgenode, child))

        # For every node in X & Y: Check if there is a connection. If yes: X and Y are not d-separated by Z.
        for x, y in itertools.product(X, Y):
            if nx.has_path(nx.to_undirected(graph_copy.bn.structure), x, y):    # Undirected graph is turned into a directed graph
                return False

        return True

    def independence(self, X: List[str], Z: List[str], Y: List[str]) -> bool:
        """
        Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. 
        Assumption: BN is faithful; A BN is faithful iff for any set of variables X, Y, Z: X independent of Y given Z => X d-separated Y given Z
        """
        # Check if X and Y are d-separated by Z
        if self.d_separation(X, Z, Y):
            return True

        return False

    @staticmethod
    def marginalization(cpt: pd.DataFrame, variable: list[str]):
        """
        Given a factor and a variable X, compute the CPT in which X is summed-out.
        """
        # Drop the variable(s) in question
        dropped = cpt.drop(variable, axis=1)

        # Group by the remaining variables and sum over the dropped variable(s)
        newcpt_vars = [v for v in cpt.columns.tolist()[:-1] if v not in variable]
        newcpt = dropped.groupby(newcpt_vars).sum().reset_index()

        return newcpt

    def maxing_out(self, CPT, X):
        """
        Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember to also keep track 
        of which instantiation of X led to the maximized value.
        """
        maximized_out = CPT.groupby(X).max().reset_index()
        return X, maximized_out
    

    def factor_multiplication(self, variables):
        """
        Given two factors f and g, compute the multiplied factor h=fg.
        """

        # variables is a list of dataframes of cpts
        # first item in variables is the dataframe you want to multiply with the other dataframes

        # set the threshold for the while loop (the length of variables - the first dataframe)
        thresh = len(variables) - 1
        i = 1

        # loop over the variables (dataframes of cpts)
        while thresh != 0:
            thresh -= 1
            factor_2 = variables[i]
            i += 1

            # create two lists of the columns of the two cpts you want to multiply, but exclude the last column (p)
            list_factor2 = factor_2.columns.values.tolist()[0:-1]
            list_factor1 = variables[0].columns.values.tolist()[0:-1]

            # check on what variable the lists match and append those to a new list
            overlapping_elements = []
            for element in list_factor2:
                if element in list_factor1:
                    overlapping_elements.append(element)
            
            # only multiply if there is overlap
            if len(overlapping_elements) > 0:

                # merg the two cpts based on the matching elements
                # this creates two new columns (p_x and p_y) with the probabilities of the two cpts
                new_table = pd.merge(factor_2, variables[0], how='left', on=overlapping_elements)

                # compute the new probabilities by multiplying the p columns of the two cpts
                new_table['p'] = (new_table['p_x'] * new_table['p_y'])

                # drop the newly created p_x and p_y columns
                new_table.drop(['p_x', 'p_y'],inplace=True, axis = 1)
                    
                # replace the old probabilities with the new one
                variables[0] = new_table
        
        return variables[0]

    def order(self, N: Optional[List[str]] = None, heuristic: str = "mindegree", order_asc: bool = True,) -> List[str]:
        """
        Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X 
        based on the min-degree heuristics (2pts) and the min-fill heuristics (3.5pts). 
        """
        # Get all nodes and get the converted graph from directed to undirected graph 
        nodes = self.bn.get_all_variables()
        graph = self.bn.structure.to_undirected()

        # Check if heuristic that is given is correct
        heuristics = ["random", "mindegree", "minfill"]
        assert heuristic in heuristics, f"Heuristic given must be in {heuristics}"

        # This function depicts the heuristic that is utilized to order the nodes
        order_heuristic = getattr(self, f"_{heuristic}_heuristic")

        # If N is None, and we can't find any nodes, we want all nodes
        if N is None:
            N = nodes.copy()

        # Dict where the nodes are keys and the values are its neighboring nodes 
        d_neighbors = self._neighbour(graph)

        # Drop the unrelevant nodes from d_neighbors
        for node in [node for node in nodes if node not in N]:
            self._rm_neighbour(d_neighbors, node)

        # If we want to order low -> high we want the minimum, and if we want to order high -> low we want the maximum 
        selection = min if order_asc else max

        # Create an empty list for the ordering
        ordering = []

        # Loop over the length of dictionary d_neighbors
        for _ in range(len(d_neighbors)):

            # Select the node to eliminate from G based on heuristic
            node_elim = selection(d_neighbors, key=lambda x: order_heuristic(d_neighbors, x))

            # Remove node to eliminate from the graph
            self._rm_adj(d_neighbors, node_elim)

            # Add node to eliminate to ordering
            ordering.append(node_elim)

        return ordering
    
    # Mindegree ordering is based on a random number between 0 and 1
    @staticmethod
    def _random_heuristic(*_) -> float:
        return random.uniform(0, 1)

    # Mindegree ordering is based on the number of neighbors a node has
    @staticmethod
    def _mindegree_heuristic(neighbouring, node) -> int:
        return len(neighbouring[node])

    # Minfill ordering is based on the number of edges that are added when a certain node is removed    ## Minfill is the amount of edges that need to be added to remove a node
    @staticmethod
    def _minfill_heuristic(neighbouring: Dict[str, set], node: str) -> int:
        edge_counter = 0
        for i, j in itertools.combinations(neighbouring[node], 2):
            if i not in neighbouring[j]:
                edge_counter += 1

        return edge_counter

    # Create a dictionary where the keys are the nodes, and the values a set of neighbouring nodes 
    @staticmethod
    def _neighbour(graph: nx.classes.graph.Graph) -> Dict[str, set]:
        return {v: set(graph[v]) - set([v]) for v in graph}

    # Remove a specific variable after which edges are added between the neighbours of the variable node
    @staticmethod
    def _rm_neighbour(neighbour: Dict[str, set], node: str) -> None:
        neighbors = neighbour[node]

        # Make the edges between the neighbours of the variable node
        for n1, n2 in itertools.combinations(neighbors, 2):
            if n2 not in neighbour[n1]:
                neighbour[n1].add(n2)
                neighbour[n2].add(n1)

        # Delete the node from the dictionary for the neighbours
        for n2 in neighbors:
            neighbour[n2].discard(node)

        # Delete the node from the dictionary neighbour dictionary
        del neighbour[node]

    def variable_elimination(self, CPT, X):
        """
        Sum out a set of variables by using variable elimination.
        """
        for lbl in X:
            CPT=CPT.drop(columns=lbl)
        X_dropped = list(CPT.columns)
        X_dropped.pop(-1)
        CPT = CPT.groupby(X_dropped).sum().reset_index()
        return CPT
    

    def md_MAP_MPE(self, Q, evidence, func):

        # Q = list of variables (e.g. ['light-on']), but can be empty in case of MPE
        # evidence = a dictionary of the evidence e.g. {'hear-bark': True} or empty {}
        # posterior marginal: P(Q|evidence) / P(evidence)
        # MAP: sum out V/Q and then max-out Q (argmax)
        # MPE: maximize out all variables with extended factors

        variables = self.ordering(Q)
        
        
        # prune the network given the evidence (# reduce all the factors w.r.t. evidence)
        self.network_pruning(Q,pd.Series(evidence)) # this function is not yet implemented #Doruk changed it

        # compute the probability of the evidence
        e_factor = 1
        for e in evidence:
            evidence_probability = self.bn.get_cpt(e)
            e_factor = e_factor * self.bn.get_cpt(e)['p'].sum()
        
        # retrieve all the cpts and delete them accordingly
        M = self.bn.get_all_cpts()

        factor = 0

        # loop over every variable which is not in Q and create an empty dictionary
        for v in variables:
            f_v = {}
            
            # loop over every cpt and check if the variable is in the cpt and if so, add it to the dictionary
            for cpt_v in M: 
                if v in M[cpt_v]:
                    f_v[cpt_v] = M[cpt_v]
            
            # sum-out Q to obtain probability of evidence and to elimate the variables
            # only multiply when there are more than one cpt
            if len(f_v) >= 2:
                m_cpt = self.multiplying(list(f_v.values()))
                new_cpt = self.Variable_el(m_cpt, [v])           

                # delete the variables from the dictionary M
                for f in f_v:
                    del M[f]
                
                # add the new cpt to the dictionary M
                factor +=1
                M["F"+str(factor)] = new_cpt
            
            # skip multiplication when there is only one cpt
            elif len(f_v) == 1:
                new_cpt = self.Variable_el(list(f_v.values())[0], [v])
                
                # delete the variables from the dictionary M
                for f in f_v:
                    del M[f]
                
                # add the new cpt to the dictionary M
                factor +=1
                M["F "+str(factor)] = new_cpt

        # compute joint probability of Q and evidence
        if len(M) > 1:
            input = list(M.values())
            joint_prob = self.multiply_factors(input[0],input[1])
        else:
            joint_prob = list(M.values())[0]
        
        # divide by the probability of the evidence
        joint_prob['p'] = joint_prob['p'] / (e_factor)

        # check what is expected of the function
        if func == 'marginal':
            return joint_prob
        if func == 'MAP':
            return joint_prob.iloc[joint_prob['p'].argmax()]
        if func == 'MPE':
            return joint_prob.iloc[joint_prob['p'].astype(float).argmax()]
        else:
            return joint_prob

    
if __name__ == "__main__":
    net = BNReasoner("testing/dog_problem.BIFXML")
    test = net.md_MAP_MPE(['light-on'], {'hear-bark': True}, "marginal", "random")
    print(test)     
    
