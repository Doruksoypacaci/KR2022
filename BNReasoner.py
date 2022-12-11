# Packages
import pandas as pd
import math
import itertools
from copy import deepcopy
from BayesNet import BayesNet
from typing import List, Tuple, Dict, Union
import networkx as nx
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

    # TODO: This is where your methods should go
    # Network Pruning: Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated. (3.5pts)
    # d-Separation: Given three sets of variables X, Y, and Z, determine whether X is d-separated of Y given Z. (4pts)
    # Independence: Given three sets of variables X, Y, and Z, determine whether X is independent of Y given Z. (Hint: Remember the connection between d-separation and independence) (1.5pt) EDIT: For this part, please assume that the BN is faithful. A BN is faithful iff for any set of variables X, Y, Z: X independent of Y given Z => X d-separated Y given Z
    # Marginalization: Given a factor and a variable X, compute the CPT in which X is summed-out. (3pts)
    # Maxing-out: Given a factor and a variable X, compute the CPT in which X is maxed-out. Remember to also keep track of which instantiation of X led to the maximized value. (5pts)
    # Factor multiplication: Given two factors f and g, compute the multiplied factor h=fg. (5pts)
    # Ordering: Given a set of variables X in the Bayesian network, compute a good ordering for the elimination of X based on the min-degree heuristics (2pts) and the min-fill heuristics (3.5pts). (Hint: you get the interaction graph ”for free” from the BayesNet class.)
    # Variable Elimination: Sum out a set of variables by using variable elimination. (5pts)
    # Marginal Distributions: Given query variables Q and possibly empty evidence e, compute the marginal distribution P(Q|e). Note that Q is a subset of the variables in the Bayesian network X with Q ⊂ X but can also be Q = X. (2.5pts)
    # MAP: Compute the maximum a-posteriory instantiation + value of query variables Q, given a possibly empty evidence e. (3pts)
    # MEP: Compute the most probable explanation given an evidence e. (1.5pts)

    def network_pruning(self, Query: List[str], Evidence: pd.Series) -> bool:

        """
        Given a set of query variables Q and evidence e, node- and edge-prune the Bayesian network s.t. queries of the form P(Q|E) can still be correctly calculated.
        """

        # First create a deepcopy of the structure of BN
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
    def marginalization(factor: pd.DataFrame,  variable: str) -> pd.DataFrame:

        """
        Given a factor and a variable X, compute the CPT in which X is summed-out
        """
        
        # Create a copy of the factor
        factor_copy = deepcopy(factor)

        # Sum over the variable
        factor_copy = factor_copy.groupby(factor_copy.columns.difference([variable])).sum()

        # Normalize the factor
        cpt = factor_copy.div(factor_copy.sum(axis = 1), axis = 0)

        return cpt
