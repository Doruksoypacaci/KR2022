# Packages
from typing import Union
from BayesNet import BayesNet
import pandas as pd
from itertools import combinations, product
from typing import Union
from BayesNet import BayesNet
from collections import defaultdict
from itertools import product, combinations, groupby
import networkx as nx
import pandas as pd
import numpy as np
import math
import copy
import random
import ttg
from BayesNet import BayesNet
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

    def multiply_factors(self, factor1,factor2):

            cols_1=list(factor1.columns)
            cols_2=list(factor2.columns)
            new_col=cols_1
            new_col.pop(-1)
            #print(cols_1)
            #print(cols_2)
            counter=0
            for i in cols_2:
                if i not in new_col and i!=("p"):
                    new_col.append(i)
                elif i!="p":
                    counter+=1
            new_table = ttg.Truths(new_col,ints=False).as_pandas()
            new_p_col=[]
            if counter==0:
                for x in (factor1["p"].tolist()):
                    for y in (factor2["p"].tolist()):
                        new_p_col.append(x*y)
                new_table["p"]=new_p_col
            else:   #common variable should discarded and then the same multiplication should be done,
                    # have a temporary solution for dog-out maybe can work on other dataframes too 
                if len(cols_1)>len(cols_2):
                    new_table=factor1
                    return new_table
                elif len(cols_2)>len(cols_1):
                    new_table=factor2
                    return new_table
                else:
                    new_table=factor2
                    return new_table
            return new_table.as_pandas()

    def Variable_el(self, cpt, X):
        for lbl in X:
            cpt=cpt.drop(columns=lbl)
        sum_out=cpt
        vars_without_X=list(sum_out.columns)
        Var_el_sum_out = sum_out.groupby(vars_without_X).sum().reset_index()
        return Var_el_sum_out


    
    def max_out (self, cpt, X):
        maximized_out = cpt.groupby(X).max().reset_index()
        return maximized_out



    def md_MAP_MPE(self, Q, evidence, func):

        # Q = list of variables (e.g. ['light-on']), but can be empty in case of MPE
        # evidence = a dictionary of the evidence e.g. {'hear-bark': True} or empty {}
        # posterior marginal: P(Q|evidence) / P(evidence)
        # MAP: sum out V/Q and then max-out Q (argmax)
        # MPE: maximize out all variables with extended factors

        variables = []

        # create a list of variables which are not in Q
        if Q != []:
            for var in self.bn.get_all_variables():
                if var != Q[0]:
                    variables.append(var)
        
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
                input = list(f_v.values())
                #print(input[0])
                #print(input[1])
                m_cpt = self.multiply_factors(input[0],input[1])
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
            #print(input[0])
            #print(input[1])
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

if __name__ == "__main__":
    net = BNReasoner("testing/dog_problem.BIFXML")
    test = net.md_MAP_MPE(['light-on'], {'hear-bark': True}, "marginal")
    print(test)