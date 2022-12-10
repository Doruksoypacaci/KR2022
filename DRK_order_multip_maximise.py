from typing import Union
from BayesNet import BayesNet
import pandas as pd
from itertools import combinations, product

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
    def ordering(self,vars,heuristic,ascending): #needed algo
        # Undirected Graph
        Graph = self.bn.structure.to_undirected()
        nodes = self.bn.get_all_variables()

        if vars is None:
            vars = nodes.copy()

        # key equals to node and items are its neighbors
        Adj_dict={v: set(G[v]) - set([v]) for v in G}

        # Remove unrelated nodes
        not_vars = [node for node in nodes if node not in vars]
        for node in not_vars:
            self.eliminate_A(Adj_dict, node)

        select = min if ascending else max

        order = []
        if heuristic=="minfill":

            for _ in range(len(Adj_dict)):
                # Select the node to eliminate from the Graph with related heuristic
                v = select(Adj_dict, key=lambda x: minfill(Adj_dict, x))
               # Remove node from Graph
                self.eliminate_A(Adj_dict, v)
                # Add it to list of order
                order.append(v)
        elif heuristic=="mindeg":
            for _ in range(len(Adj_dict)):
                # Select the node to eliminate from the Graph with related heuristic
                v = select(Adj_dict, key=lambda x: mindeg(Adj_dict, x))
                # Remove node from Graph
                self.eliminate_A(Adj_dict, v)
                # Add it to list of order
                order.append(v)
        

        return order
    def eliminate_A(adjacenct, node):
        # Eliminating variable and adding edge between the neighbors it has
        # done rather than the given adjacency dict, so there is no dict returned
        neighbors = adjacenct[node]

        # Create edges in between all neighbors
        for x, n in combinations(neighbors, 2):
            if n not in adjacenct[x]:
                adjacenct[x].add(n)
                adjacenct[n].add(x)

        # Remove node in every neighbor
        for n in neighbors:
            adjacenct[n].discard(node)

        # Remove from adj dict
        del adjacenct[node]
    def mindeg(adjacency,node):
        return len(adjacency[node])

    def minfill(adjacency,node):
        int_wanted = 0
        for x, n in combinations(adjacency[node], 2):
            if x not in adjacency[n]:
                int_wanted += 1

        return int_wanted

    def multiplication_factors(CPT, node, E): #needed algo

        # Select all used CPT's figure out the columns
        used = {}
        columns = {node}

        for cpt in CPT:
            cpt_columns = (set(CPT[cpt].columns)) - set("p")
            if node in cpt_columns:
                used[cpt] = CPT[cpt]
                columns = columns.union(cpt_columns)

        # Create newCPT for the factor stuf and shortened
        newCPT = pd.DataFrame(product([False, True], repeat=len(columns)), columns=columns)
        newCPT["p"] = 1.0
        newCPT = BayesNet.get_compatible_instantiations_table(E, newCPT)
        newCPT = newCPT.reset_index(drop=True)
        newCPT["valid"] = True

        # Factor multip. and row check
        for i in range(newCPT.shape[0]):
            row = pd.Series(newCPT.iloc[i][:-2], columns)  

            for cpt in used.values():
                p = BayesNet.get_compatible_instantiations_table(row, cpt)["p"]

                if len(p) == 0:
                    newCPT.at[i, "valid"] = False
                    break

                p = p.values[0]
                newCPT.at[i, "p"] *= p

        newCPT = newCPT.loc[newCPT["valid"]].reset_index(drop=True)
        del newCPT["valid"]

        return used, newCPT, columns
    def maximise_out(newCPT, cols, node): #needed algo

        if len(cols) > 1:
            maxCPT = newCPT.groupby(list(cols - {node}))["p"].max().reset_index()
            fill = []

            for i in range(maxCPT.shape[0]):
                start = pd.Series(maxCPT.iloc[i], maxCPT.columns)
                row = BayesNet.get_compatible_instantiations_table(start, newCPT)
                fill.append(row)

            newCPT = pd.concat(fill, ignore_index=True)

        else:
            newCPT = newCPT.query("p == p.max()").reset_index(drop=True)

        return newCPT

