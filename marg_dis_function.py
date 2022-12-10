from BayesNet import BayesNet

net = BayesNet()
net.load_from_bifxml("testing/dog_problem.BIFXML")
test = net.get_all_variables()
df_real = net.get_all_cpts()

# Q = ['light-on']
# evidence = empty or dictionary {'hear-bark: True}

# function to calculate the marginal distribution of Q given evidence
def marginal_dis(self, Q, evidence):

    # Q = list of variables (e.g. ['light-on'])
    # evidence = a dictionary of the evidence e.g. {'hear-bark': True} or empty {}
    # posterior marginal: P(Q|evidence) / P(evidence)

    variables = []

    # create a list of variables which are not in Q
    for var in self.get_all_variables():
        if var != Q[0]:
            variables.append(var)
    
    # prune the network given the evidence (# reduce all the factors w.r.t. evidence)
    # self.pruning_function(Q,evidence) # this function is not yet implemented

    # compute the probability of the evidence
    e_factor = 1
    for e in evidence:
        evidence_probability = self.get_cpt(e)
        e_factor = e_factor * self.get_cpt(e)['p'].sum()
    
    # retrieve all the cpts and delete them accordingly
    M = self.get_all_cpts()

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
            m_cpt = self.MultiplyFactors(list(f_v.values()))            
            new_cpt = self.sumOutVars(m_cpt, [v])           

            # delete the variables from the dictionary M
            for f in f_v:
                del M[f]
            
            # add the new cpt to the dictionary M
            factor +=1
            M["F"+str(factor)] = new_cpt
        
        # skip multiplication when there is only one cpt
        elif len(f_v) == 1:
            new_cpt = self.sumOutVars(list(f_v.values())[0], [v])
            
            # delete the variables from the dictionary M
            for f in f_v:
                del M[f]
            
            # add the new cpt to the dictionary M
            factor +=1
            M["F "+str(factor)] = new_cpt

    # compute joint probability of Q and evidence
    if len(M) > 1:
        m_distribution = self.MultiplyFactors(list(M.values()))
    else:
        m_distribution = list(M.values())[0]
    
    # divide by the probability of the evidence
    m_distribution['p'] = m_distribution['p'] / (e_factor)

    return m_distribution

test = marginal_dis(net, ['light-on'], {'hear-bark': True})
print(test)
