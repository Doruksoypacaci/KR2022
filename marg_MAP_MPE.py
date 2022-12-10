# DISCLAIMER: this only works with some functions used from the KR21_group60 project (variable elimination, factor multiplication and network pruning)

from BayesNet import BayesNet

net = BayesNet()
net.load_from_bifxml("testing/dog_problem.BIFXML")


# function to calculate the marginal distribution, the MAP, or the MPE (func = 'marginal', 'MAP', 'MPE')
def marginal_dis(self, Q, evidence, func):

    # Q = list of variables (e.g. ['light-on']), but can be empty in case of MPE
    # evidence = a dictionary of the evidence e.g. {'hear-bark': True} or empty {}
    # posterior marginal: P(Q|evidence) / P(evidence)
    # MAP: sum out V/Q and then max-out Q (argmax)
    # MPE: maximize out all variables with extended factors

    variables = []

    # create a list of variables which are not in Q
    if Q != []:
        for var in self.get_all_variables():
            if var != Q[0]:
                variables.append(var)
    
    # prune the network given the evidence (# reduce all the factors w.r.t. evidence)
    self.network_pruning(Q,evidence) # this function is not yet implemented

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
        joint_prob = self.MultiplyFactors(list(M.values()))
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

test = marginal_dis(net, [], {'hear-bark': True}, "MPE")
print(test)
