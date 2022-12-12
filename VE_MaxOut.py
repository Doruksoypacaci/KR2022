def Variable_el(self, CPT, X):
    for lbl in X:
        CPT=CPT.drop(columns=lbl)
    X_dropped = list(CPT.columns)
    #print(X_dropped)
    X_dropped.pop(-1)
    #print(X_dropped)
    CPT = CPT.groupby(X_dropped).sum().reset_index()
    return CPT
def max_out (self, CPT, X):
    maximized_out = CPT.groupby(X).max().reset_index()
    return maximized_out