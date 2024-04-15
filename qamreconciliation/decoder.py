# -*- mode: python; gud -*- 
# from parfor import pmap
import numpy as np
import pandas as pd
from galois import GF2
from functools import reduce, partial

class Decoder():
    """ Decoder

        Decoder(edge_data)
    
        edge_data: pandas.DataFrame with columns
        - 'eid' (edge ID)
        - 'cid' (check node ID)
        - 'vid' (variable node ID)
    
    """
    def __init__(self, edge_data:pd.DataFrame, num_data_first_row = True):
        if (num_data_first_row):
            self.__edge_df  = edge_data[:][1:]
            self.__edge_num = edge_data['eid'][0]
            self.__chk_num  = edge_data['cid'][0]
            self.__var_num  = edge_data['vid'][0]
        else:
            self.__edge_df  = edge_data
            self.__edge_num = len(edge_data)
            self.__chk_num  = edge_data['cid'].max() + 1
            self.__var_num  = edge_data['vid'].max() + 1
        return

    
    @property
    def cnum(self):
        """ Number of check nodes """
        return self.__chk_num

    
    @property
    def vnum(self):
        """ Number of variable nodes """
        return self.__var_num


    @property
    def enum(self):
        """ Number of edges """
        return self.__edge_num


    @property
    def edges(self):
        """ DataFrame that describes the graph """
        return self.__edge_df


    """ Syndrome checking functions """    
    def __check_synd_node(self, check_node_index:int):
        index_set = np.array(self.edges.vid[self.edges.cid==check_node_index])
        return self.__word[index_set].sum() == self.__synd[check_node_index]


    def check_synd_node(self, check_node_index:int, word:GF2, synd:GF2):
        self.__word = word
        self.__synd = synd

        try:
            return self.__check_synd_node(check_node_index)
        finally:
            del self.__word
            del self.__synd


    
    def __check_word(self):
        return sum(map(self.__check_synd_node,
                       range(self.cnum))
                   ) == self.cnum


    def check_word(self, word:GF2, synd:GF2):
        self.__word = word
        self.__synd = synd

        try:
            return self.__check_word()
        finally:
            del self.__word
            del self.__synd

            
    
    def check_lappr(self, lappr, synd:GF2):
        self.__word = GF2((lappr<0).astype(int))
        self.__synd = synd

        try:
            return self.__check_word()
        finally:
            del self.__word
            del self.__synd
    

    """ Message passing processing functions """
    def __process_var_node(self, node_index:int):
        index_set = np.array(self.edges.eid[self.edges.vid==node_index])
        for i in range(len(index_set)):
            self.__var_to_check[index_set[i]] = \
                self.__check_to_var[index_set[   :i]].sum() + \
                self.__check_to_var[index_set[i+1: ]].sum() + \
                self.__lappr_data[node_index]
        self.__updated_lappr[node_index] = \
            self.__var_to_check[index_set[0]] + \
            self.__check_to_var[index_set[0]]
        return


    def process_var_node(self, node_index, lappr_data, check_to_var:GF2, var_to_check:GF2, updated_lappr):
        self.__lappr_data = lappr_data
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__updated_lappr = updated_lappr
        
        try:
            self.__process_var_node(node_index)
        finally:
            del self.__lappr_data
            del self.__check_to_var
            del self.__var_to_check
            del self.__updated_lappr

        return

            
    def __process_check_node(self, node_index):
        index_set = np.array(self.edges.eid[self.edges.cid==node_index])
        prefactor = -2 if self.__synd[node_index] else 2
        for i in range(len(index_set)):
            self.__check_to_var[index_set[i]] = prefactor*np.arctanh(
                reduce(lambda x, y: x*y,
                       map(lambda x: np.tanh(0.5*x),
                           [*np.array(self.__var_to_check[index_set[   :i]]),
                            *np.array(self.__var_to_check[index_set[i+1: ]])]))
            )

        return

    
    def process_check_node(self, node_index, synd, check_to_var, var_to_check):
        self.__check_to_var = check_to_var
        self.__var_to_check = var_to_check
        self.__synd         = synd

        try:
            self.__process_check_node(node_index)
        finally:
            del self.__check_to_var 
            del self.__var_to_check 
            del self.__synd
            
        return
    
        
    def decode(self, lappr_data, synd:GF2, max_iterations:int=20):    
        if (self.check_lappr(lappr_data, synd)):
            return (True, 0, lappr_data)

        self.__check_to_var  = np.zeros(self.enum, dtype=np.double)
        self.__var_to_check  = np.empty_like(self.__check_to_var)
        self.__updated_lappr = np.empty_like(lappr_data)
        self.__lappr_data    = lappr_data
        self.__synd          = synd
        self.__word          = GF2.Zeros(self.vnum)
        
        try:
            # First half iteration to propagate lapprs to check nodes
            # The following line also initializes var_to_check
            for v in range(self.vnum):
                self.__process_var_node(v)
            
            for iter_index in range(max_iterations):
                for c in range(self.cnum):
                    self.__process_check_node(c)
                    
                for v in range(self.vnum):
                    self.__process_var_node(v)

                self.__word[:] = GF2((self.__updated_lappr < 0).astype(int))
                if (self.__check_word()):
                    return (True, iter_index+1, self.__updated_lappr)

            return (False, max_iterations, self.__updated_lappr)
        
        finally:
            del self.__check_to_var 
            del self.__var_to_check 
            del self.__updated_lappr
            del self.__lappr_data
            del self.__synd
            del self.__word
            

        
if __name__=="__main__":
    df = pd.read_csv("edges_new.csv")
    dec = Decoder(df)

    word = GF2.Random(dec.vnum)
    synd = GF2([*map(lambda i: word[df[df.cid==i].vid].sum(),
                     range(dec.cnum))])

    wrong_word = GF2((np.random.rand(word.size) > 0.9).astype(int)) + word

    kind_a_lappr = -(np.array([*map(lambda b: 1.0 if b else 0.0,
                                    wrong_word)]) * 2 - 1)


    (success, itcount, final_lappr) = dec.decode(kind_a_lappr, synd)
