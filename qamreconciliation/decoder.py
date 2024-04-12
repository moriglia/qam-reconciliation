from parfor import pmap
import numpy as np
import pandas as pd
from galois import GF2
from functools import reduce

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
    
    def check_synd_node(self, check_node_index:int, word:GF2, synd:GF2):
        index_set = np.array(self.edges.vid[self.edges.cid==check_node_index])
        return word[index_set].sum() == synd[check_node_index]

    
    def check_word(self, word:GF2, synd:GF2):
        return sum(pmap(self.check_synd_node,
                        range(self.cnum),
                        (word, synd))
                   ) == self.cnum

    
    def check_lappr(self, lappr, synd:GF2):
        return self.check_word(GF2((lappr<0).astype(int)),
                               synd)
    

    """ Message passing processing functions """
    
    def process_var_node(self, node_index, lappr_data, check_to_var, var_to_check):
        index_set = np.array(self.edges.eid[self.edges.vid==node_index])
        for i in range(len(index_set)):
            var_to_check[index_set[i]] = check_to_var[index_set[:i]].sum() + \
                check_to_var[index_set[i+1:]].sum() + \
                lappr_data[node_index]
        return check_to_var[index_set].sum() + lappr_data[node_index]


    def process_check_node(self, node_index, s, check_to_var, var_to_check):
        index_set = np.array(self.edges.eid[self.edges.cid==node_index])
        prefactor = -2 if s[node_index] else 2
        for i in range(len(index_set)):
            check_to_var[index_set[i]] = prefactor*np.arctanh(
                reduce(lambda x, y: x*y,
                       map(lambda x: np.tanh(0.5*x),
                           [*np.array(var_to_check[index_set[   :i]]),
                            *np.array(var_to_check[index_set[i+1: ]])]))
            )

        return

    
    def decode(self, lappr_data, synd:GF2, max_iterations:int=20):
        check_to_var = np.zeros(self.enum, dtype=np.double)
        var_to_check = np.zeros(self.enum, dtype=np.double)

        if (self.check_lappr(lappr_data, synd)):
            return (True, 0, lappr_data)

        # Identical to updated_lappr[:] = lappr_data[:]
        # But the messages for the checknodes have to be updated anyways
        updated_lappr = np.array(pmap(self.process_var_node,
                                      range(self.vnum),
                                      (lappr_data,
                                       check_to_var,
                                       var_to_check)))
    
        for iter_index in range(max_iterations):
            pmap(self.process_check_node,
                 range(self.cnum),
                 (synd, check_to_var, var_to_check))
            
            updated_lappr = np.array(pmap(self.process_var_node,
                                          range(self.vnum),
                                          (lappr_data,
                                           check_to_var,
                                           var_to_check)))
            
            if (self.check_lappr(updated_lappr, synd)):
                return (True, iter_index+1, updated_lappr)

        return (False, max_iterations, updated_lappr)

        
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
