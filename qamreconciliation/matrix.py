import pandas as pd
from galois import GF2
import numpy as np


class Matrix:
    def __init__(self, edge_data:pd.DataFrame, num_data_first_row:bool=True):
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



    """ Eval syndrome of a given word """
    def eval_syndrome(self, word:GF2):
        synd = GF2.Zeros(self.cnum)
        for i in range(self.cnum):
            synd[i] = word[self.edges.vid[self.edges.cid==i]].sum()

        return synd

    
