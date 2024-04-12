from qamreconciliation import decoder
import unittest
import pandas as pd
from galois import GF2
import numpy as np

class TestDecoder(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'eid' : [4, *range(4)],
                           'cid' : [2, 0, 0, 1, 1],
                           'vid' : [3, 0, 1, 1, 2]})
        self.uut0 = decoder.Decoder(df)
        self.uut1 = decoder.Decoder(df[:][1:], False)

        self.synd0 = GF2([1, 1])
        self.word0 = GF2([[1, 0 ,1],
                          [0, 1, 0]])
        
        self.synd1 = GF2([0, 1])
        self.word1 = GF2([[0, 0, 1],
                          [1, 1, 0]])

        
        return

    def tearDown(self):
        del self.uut0
        del self.uut1
        del self.synd0
        del self.synd1
        del self.word0
        del self.word1
        return

    """ Test properties """
    
    @unittest.skip("DataFrame equality is ambiguous")
    def test_edges(self):
        self.assertEqual(self.uut0.edges,
                         self.uut1.edges)
        return


    def test_cnum(self):
        self.assertEqual(self.uut0.cnum, 2)
        self.assertEqual(self.uut1.cnum, 2)
        return

    def test_vnum(self):
        self.assertEqual(self.uut0.vnum, 3)
        self.assertEqual(self.uut1.vnum, 3)
        return

    def test_enum(self):
        self.assertEqual(self.uut0.enum, 4)
        self.assertEqual(self.uut1.enum, 4)
        return

    """ Test check functions """
    
    def test_check_synd_node(self):
        
        self.assertTrue (self.uut0.check_synd_node(0, self.word0[0], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(0, self.word0[0], self.synd0))
        self.assertTrue (self.uut0.check_synd_node(0, self.word0[1], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(0, self.word0[1], self.synd0))
        self.assertTrue (self.uut0.check_synd_node(0, self.word1[0], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(0, self.word1[0], self.synd1))
        self.assertTrue (self.uut0.check_synd_node(0, self.word1[1], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(0, self.word1[1], self.synd1))

        self.assertTrue (self.uut0.check_synd_node(1, self.word0[0], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(1, self.word0[0], self.synd0))
        self.assertTrue (self.uut0.check_synd_node(1, self.word0[1], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(1, self.word0[1], self.synd0))
        self.assertTrue (self.uut0.check_synd_node(1, self.word1[0], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(1, self.word1[0], self.synd1))
        self.assertTrue (self.uut0.check_synd_node(1, self.word1[1], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(1, self.word1[1], self.synd1))
        
        self.assertFalse(self.uut0.check_synd_node(0, self.word0[0], self.synd1))
        self.assertFalse(self.uut1.check_synd_node(0, self.word0[0], self.synd1))
        self.assertFalse(self.uut0.check_synd_node(0, self.word0[1], self.synd1))
        self.assertFalse(self.uut1.check_synd_node(0, self.word0[1], self.synd1))
        self.assertFalse(self.uut0.check_synd_node(0, self.word1[0], self.synd0))
        self.assertFalse(self.uut1.check_synd_node(0, self.word1[0], self.synd0))
        self.assertFalse(self.uut0.check_synd_node(0, self.word1[1], self.synd0))
        self.assertFalse(self.uut1.check_synd_node(0, self.word1[1], self.synd0))

        self.assertTrue (self.uut0.check_synd_node(1, self.word0[0], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(1, self.word0[0], self.synd1))
        self.assertTrue (self.uut0.check_synd_node(1, self.word0[1], self.synd1))
        self.assertTrue (self.uut1.check_synd_node(1, self.word0[1], self.synd1))
        self.assertTrue (self.uut0.check_synd_node(1, self.word1[0], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(1, self.word1[0], self.synd0))
        self.assertTrue (self.uut0.check_synd_node(1, self.word1[1], self.synd0))
        self.assertTrue (self.uut1.check_synd_node(1, self.word1[1], self.synd0))

        return


    def test_check_word(self):        
        self.assertTrue (self.uut0.check_word(self.word0[0], self.synd0))
        self.assertTrue (self.uut1.check_word(self.word0[0], self.synd0))
        self.assertTrue (self.uut0.check_word(self.word0[1], self.synd0))
        self.assertTrue (self.uut1.check_word(self.word0[1], self.synd0))
        self.assertTrue (self.uut0.check_word(self.word1[0], self.synd1))
        self.assertTrue (self.uut1.check_word(self.word1[0], self.synd1))
        self.assertTrue (self.uut0.check_word(self.word1[1], self.synd1))
        self.assertTrue (self.uut1.check_word(self.word1[1], self.synd1))

        self.assertFalse(self.uut0.check_word(self.word0[0], self.synd1))
        self.assertFalse(self.uut1.check_word(self.word0[0], self.synd1))
        self.assertFalse(self.uut0.check_word(self.word0[1], self.synd1))
        self.assertFalse(self.uut1.check_word(self.word0[1], self.synd1))
        self.assertFalse(self.uut0.check_word(self.word1[0], self.synd0))
        self.assertFalse(self.uut1.check_word(self.word1[0], self.synd0))
        self.assertFalse(self.uut0.check_word(self.word1[1], self.synd0))
        self.assertFalse(self.uut1.check_word(self.word1[1], self.synd0))
        return


    def test_check_lappr(self):
        lappr_0 = np.array([-3.4, 0.8, -0.1])
        lappr_1 = np.array([-0.77, -0.8, 0.98])
        self.assertTrue (self.uut0.check_lappr(lappr_0, self.synd0))
        self.assertFalse(self.uut0.check_lappr(lappr_0, self.synd1))
        self.assertTrue (self.uut0.check_lappr(lappr_1, self.synd1))
        self.assertFalse(self.uut0.check_lappr(lappr_1, self.synd0))
        self.assertTrue (self.uut1.check_lappr(lappr_0, self.synd0))
        self.assertFalse(self.uut1.check_lappr(lappr_0, self.synd1))
        self.assertTrue (self.uut1.check_lappr(lappr_1, self.synd1))
        self.assertFalse(self.uut1.check_lappr(lappr_1, self.synd0))
        return



class TestDecoderProcessing(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({'eid' : [*range(8)],
                           'cid' : [0, 0, 0, 1, 1, 2, 2, 2],
                           'vid' : [0, 1, 3, 1, 2, 1, 3, 4]})
        self.uut = decoder.Decoder(df, False)
        return

    
    def make_message_arrays(self):
        return (np.random.randn(self.uut.enum),
                np.random.randn(self.uut.enum))

    
    def test_process_var_node(self):
        check_to_var, var_to_check = self.make_message_arrays()
        lappr_data = np.random.randn(self.uut.vnum)

        # Check node of degree 3
        self.uut.process_var_node(1, lappr_data, check_to_var, var_to_check)
        self.assertEqual(var_to_check[1],
                         check_to_var[3] + check_to_var[5] + lappr_data[1])
        self.assertEqual(var_to_check[3],
                         check_to_var[1] + check_to_var[5] + lappr_data[1])
        self.assertEqual(var_to_check[5],
                         check_to_var[1] + check_to_var[3] + lappr_data[1])

        # Check node of degree 1
        self.uut.process_var_node(2, lappr_data, check_to_var, var_to_check)
        self.assertEqual(var_to_check[4], lappr_data[2])

        # check node of degree 2
        self.uut.process_var_node(3, lappr_data, check_to_var, var_to_check)
        self.assertEqual(var_to_check[2],
                         check_to_var[6] + lappr_data[3])
        self.assertEqual(var_to_check[6],
                         check_to_var[2] + lappr_data[3])

        return
        


    def test_process_check_node(self):
        check_to_var, var_to_check = self.make_message_arrays()
        s = GF2.Random(self.uut.cnum)

        # Check degree 2
        self.uut.process_check_node(1, s, check_to_var, var_to_check)
        pre = -2 if (s[1]) else 2
        self.assertAlmostEqual(check_to_var[3],
                               pre*var_to_check[4]/2,
                               delta=np.abs(check_to_var[3])*1e-6)
        self.assertAlmostEqual(check_to_var[4],
                               pre*var_to_check[3]/2,
                               delta=np.abs(check_to_var[4])*1e-6)


        # Check degree 3
        self.uut.process_check_node(2, s, check_to_var, var_to_check)
        pre = -2 if (s[2]) else 2
        self.assertAlmostEqual(check_to_var[5],
                               pre*np.arctanh(np.tanh(var_to_check[6]/2)*\
                                              np.tanh(var_to_check[7]/2)),
                               delta=np.abs(check_to_var[5])*1e-6)
        self.assertAlmostEqual(check_to_var[6],
                         pre*np.arctanh(np.tanh(var_to_check[5]/2)*\
                                        np.tanh(var_to_check[7]/2)),
                               delta=np.abs(check_to_var[6])*1e-6)
        self.assertAlmostEqual(check_to_var[7],
                         pre*np.arctanh(np.tanh(var_to_check[6]/2)*\
                                        np.tanh(var_to_check[5]/2)),
                               delta=np.abs(check_to_var[7])*1e-6)
        
        return




class TestDecoderDecoding(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("test/hamming_7-4.csv")
        self.uut = decoder.Decoder(df)
        return


    def tearDown(self):
        del self.uut
        return


    def test_decode_correct_word(self):
        lappr = np.array([1.2, -0.8, -1.3, 1.1, -0.4, 0.5, 1.9])
        synd  = GF2([1, 1, 0])

        success, iter_count, updated_lappr = self.uut.decode(lappr, synd)

        self.assertTrue(success)
        self.assertEqual(iter_count, 0)
        self.assertEqual((updated_lappr != lappr).sum(), 0)


        return

    def test_decode_wrong_word_one_bit(self):
        lappr = np.array([-0.1, -0.8, -1.3, 1.1, -0.4, 0.5, 0.1])
        synd  = GF2([1, 1, 0])

        success, iter_count, updated_lappr = self.uut.decode(lappr, synd)

        print(success, iter_count, updated_lappr)
        
        self.assertTrue(success)
        self.assertEqual(
            np.array(GF2((updated_lappr<0).astype(int))+\
                     GF2([0, 1, 1, 0, 1, 0, 0]),
                     dtype=np.uint).sum(),
            0)
        self.assertLessEqual(iter_count, 20)

        return
