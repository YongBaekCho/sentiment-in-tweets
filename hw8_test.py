from hw8 import *
import hw8
import unittest, json, numpy as np, pandas as pd, io
from compare_pandas import *
from contextlib import redirect_stdout

''' 
Auxiliary files needed:
    compare_pandas.py
    sm.pkl
The following files are needed by hw7 and so are also necessary:
    sinema_tweets_run437pm.json, sinema_tweets_run949pm.json
    mcsally_tweets_run437pm.json, mcsally_tweets_run949pm.json
'''

class TestFns(unittest.TestCase):
    def test_get_sentiment(self):
        params = [
            [-0.07917530164870593, 0.32343439691520814],
            [0.17784397060591917, 0.31276376572695747],
            [-0.022028785390854354, 0.32788514704679705],
            [0.04377983127983128, 0.3534327411603374]
            ]
        self.assertTrue(compare_lists(params[0], get_sentiment('sinema_tweets_run437pm.json')))
        self.assertTrue(compare_lists(params[1], get_sentiment('sinema_tweets_run949pm.json')))
        self.assertTrue(compare_lists(params[2], get_sentiment('mcsally_tweets_run437pm.json')))
        self.assertTrue(compare_lists(params[3], get_sentiment('mcsally_tweets_run949pm.json')))
   
    def test_get_ct_sentiment_frame(self):
        correct = pd.read_pickle('sm.pkl')
        self.assertTrue(compare_frames(correct, get_ct_sentiment_frame(), 0.005))
   
def main():
    test = unittest.defaultTestLoader.loadTestsFromTestCase(TestFns)
    results = unittest.TextTestRunner().run(test)
    print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 60) + ' / 60')
    hw8.main()
    
if __name__ == "__main__":
    main()