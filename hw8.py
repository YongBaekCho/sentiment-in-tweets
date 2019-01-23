import pandas as pd
import numpy as np
import json
from textblob import TextBlob as tb
import math
import matplotlib.pyplot as plt
#Author: YongBaek Cho
#Date : 11/01/2018
#Description: This programe is to sentiment analysis and making bar plots from dateaframe
#             analyzing the sentiment expressed in tweets containing the keywords 'Sinema' and 'McSally'
def get_sentiment(filename):
    '''
    This function takes a filename
    filename contains a json string representing a list of the text of a number of tweets
    '''
    with open(filename) as fp:
        text = json.loads(fp.read())
        text = [tb(tweet) for tweet in text]
        text_polarities = np.array([blob.polarity for blob in text if not (blob.polarity == 0.0 and blob.subjectivity == 0.0)])
        mean_polarity = text_polarities.mean()
        ssd_polarity = text_polarities.std(ddof=1)
    return [mean_polarity, ssd_polarity]
    
def get_ct_sentiment_frame():
    '''
    This function takes no arguments and returns Data frame
    '''
    a = get_sentiment('sinema_tweets_run437pm.json')
    b = get_sentiment('sinema_tweets_run949pm.json')
    c = get_sentiment('mcsally_tweets_run437pm.json')
    d = get_sentiment('mcsally_tweets_run949pm.json')
    
    index = ['Sinema', 'McSally']
    
    df = pd.DataFrame({'pre_mean': [a[0], c[0]], 'pre_std': [a[1], c[1]], 'post_mean': [b[0], d[0]], 'post_std': [b[1], d[1]]}, index = index)
    return df
    
    
def make_fig(df):
    '''
    This function takes a sentiment frame
    create the barplot with error bar
    '''
    df = get_ct_sentiment_frame()
    plt.style.use('dark_background')
    df[['pre_mean', 'post_mean']].T.plot(kind='bar', yerr=df[['pre_std', 'post_std']].values.T,legend = False,color = ['blue','green','blue','green'],edgecolor = 'black', error_kw=dict(ecolor='k'), capsize = 8)
    ax = plt.gca()
    plt.ylabel('Sentiment', fontsize = 24, color = 'Red')
    
    ax.set_xticklabels(['Sinema-McSally 4:37 pm', 'Sinema-McSally 9:49 pm'], rotation = 360)
    ax.set_facecolor("lavenderblush")
    ax.spines['bottom'].set_color('red')
    ax.spines['top'].set_color('red')
    ax.spines['right'].set_color('red')
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='x', colors='red', labelsize = 18)
    ax.tick_params(axis='y', colors='red', labelsize = 18)
    
    
    
    
        
def main():
    '''
    get the sentiment frame make figure and display the bar plots
    '''
    df = get_ct_sentiment_frame()
    make_fig(df)
    plt.show()
if __name__ == "__main__":
    main()