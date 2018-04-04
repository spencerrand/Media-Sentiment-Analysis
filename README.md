
<h1>Observations</h1>
<p>1. For the last 100 tweets as of 4/2/2018, CBS was the most positive, followed by BBC. CNN and the NY Times were basically neutral, with Fox having a slightly negative tilt.</p>
<p>2. Despite having an overall compound score that was neutral, the NY Times had the highest standard deviation when compared to the other organizations. This says to me that they have roughly the same number of positive tweets as negative ones.</p>
<p>3. CBS was the only organization with a non-zero median.  In this case the median for CBS was positive, which indicates that their compound scores aren't because of a few strongly positive tweets.</p>

It would be interesting to see how these scores changed during a negative event such as a recession or during a positive event like the Olympics.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (twitter_consumer_key, twitter_consumer_secret, twitter_access_token, twitter_access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
auth.set_access_token(twitter_access_token, twitter_access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Increase the size of the scatter plots
plt.rcParams["figure.figsize"] = [10,8]
```


```python
# Target User Accounts
target_user = ("@BBC", "@CBS", "@CNN", "@FoxNews", "@NYTimes")

# Spelled out names of twitter accounts
target_names = ('BBC','CBS','CNN','Fox','NYT')

# List for dictionaries of results
results_list = []

# Variables for holding sentiments
sentiments = []

# Loop through each user
for user in target_user:

    # Variables for holding sentiments
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []

    # Counter
    counter = 1

    # Variable for max_id
    oldest_tweet = None

    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, max_id = oldest_tweet)

        # Loop through all tweets 
        for tweet in public_tweets:

            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            tweets_ago = counter

            # Get Tweet ID, subtract 1, and assign to oldest_tweet
            oldest_tweet = tweet['id'] - 1

            # Add sentiments for each tweet into a list
            sentiments.append({"Username": user,
                               "Date": tweet["created_at"], 
                               "Tweet": tweet["text"],
                               "Compound": compound,
                               "Positive": pos,
                               "Negative": neg,
                               "Neutral": neu,
                               "Tweets Ago": counter})

            # Add to counter 
            counter += 1
```


```python
# Convert sentiments to a dataframe
sentiments_pd = pd.DataFrame.from_dict(sentiments).set_index("Username")

# Export the dataframe to CSV
sentiments_pd.to_csv("Sentiment Analysis of Media Tweets.csv")

# Print out the dataframe to view results
sentiments_pd.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweet</th>
      <th>Tweets Ago</th>
    </tr>
    <tr>
      <th>Username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.0000</td>
      <td>Mon Apr 02 19:23:07 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>When this woman visited an uninhabited Caribbe...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.1531</td>
      <td>Mon Apr 02 19:02:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.897</td>
      <td>0.103</td>
      <td>In a revealing and emotional journey, @LennyHe...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.5423</td>
      <td>Mon Apr 02 18:45:06 +0000 2018</td>
      <td>0.000</td>
      <td>0.837</td>
      <td>0.163</td>
      <td>A fictionalised account of how Dave Allen beca...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>0.0000</td>
      <td>Mon Apr 02 17:08:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Would you try a blue cheese and pear ice cream...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>@BBC</th>
      <td>-0.5574</td>
      <td>Mon Apr 02 14:14:17 +0000 2018</td>
      <td>0.184</td>
      <td>0.816</td>
      <td>0.000</td>
      <td>RT @BBCBreaking: South African anti-apartheid ...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This section creates the scatterplot of sentiments of the last 100 tweets sent out by each news organization

# Clear plot, just in case
plt.clf()

# Set list of colors
colors = ['red', 'orange', 'yellow', 'green', 'blue']

# Loop through each twitter account
for index, user in enumerate(target_user):
    
    # Create scatterplot for each twitter account
    plt.scatter(sentiments_pd[sentiments_pd.index==user]["Tweets Ago"], 
                sentiments_pd[sentiments_pd.index==user]["Compound"],
                marker="o", facecolors=colors[index], edgecolors="black", s=100, alpha=0.75)

# Adjust other graph properties
now = datetime.now()
now = now.strftime("%m-%d-%Y %H:%M%p")
plt.title(f"Sentiment Analysis of Media Tweets ({now})")
plt.legend(target_names, loc='best', bbox_to_anchor=(1.25, 1.01), prop={'size': 14})
plt.xlim(len(sentiments_pd[sentiments_pd.index==user]["Tweets Ago"])+5, -5)
plt.ylim(-1.1, 1.1)
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.show()

# Save the chart
plt.savefig("Sentiment Analysis of Media Tweets.png")
```


![png](output_4_0.png)



```python
# This section creates the bar graph showing the overall sentiments of the last 100 tweets from each organization

# Clear plot, just in case
plt.clf()

# List for holding overall sentiment
overall_sentiment_list = []

# Create values for x-axis
x_axis = np.arange(len(target_user))

# Loop through each twitter account
for user in target_user:
    
    # Append average compound score to list
    overall_sentiment_list.append(sentiments_pd[sentiments_pd.index==user]["Compound"].mean())

# Create bar chart
plt.bar(x_axis, overall_sentiment_list, color=colors, align="edge")
    
# Adjust other graph properties
plt.title(f"Overall Media Sentiment Based on Twitter ({now})")
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, target_names)
plt.xlabel("Media Organization")
plt.ylabel("Tweet Polarity")
plt.show()

# Save the chart
plt.savefig("Overall Media Sentiment.png")
```


![png](output_5_0.png)



```python
# This section compares general statistics for the compound score for each organization

# Variable for holding compound scores
compound_list = []

# Loop through each user
for user in target_user:
    
    # Append average compound score to list
    compound_list.append({"Username": user,
                          "Mean": sentiments_pd[sentiments_pd.index==user]["Compound"].mean(),
                          "Median": sentiments_pd[sentiments_pd.index==user]["Compound"].median(),
                          "Min": sentiments_pd[sentiments_pd.index==user]["Compound"].min(),
                          "Max": sentiments_pd[sentiments_pd.index==user]["Compound"].max(),
                          "Std Dev": np.std(sentiments_pd[sentiments_pd.index==user]["Compound"])
                        })
    
# Convert list to a dataframe
compound_df = pd.DataFrame.from_dict(compound_list).set_index("Username").round(3)

# Print out the dataframe to view results
compound_df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Max</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Min</th>
      <th>Std Dev</th>
    </tr>
    <tr>
      <th>Username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBC</th>
      <td>0.930</td>
      <td>0.151</td>
      <td>0.00</td>
      <td>-0.875</td>
      <td>0.388</td>
    </tr>
    <tr>
      <th>@CBS</th>
      <td>0.945</td>
      <td>0.345</td>
      <td>0.44</td>
      <td>-0.698</td>
      <td>0.365</td>
    </tr>
    <tr>
      <th>@CNN</th>
      <td>0.710</td>
      <td>-0.038</td>
      <td>0.00</td>
      <td>-0.922</td>
      <td>0.363</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.655</td>
      <td>-0.124</td>
      <td>0.00</td>
      <td>-0.869</td>
      <td>0.356</td>
    </tr>
    <tr>
      <th>@NYTimes</th>
      <td>0.765</td>
      <td>-0.014</td>
      <td>0.00</td>
      <td>-0.844</td>
      <td>0.419</td>
    </tr>
  </tbody>
</table>
</div>


