import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_reviews = 200
data = {
    'Review': [f"Review {i+1}: This is a sample review." for i in range(num_reviews)],
    'Churn': np.random.choice([0, 1], size=num_reviews, p=[0.8, 0.2]), # 80% not churn, 20% churn
    'Subscription_Length_Months': np.random.randint(1, 24, size=num_reviews)
}
# Add some sentiment to reviews (positive, negative, neutral)
analyzer = SentimentIntensityAnalyzer()
for i in range(num_reviews):
    if data['Churn'][i] == 1: #Churned customers are more likely to have negative reviews
        sentiment = analyzer.polarity_scores("This service is terrible. I'm cancelling.")
    elif data['Subscription_Length_Months'][i] < 6: #Short subscriptions are more likely to be neutral
        sentiment = analyzer.polarity_scores("It was okay.")
    else: #Longer subscriptions are more likely to be positive
        sentiment = analyzer.polarity_scores("I love this service!")
    data['Review'][i] += f" Sentiment: {sentiment}"
    data['Compound_Sentiment'] = [analyzer.polarity_scores(review).get('compound') for review in data['Review']]
df = pd.DataFrame(data)
# --- 2. Analysis ---
# Calculate average sentiment score for churned and non-churned customers
churned_sentiment = df[df['Churn'] == 1]['Compound_Sentiment'].mean()
not_churned_sentiment = df[df['Churn'] == 0]['Compound_Sentiment'].mean()
print(f"Average sentiment score for churned customers: {churned_sentiment}")
print(f"Average sentiment score for non-churned customers: {not_churned_sentiment}")
# --- 3. Visualization ---
plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='Compound_Sentiment', data=df)
plt.title('Sentiment vs. Churn')
plt.xlabel('Churned (1=Yes, 0=No)')
plt.ylabel('Compound Sentiment Score')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'sentiment_vs_churn.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Subscription_Length_Months', y='Compound_Sentiment', hue='Churn', data=df)
plt.title('Sentiment, Subscription Length, and Churn')
plt.xlabel('Subscription Length (Months)')
plt.ylabel('Compound Sentiment Score')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename2 = 'subscription_length_sentiment.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")