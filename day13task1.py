import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

data = {
    'Price': [250000, 300000, 400000, 350000, 450000, 500000, 600000, 550000, 700000, 650000],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'Chicago', 'New York', 'Chicago', 'Los Angeles', 'New York']
}

df = pd.DataFrame(data)

sns.histplot(df['Price'], kde=True)
plt.show()

print("Skewness:", skew(df['Price']))
print("Kurtosis:", kurtosis(df['Price']))

sns.countplot(x='City', data=df)
plt.show()