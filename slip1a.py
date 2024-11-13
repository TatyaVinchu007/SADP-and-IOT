#Slip 1A, 22-26B 
#Use Apriori algorithm on groceries dataset to find which items are brought together.Use minimum support =0.25
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample dataset
data = [['Milk', 'Bread', 'Eggs'],
        ['Bread', 'Diaper', 'Beer', 'Eggs'],
        ['Milk', 'Diaper', 'Beer', 'Cola'],
        ['Bread', 'Milk', 'Diaper', 'Beer'],
        ['Bread', 'Milk', 'Cola']]

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# One-hot encoding
onehot = pd.get_dummies(df.stack()).groupby(level=0).sum()

# Apply Apriori algorithm with minimum support of 0.25
frequent_itemsets = apriori(onehot, min_support=0.25, use_colnames=True)
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules)