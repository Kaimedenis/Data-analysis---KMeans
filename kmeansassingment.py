#loading data
import pandas as pd
data=pd.read_csv('OnlineRetail.csv',encoding="unicode_escape")
print(data.head(10))
print(data.shape)
print(data.info())
print(data.columns)
print(data.describe())

#data cleaning
print(data.isnull().sum())
data_null=round(100*(data.isnull().sum())/len(data),2)#to give null values in percentage raised to 2dp
print(data_null)
#data=data.drop(['StockCode'], axis=1)
# Changing the datatype of Customer Id as per Business understanding
data['CustomerID']= data['CustomerID'].astype(str)
print(data)

#Data Preparation
#1)Recency(R)-number of days since last purchase
data['Amount'] = data['Quantity']*data['UnitPrice']
print(data.info())
print(data.head())
#data.monitoring = data.groupby('CustomerID')['Amount'].sum()
data['monitoring'] = data.groupby('CustomerID')['Amount'].sum()
print(data.monitoring)
print(data.monitoring.head())
#mostsold product
data['monitoring'] = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print(data.monitoring.head())
#region
data['monitoring'] = data.groupby('Country')['Quantity'].sum()
print(data.monitoring.head())

#frequently sold
data['monitoring'] = data.groupby('Description')['InvoiceNo'].count()
print(data.monitoring.head())
# Convert to datetime to proper datatype

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
print(data)
# Compute the maximum date to know the last transaction date

max_date = max(data['InvoiceDate'])
print(max_date)
# Compute the minimum date to know the last transaction date

min_date = min(data['InvoiceDate'])
print(min_date)
data['Diff'] = max_date - data['InvoiceDate']
data.head()

# The 'InvoiceDate' column is already in datetime format
date_difference = data['InvoiceDate'].max() - data['InvoiceDate'].min()

# Convert the difference to days
date_difference_days = date_difference.days
print("Difference between max and min dates:", date_difference_days, "days")

from datetime import timedelta
diff_time=max_date - timedelta(days=30)
print(diff_time)

dtt=data[data['InvoiceDate'] >diff_time]
total_amount = dtt['Amount'].sum()
print(total_amount)

total_amt_sales = dtt['Amount'].count()
print(total_amt_sales)

#spliting data
X = data[['Quantity', 'UnitPrice', 'CustomerID']]  # Features
y = data['Amount']  # Target variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
print("Size of training set:", X_train.shape)
print("Size of test set:", X_test.shape)


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Selecting features for clustering
X = data[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choosing the number of clusters (let's say 4 for demonstration)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Adding cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Visualizing clusters
plt.figure(figsize=(10, 6))
for cluster in data['Cluster'].unique():
    plt.scatter(data[data['Cluster'] == cluster]['Quantity'], data[data['Cluster'] == cluster]['UnitPrice'], label=f'Cluster {cluster}')
plt.xlabel('Quantity')
plt.ylabel('Unit Price')
plt.title('Clusters by Quantity and Unit Price')
plt.legend()
plt.show()


# Elbow Method for finding the optimal number of clusters

wcss = []  # Within-cluster sum of squares

# Selecting features for clustering
X = data[['Quantity', 'UnitPrice']].dropna()

# Standardize the features
X_scaled = scaler.fit_transform(X)

# Trying different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
