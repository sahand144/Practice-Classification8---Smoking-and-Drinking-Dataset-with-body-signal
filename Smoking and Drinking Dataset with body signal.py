#Load the dataset
path = r'D:\datasets\New To Work on 3\Smoking and Drinking Dataset with body signal\Smoking and Drinking Dataset with body signal.zip'

#Import the necessary libraries for the task
import zipfile as zip
import pandas as pd

#Extract the dataset
with zip.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall()
    filename = zip_ref.namelist()
    print(filename)
    df = pd.read_csv(filename[0])

#Explore the dataset
print(df.head())
print(df.columns)
print(df.info())

#Check for missing values
df.isnull().sum()#no missing values

#Check for duplicates
df.duplicated().sum() #26 duplicates
#Drop the duplicates
df.drop_duplicates(inplace=True)

#show the shape of the dataset
print(df.shape)
#describe the dataset
print(df.describe())

#show the number of unique values in the dataset
print(df.nunique().sort_values(ascending=False))

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Plot the distribution of the target variable
plt.figure(figsize=(10,5))
df['SMK_stat_type_cd'].value_counts().plot(kind='bar')
plt.show()

df['SMK_stat_type_cd'] = df['SMK_stat_type_cd'].replace({1:'Never', 2:'Used to smoke' , 3:'Current smoker'})
df['DRK_YN'] = df['DRK_YN'].replace({'N':'Not Drinker', 'Y':'Drinker'})

#Plot the distribution of the target variable
plt.figure(figsize=(10,5))
df['DRK_YN'].value_counts().plot(kind='bar')#we can see that the dataset is balanced
plt.show()
print(df['DRK_YN'].value_counts())

#Plot the distribution of the target variable
plt.figure(figsize=(10,5))
df['SMK_stat_type_cd'].value_counts().plot(kind='bar')
plt.show()
print(df['SMK_stat_type_cd'].value_counts())

#Plot the distribution of the target variable
plt.figure(figsize=(12,7))
sns.catplot(y='age',x='DRK_YN',hue='SMK_stat_type_cd',data=df , kind='bar')
plt.xlabel('Drinking Status')
plt.ylabel('Age')
plt.title('Drinking Status vs Smoking Status')
plt.show()

#Plot the distribution of the target variable
colors=['red','yellow','green','brown']
explode=[0 , 0.1 , 0]
values=df['SMK_stat_type_cd'].value_counts().values
labels=df['SMK_stat_type_cd'].value_counts().index
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('SMK_stat_type_cd',color='black',fontsize=10)
plt.show()

#Plot the boxplots of the numerical columns
def plot_boxplots(dataframe):
    num_columns = dataframe.select_dtypes(include=['number']).columns
    num_plots = len(num_columns)
    rows = (num_plots + 1) // 2

    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(20, 20))

    for i, column in enumerate(num_columns):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        sns.boxplot(x=dataframe[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)

    plt.tight_layout()
    plt.show()

plot_boxplots(df)

#Plot the histograms of the numerical columns
numeric_cols = df.select_dtypes(include = ['int64' , 'float64']).columns
for col in numeric_cols:
    plt.figure(figsize = (10,6))
    sns.histplot(df[col] , kde =True , bins = 30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
    
#Plot the barplots of the numerical columns
plt.figure(figsize=(18,8))

plt.subplot(4, 2, 1)
sns.barplot(data=df, x= 'age', y = 'sight_left', hue = "DRK_YN", palette = 'bright')
plt.subplot(4, 2, 2)
sns.barplot(data=df, x= 'age', y = 'sight_left', hue = "SMK_stat_type_cd", palette = 'bright')
plt.subplot(4, 2, 3)
sns.barplot(data=df, x= 'age', y = 'sight_right', hue = "DRK_YN", palette = 'bright')
plt.subplot(4, 2, 4)
sns.barplot(data=df, x= 'age', y = 'sight_right', hue = "SMK_stat_type_cd", palette = 'bright')
plt.subplot(4, 2, 5)
sns.barplot(data=df, x= 'age', y = 'hear_right', hue = "DRK_YN", palette = 'bright')
plt.subplot(4, 2, 6)
sns.barplot(data=df, x= 'age', y = 'hear_right', hue = "SMK_stat_type_cd", palette = 'bright')
plt.subplot(4, 2, 7)
sns.barplot(data=df, x= 'age', y = 'hear_left', hue = "DRK_YN", palette = 'bright')
plt.subplot(4, 2, 8)
sns.barplot(data=df, x= 'age', y = 'hear_left', hue = "SMK_stat_type_cd", palette = 'bright')
plt.tight_layout()
plt.show()

#Plot the heatmap of the correlation matrix
numerical_variables = df.select_dtypes(exclude=['object']).columns.tolist()
corr_matrix = df[numerical_variables].corr()    
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


def get_highly_correlated_features(df, threshold=0.6):
    """
    Returns a list of tuples with pairs of column names that have a high correlation.

    Parameters:
    df (pd.DataFrame): DataFrame with integer features
    threshold (float): Absolute correlation threshold to consider (default is 0.8)

    Returns:
    List[Tuple[str, str]]: List of column name pairs with high correlation
    """
    # Ensure only int columns are used
    int_df = df.select_dtypes(include='int')
    
    # Compute correlation matrix
    corr_matrix = int_df.corr()
    
    # Extract upper triangle of correlation matrix to avoid duplicates
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2))
    
    return high_corr_pairs

highly_correlated_features = get_highly_correlated_features(df)
print(highly_correlated_features)

#Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

df.info()

#Train test split
from sklearn.model_selection import train_test_split
X = df.drop('DRK_YN', axis=1)
y = df['DRK_YN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train_scaled.shape: {X_train_scaled.shape}, \nX_test_scaled.shape: {X_test_scaled.shape}, \ny_train.shape: {y_train.shape} ,\ny_test.shape: {y_test.shape}")


#Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

#train with different models and in the and add the results to a dataframe

#Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled,y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
test_accuracy_lr = accuracy_score(y_test,y_pred_lr)
train_accuracy_lr = accuracy_score(y_train,model_lr.predict(X_train_scaled))
cm_lr = confusion_matrix(y_test,y_pred_lr)
sns.heatmap(cm_lr, annot=True, cmap='coolwarm', fmt='d')
plt.show()
print(classification_report(y_test,y_pred_lr))
#print the results in a dataframe
results_lr = pd.DataFrame({
    'Model': ['Logistic Regression'],
    'Accuracy': [test_accuracy_lr],
    'Precision': [precision_score(y_test,y_pred_lr)],
    'Recall': [recall_score(y_test,y_pred_lr)],
    'F1 Score': [f1_score(y_test,y_pred_lr)]
})

results_lr.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Accuracy Scores')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scaled,y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
test_accuracy_rf = accuracy_score(y_test,y_pred_rf)
train_accuracy_rf = accuracy_score(y_train,model_rf.predict(X_train_scaled))
cm_rf = confusion_matrix(y_test,y_pred_rf)
sns.heatmap(cm_rf, annot=True, cmap='coolwarm', fmt='d')
plt.show()
print(classification_report(y_test,y_pred_rf))

#print the results in a dataframe
results_rf = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [test_accuracy_rf],
    'Precision': [precision_score(y_test,y_pred_rf)],
    'Recall': [recall_score(y_test,y_pred_rf)],
    'F1 Score': [f1_score(y_test,y_pred_rf)]
})
#use barolot with numbers in the y axis
results_rf.plot(kind='bar', figsize=(10, 6), ylim=(0, 1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Accuracy Scores')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


#Support Vector Machine
model_svm = SVC()
model_svm.fit(X_train_scaled,y_train)
y_pred_svm = model_svm.predict(X_test_scaled)
test_accuracy_svm = accuracy_score(y_test,y_pred_svm)
train_accuracy_svm = accuracy_score(y_train,model_svm.predict(X_train_scaled))
cm_svm = confusion_matrix(y_test,y_pred_svm)
sns.heatmap(cm_svm, annot=True, cmap='coolwarm', fmt='d')
plt.show()
print(classification_report(y_test,y_pred_svm))

#print the results in a dataframe
results_svm = pd.DataFrame({
    'Model': ['Support Vector Machine'],
    'Accuracy': [test_accuracy_svm],
    'Precision': [precision_score(y_test,y_pred_svm)],
    'Recall': [recall_score(y_test,y_pred_svm)],
    'F1 Score': [f1_score(y_test,y_pred_svm)]
})

results_svm.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Accuracy Scores')
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# do not expect much from the decision tree as i think it is not a good model for this dataset
# #Decision Tree
# model_dt = DecisionTreeClassifier()
# model_dt.fit(X_train_scaled,y_train)
# y_pred_dt = model_dt.predict(X_test_scaled)
# test_accuracy_dt = accuracy_score(y_test,y_pred_dt)
# train_accuracy_dt = accuracy_score(y_train,model_dt.predict(X_train_scaled))
# cm_dt = confusion_matrix(y_test,y_pred_dt)
# sns.heatmap(cm_dt, annot=True, cmap='coolwarm', fmt='d')
# plt.show()
# print(classification_report(y_test,y_pred_dt))

# #print the results in a dataframe
# results_dt = pd.DataFrame({
#     'Model': ['Decision Tree'],
#     'Accuracy': [test_accuracy_dt],
#     'Precision': [precision_score(y_test,y_pred_dt)],
#     'Recall': [recall_score(y_test,y_pred_dt)],
#     'F1 Score': [f1_score(y_test,y_pred_dt)]
# })

# results_dt.plot(kind='bar', figsize=(10, 6))
# plt.xlabel('Model')
# plt.ylabel('Scores')
# plt.title('Model Accuracy Scores')
# plt.xticks(rotation=45)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

