
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("C:/Users/chauh/Downloads/titanic/train.csv")

# Show data
print(df.head())
print(df.isnull().sum())

# Cleaning
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Analysis
gender_survival = df.groupby('Sex')['Survived'].sum()
print("\nSurvival by Gender:\n", gender_survival)

class_survival = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival by Class:\n", class_survival)

# Age groups
bins = [0, 12, 20, 40, 60, 100]
labels = ['Child', 'Teen', 'Adult', 'Middle Age', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

age_survival = df.groupby('AgeGroup')['Survived'].mean()
print("\nSurvival by Age Group:\n", age_survival)

# Visualizations
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

plt.hist(df['Age'], bins=20)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()