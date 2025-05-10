import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper as hp
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("StudentsPerformance.csv")


cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)

#EDA
df.head()
df.shape
df.info()
df[cat_cols].head()
df[num_cols].head()

df.isnull().sum()
#There is no null value. But need to check if there is specific value for null values

df.describe().T
#Because of limit of numeric variables there is no outlier

df[num_cols].corr()
#Writing and reading correlation is too high compared to other correlations

df["gender"].value_counts()
df.groupby("gender").agg({"math score": "mean",
                          "reading score": "mean",
                          "writing score": "mean"})
#In math males are more successful compared to female. But other two areas females more successful.

df.groupby(["gender", "race/ethnicity"]).agg({"math score": "mean",
                          "reading score": "mean",
                          "writing score": "mean"})
#In both gender group A race most unsuccessful in all areas. We can check group A parental level of education.
#group E race most successful group.**

df.groupby(["gender", "test preparation course"]).agg({"math score": "mean",
                          "reading score": "mean",
                          "writing score": "mean"})
#Test preparation course effect is obviously visible

df.groupby(["gender", "test preparation course", "race/ethnicity"]).agg({"math score": "mean",
                          "reading score": "mean",
                          "writing score": "mean"})
#Difference of group A's course completed ones and didnt one's. Course is probably most effective thing compared to others.
#But even from group E's who didnt attend course were able to get the same or even higher scores compared to group D or C,B,A

df.groupby(["parental level of education", "race/ethnicity"]).agg({"math score": "mean",
                                                           "reading score": "mean",
                                                           "writing score": "mean"})
#group E most successful group among in all groups.
#Parents who have bachelor's degree group E most successful in math.
#For math we can expect it would higher than high school or associate degree but more than master degree is interesting.
#Reading and writing score higher at master degree.
#Some groups are higher than group E at reading score or writing score. But there is not much difference.

df["parental level of education"].value_counts()
#Master degree count is low to others. This may increase scores bec of low population.
#I think it is normal.

df["race/ethnicity"].value_counts()
#Most successful group E count is lower than other 3 group. So their scores are really good. With low population they are first.

df.head()
df["test preparation course"].value_counts()
df.groupby("test preparation course").agg({"math score": "mean",
                                           "reading score": "mean",
                                           "writing score": "mean"})

pd.crosstab(df["parental level of education"], df["test preparation course"])
#Completed ones are always low compared to didnt one's. Maybe this course is not free or expensive.

df.head()
df["lunch"].value_counts()
pd.crosstab(df["lunch"], df["test preparation course"])
df.groupby(["parental level of education", "test preparation course", "lunch"]).size().unstack()

df.groupby(["lunch"]).agg({"math score": "mean",
                           "reading score": "mean",
                           "writing score": "mean"})
#The ones who are eating standart meal is more successful.

df.groupby(["lunch", "test preparation course"]).agg({"math score": "mean",
                           "reading score": "mean",
                           "writing score": "mean"})
#Meal is important thing too. As we can see the ones eating standart lunch and not attend course students almost win all areas.

df.groupby(["race/ethnicity", "lunch"]).agg({"math score": "mean",
                           "reading score": "mean",
                           "writing score": "mean"})
#Group E mostly gets highest score their own field.


##Summary of numeric and categoric columns
for col in num_cols:
    hp.num_summary(df,col, plot = True)
#distribution for numerical cols are good.

for col in cat_cols:
    hp.cat_summary(df, col, plot= True)
#distribution for categorical cols not bad.

for col in cat_cols:
    print(df[col].value_counts())
#I checked value_counts here for thing that I said at null value section.
#I wanted to look again if there is any different value for null values.


#FEATURE ENGINEERING
df["average_score"] = (df["math score"] + df["reading score"] + df["writing score"])//3
df["passed"] = [1 if value > 70 else 0 for value in df["average_score"]]

df.to_csv("students_cleaned.csv", index = False)
df.head()
cat_cols, num_cols, cat_but_car = hp.grab_col_names(df)

#Encoding
df["gender"] = [1 if value == "female" else 0 for value in df["gender"]]
df["lunch"] = [1 if value == "standard" else 0 for value in df["lunch"]]
df["test preparation course"] = [1 if value == "completed" else 0 for value in df["test preparation course"]]

df.head()
binary_cols = [col for col in cat_cols if (df[col].nunique() == 2)]
binary_cols

ohe_cols = [col for col in cat_cols if (col not in binary_cols)]
ohe_cols

df = pd.get_dummies(df, columns = ohe_cols, drop_first=True, dtype=int)
df.head()

#Scaling
sc = StandardScaler()
num_cols
df[num_cols] = sc.fit_transform(df[num_cols])
df.head()


##MODELLING
x = df.drop(["passed", "math score", "reading score", "writing score", "average_score"], axis = 1)
#Variables like average_score maybe usable in regression problem. But wont use it here
y = df["passed"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

#RandomForest
rf_model = RandomForestClassifier(random_state=42).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_pred, y_test)
print(confusion_matrix(y_test,y_pred))

#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)
accuracy_score(y_pred_lr, y_test)


importances = pd.Series(rf_model.feature_importances_, index=x.columns).sort_values(ascending=False)
importances.plot(kind='bar', figsize=(10,5))
plt.title("Feature Importance")
plt.show(block = True)

#This mini-project is just for weekly repetition. Just for remembering foundation.
#I wont do any model optimization here. Just little analyse and mini model