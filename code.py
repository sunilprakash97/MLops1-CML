import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
seed = 42

df = pd.read_csv("wine_quality.csv")
# df.describe()
# df.info()
# df.head(10)

y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

model = RandomForestRegressor(max_depth = 5, random_state = seed)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)*100
# print(train_score)
test_score = model.score(X_test, y_test)
# print(test_score)

with open("metrics.txt", 'w') as outfile:
    outfile.write("Training Variance: %2.1f%%\n" % train_score)
    outfile.write("Test Variance: %2.1f%%\n" % test_score)

# Feature Importance
importance = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importance)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by = 'importance', ascending = False)

# Image formatting Font size
axis_fs = 18 
title_fs = 22 
sns.set(style = "whitegrid")

# Feature Importance Plot
ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()

# Plot Residuals
# How far true qualities of predicted from Actual

y_pred = model.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

# Residual Plot

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted wine quality', fontsize = axis_fs)
ax.set_title('Residuals', fontsize = title_fs)

ax.plot([1, 10], [1, 10], 'black', linewidth = 1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 
plt.close()
