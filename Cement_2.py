import os
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir('C:\\WORK\\CementProject')

# csv Latin-1 data loaded
DataFrame = pd.read_csv(r"C:\WORK\CementProject\CONCRETEDATA.csv")
DataFrame.shape # 1030 by 9

# Check missing values, found none
DataFrame.isnull().sum()

# Making friends with data
DataFrame.describe()
DataFrame.info()

# Define your Matrix of Features, and Dependent Variable
x = DataFrame.iloc[:, :-1].values
y = DataFrame.iloc[:, -1].values

# We split first to prevent leakage, also, lost column names, but with 8, no issues. Do add them on larger Datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 13)

# Only now we scale, since we are not time travellers
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
StandardScaler.fit(x_train) 
x_train = pd.DataFrame(StandardScaler.transform(x_train))
x_test = pd.DataFrame(StandardScaler.transform(x_test))

# Let's just add a constant first, just in case
from statsmodels.api import add_constant
DataFrame = add_constant(DataFrame)
DataFrame.shape

# Knowing cements, I am suspicious of a couple of these features
# VIF for multicollinearity and remove threshhold very high, since only 8 features

Correlation = DataFrame.corr()
sns.heatmap(Correlation,cmap="rainbow",annot=True)


# See Appendix 1 for remarks on Correlation
# No variance_inflation_factor used, no feature removed TLDR

# Next, remove features with ridiculously low P-Value, though consider the scarcity of features
# OLS first, then your regular P-Value Cutoff whileloop
from statsmodels.api import OLS
m1ModelBuild = OLS(y_train, x_train).fit()
m1ModelBuild.summary()

dir(m1ModelBuild)
m1ModelBuild.pvalues

# Unremarkable P-Values given our domain knowledge

tempMaxPValue = 0.1
maxPValueCutoff = 0.1
train_x_copy = x_train.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    
    print(counter)    
    
    tempModelDf = pd.DataFrame()    
    Model = OLS(y_train, train_x_copy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = train_x_copy.columns
    tempModelDf.dropna(inplace=True) 
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): 
        print(tempColumnName, tempMaxPValue)    
        train_x_copy = train_x_copy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1

highPValueColumnNames

# P-Value filter recommends removing two IRL important columns
# Judgement call NO REMOVE, domain knowledge+column scarcity majors ML recommendation

# Proceed to model building
# I will implement vanilla regression models through a dictionary loop
# And choose my favorite model through that, to educate better. Fat chance I don't mess up


from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import GridSearchCV

models_dictionary = {"Linear Regression" : LinearRegression(),
                     "SVM (Linear)" : LinearSVR(),
                     "SVM (RBF)" : SVR(),
                     "Decision Tree" : DecisionTreeRegressor(),
                     "Random Forest" : RandomForestRegressor(),
                     "Gradient Boosting" : GradientBoostingRegressor(),
                     "AdaBoost" : AdaBoostRegressor()
                     }

# But, M'Lord, Polyno-YES, I'll loop it separately next.
# I have also added Gradient and ADA from Udemy.

for name, model in models_dictionary.items():
    model.fit(x_train, y_train)
    
# Polynomial Regression with its own loop
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

Poly_Scores = []
Poly = LinearRegression()
for i in np.arange(2,5): 
    degree = PolynomialFeatures(degree=i)
    x_polly = degree.fit_transform(x_train)
    x_polly_test = degree.fit_transform(x_test)
    Poly.fit(x_polly,y_train)
    y_head=Poly.predict(x_polly_test)
    Poly_Scores.append(r2_score(y_test,y_head))
    

    
# All models fitted successfully...and now?
# Now we look at their R2 and MAPE, find out which is best, and why it's Random Forest



for name, model in models_dictionary.items():
    print(name + "R^2 : {:.5f}".format(model.score(x_test,y_test)))
    
# Here are the results for ya convenience
# Linear RegressionR^2 :            0.50963
# SVM (Linear)R^2 :                 0.52408
# SVM (RBF)R^2 :                    0.55650
# Decision TreeR^2 :                0.75148
# Random ForestR^2 :                0.84055
# Gradient BoostingR^2 :            0.85954
# AdaBoostR^2 :                     0.72962

# Poly_Scores are as follows -
# Degree 2 gives                    0.7079507157090463
# Degree 3 gives                    0.843791596567056
# Degree 4 gives                    -6.235950858092608

# Random Forest and Gradient Boosting are fantastic. Poly2 okay enough, like me for my Dad
# Let's improve Gradient Boosting. Evict the rest. Like me for my okay sorry


# Optimizing Gradient Boosting finally, huge grid again
GBA_params = {'learning_rate' : [0.01, 0.1, 1.0],
          'n_estimators' : [50, 100, 150, 200],
          'max_depth' : [3, 4, 5, 6]
}

GBA_Model = GradientBoostingRegressor()
GBA_Model.fit(x_train, y_train)

GBA_Model = GridSearchCV(GBA_Model, GBA_params)
GBA_Model.fit(x_train, y_train)

# Found the best parameter
GBA_Model.best_params_
# 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200

print("Gradient Boosting Final Score ", (GBA_Model.score(x_test,y_test)))

# R2_Score before Grid Search was 85.9 %
# We improved it to 89.99 %, which is almost 90 %
# More on Appendix 2


# Let's just quickly measure up to see


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

GBA_y_pred = GBA_Model.predict(x_test)

r2_score(y_test, GBA_y_pred) #0.899 
mean_absolute_error(y_test, GBA_y_pred) # 3.26
mean_squared_error(y_test, GBA_y_pred) # 26.4
mean_absolute_percentage_error(y_test, GBA_y_pred) # 0.1

# More on this in Appendix 3



















