import pandas as pd
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

to_kg = lambda x: x * 0.453592
to_cm = lambda x: x * 2.54

df = pd.read_csv('weight-height.csv')
print(df)
df.Height = df.Height.apply(to_cm)
df.Weight = df.Weight.apply(to_kg)
print(df)
male = df[df.Gender == "Male"]
female = df[df.Gender == "Female"]



def l_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    linear = LinearRegression()
    linear.fit(x_train, y_train)
    y_linear = linear.predict(x_test)
    return y_test, y_linear


X_m = pd.DataFrame(columns={"Height"})
Y_m = pd.DataFrame(columns={"Weight"})
X_f = pd.DataFrame(columns={"Height"})
Y_f = pd.DataFrame(columns={"Weight"})

X_m.Height = male.Height
Y_m.Weight = male.Weight
X_f.Height = female.Height
Y_f.Weight = female.Weight

yt_male, yl_male = l_regression(X_m, Y_m)
yt_female, yl_female = l_regression(X_f, Y_f)

pyplot.scatter(yt_male, yl_male)
pyplot.show()
pyplot.scatter(yt_female, yl_female)
pyplot.show()

print("Среднеквадратичная ошибка м " + str(metrics.mean_absolute_error(yt_male, yl_male)))
print("Кэффициент дискриминации м " + str(metrics.r2_score(yt_male, yl_male)))
print("Среднеквадратичная ошибка ж " + str(metrics.mean_absolute_error(yt_female, yl_female)))
print("Кэффициент дискриминации ж " + str(metrics.r2_score(yt_female, yl_female)))
