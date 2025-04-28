# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

h = np.array([6.2, 5.8, 5.4, 5.0, 4.6, 4.2, 3.8, 3.4, 3.0])

T = np.array([1.11, 1.11, 1.06, 0.99, 1.0, 0.95, 0.87, 0.82, 0.80])
T2 = T**2

regr = LinearRegression()
regr.fit(X=T2.reshape(-1,1), y=h)

h_pred = regr.predict(T2.reshape(-1,1))

print(f"y-Achsenabschnitt: {regr.intercept_}")
print(f"Steigung: {regr.coef_}")
print(f"g: {2*regr.coef_}")
print(f"R^2-Score: {regr.score(T2.reshape(-1,1), h)}")

fig, ax = plt.subplots()
ax.scatter(x=T2, y=h, label="Messung 2")
ax.plot(T2, h_pred, "r--", label=r"Lin. Regression")
ax.set(xlabel=r"Fallzeit $T^2$ / s$^2$", ylabel=r"Höhe $h$ / m")
ax.legend()
ax.grid(True)