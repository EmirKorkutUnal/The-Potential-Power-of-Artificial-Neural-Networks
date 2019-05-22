<h1>The Potential Power of Artificial Neural Networks</h1>
A report about the power of Artifical Neural Networks and how they compare with Linear Regression, Generalized Additive Models, and Polynomial Regression withon Python.
<h2>Brief Description</h2>
Since it's not my style to write complex descriptions which also would bore me, I'm going to keep this part short.<br>
<b>Artifical Neural Networks</b> is the name given to a method use for modeling data. Just like any other data modeling method, you can train an ANN to make predictions.<br><br>
Your input goes through a series of <b>"hidden layers"</b> to create an output. Unlike most other models, the evaluation and interpretation processes of variables use changing weights; this is why these models are called <b>"Black box models"</b> - because you don't precisely know what's going on inside your hidden layers. The limits for the changing variable weights can be found out, but the exact way to evaluate and calculate each output remains a mystery.<br><br>
Let's see the potential power of these models compared to other methods.
<h2>DISCLAIMER</h2>
<b>Each and every dataset has its own unique characteristics. Remember, what works for your data is best for your data - you don't always need a more complicated model than you already have.</b>
<h2>Methodology</h2>
<h3>Data</h3>
On this article, we'll go through a dataset that contains features of various diamonds. The categorical variables can be turned into interval variables which makes our job easier. You can download this Kaggle dataset <a href="https://www.kaggle.com/shivam2503/diamonds/downloads/diamonds.zip/1">here</a>.
<h3>Models</h3>
Alongside ANN, some other methods are used to create a model based on the dataset. These are:<br><br>
<ul>
  <li><b>Linear Regression</b>: Tries to fit linear lines through your predictor and target variables. A relatively simple but effective tool.</li>
  <li><b>Generalized Additive Models</b>: Does the same thing as Linear Regression, plus uses some smoothing functions to increase accuracy.</li>
  <li><b>Polynomial Regression</b>: Uses polynomial functions instead of straight lines.</li>
</ul>
You can find more information on these methods <a href="https://www.google.com/search?ei=IsTkXLLOEfODk74Ps4uV6Ac&q=regression+types">all over the internet</a> and explaining these ascends the scope of this article.<br>
Let's start.
<h2>Loading Modules and Dataset in Jupyter</h2>
<pre>
import pandas as pd                                          # General tools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import statsmodels.api as sm                                 # Linear Regression
from statsmodels.stats import outliers_influence

from pygam import LinearGAM, s, f, te                        # GAM

from sklearn.linear_model import LinearRegression            # Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import MinMaxScaler               # Artifical Neural Network
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import initializers
</pre>
<pre>
df = pd.read_csv('C:/Users/Emir/Desktop/diamonds.csv')
df.head()
</pre>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
<h3>Preparing data: Turning categorical variables into interval variables, and dropping inxed column</h3>
<pre>
df['cut']=np.where(df['cut'] == 'Ideal', 5,
          np.where(df['cut'] == 'Premium', 4,
          np.where(df['cut'] == 'Very Good', 3,
          np.where(df['cut'] == 'Good', 2,
          np.where(df['cut'] == 'Fair', 1, 0)))))
df['color']=np.where(df['color'] == 'D', 7,
            np.where(df['color'] == 'E', 6,
            np.where(df['color'] == 'F', 5,
            np.where(df['color'] == 'G', 4,
            np.where(df['color'] == 'H', 3,
            np.where(df['color'] == 'I', 2,
            np.where(df['color'] == 'J', 1, 0)))))))
df['clarity']=np.where(df['clarity'] == 'FL', 11,
              np.where(df['clarity'] == 'IF', 10,
              np.where(df['clarity'] == 'VVS1', 9,
              np.where(df['clarity'] == 'VVS2', 8,
              np.where(df['clarity'] == 'VS1', 7,
              np.where(df['clarity'] == 'VS2', 6,
              np.where(df['clarity'] == 'SI1', 5,
              np.where(df['clarity'] == 'SI2', 4,
              np.where(df['clarity'] == 'I1', 3,
              np.where(df['clarity'] == 'I2', 2,
              np.where(df['clarity'] == 'I3', 1, 0)))))))))))
df.drop(['Unnamed: 0'], axis=1, inplace=True)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>5</td>
      <td>6</td>
      <td>4</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>2</td>
      <td>6</td>
      <td>7</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
