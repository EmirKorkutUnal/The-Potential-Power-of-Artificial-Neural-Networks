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
On this article, we'll go through a dataset that contains features of various diamonds. Our target variable will be <b>price</b>. The categorical variables can be turned into interval variables which makes our job easier. You can download this Kaggle dataset <a href="https://www.kaggle.com/shivam2503/diamonds/downloads/diamonds.zip/1">here</a>.
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
<h3>Preparing data: Turning categorical variables into interval variables, and dropping index column</h3>
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
</pre>
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
<h3>Plotting data</h3>
<pre>
pd.plotting.scatter_matrix(df, figsize=(20, 20), marker='o', s=15)
plt.show()        # You don't need this line for the code to work. It simply suppresses creation details of individual graphs which you probably don't need.
</pre>
<img src="https://github.com/EmirKorkutUnal/The-Potential-Power-of-Artificial-Neural-Networks/blob/master/Images/PairPlots.png">
We can already see some correlations between our target variable price and other variables. At the first glance, Polynomial Regression seems to fit this dataset better than Linear Regression because of curvilinear correlations of price with carat, x, y, and z variables; but we'll use all our methods and let the numbers play a decisive role rather than our geometry skills.
<h3>Defining target and predictors</h3>
<pre>
y = df.filter(['price'])
x = df.drop(['price'], axis=1)
</pre><br>
For Linear Regression to work properly, we need a constant variable. We add this with the following line:
<pre>
x1 = sm.add_constant(x)
</pre>
<h2>Removing Collinearity for Linear Regression and GAM</h2>
When using Linear Regression or GAM, collinear variables alter the model's ability to interpret variable importance, so it's wise to remove variables which cause collinearity.
<pre>
vif = pd.DataFrame()
vif["VIF Factor"] = [outliers_influence.variance_inflation_factor(x1.values, i) for i in range(x1.shape[1])]
vif["features"] = x1.columns
vif.sort_values('VIF Factor', inplace=True, ascending=False)
vif.round(1)
</pre>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6737.0</td>
      <td>const</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56.9</td>
      <td>x</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23.5</td>
      <td>z</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>carat</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20.5</td>
      <td>y</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.7</td>
      <td>depth</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.6</td>
      <td>table</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>cut</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.2</td>
      <td>clarity</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>color</td>
    </tr>
  </tbody>
</table>
<br>
After 3 rounds of VIF calculation, variables <b>x, z, and y</b> are left out of model. x and z had the highest variance inflation on their respective rounds, and the variable <b>carat</b> provides more information than y by itself so we're including carat into the model rather than y.<br>
Final collinearity table looks like this:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5334.2</td>
      <td>const</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.6</td>
      <td>table</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>cut</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.3</td>
      <td>depth</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.3</td>
      <td>carat</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.2</td>
      <td>clarity</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>color</td>
    </tr>
  </tbody>
</table>
<h3>Train Test Split</h3>
This tool is used for splitting databases into two groups, using one part to train the model and the other to test the model on.<br>
We need train and test data with constants for Linear Regression and GAM, and without constants for Polynomial Regression and ANN. Because of this, we're splitting the data without the constants and adding constants afterwards. Alternatively, you can make two splits (one with and one without constants); if you want to go that way, remember to use the same random_state for both splits for a fair comparison. Random state parameter accepts integers.
<pre>
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=7)
x_train1 = sm.add_constant(x_train)
x_test1 = sm.add_constant(x_test)
</pre>
<h2>Linear Regression</h2>
