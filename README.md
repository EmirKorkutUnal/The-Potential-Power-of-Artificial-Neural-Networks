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
Alongside ANN, some other methods are used to create various models based on the same dataset. These are:<br><br>
<ul>
  <li><b>Linear Regression</b>: Tries to fit linear lines through your predictor and target variables. A relatively simple but effective tool.</li>
  <li><b>Generalized Additive Models</b>: Does the same thing as Linear Regression, plus uses some smoothing functions to increase accuracy.</li>
  <li><b>Polynomial Regression</b>: Uses polynomial functions instead of straight lines.</li>
</ul>
You can find more information on these methods <a href="https://www.google.com/search?ei=IsTkXLLOEfODk74Ps4uV6Ac&q=regression+types">all over the internet</a> and explaining these ascends the scope of this article.
<h3>Comparison</h3>
At the end of this article, all models will be compared for prediction accuracy. This will be done both by calculating total errors of models, and by graphical comparisın of all models versus the ideal prediction line.<br>
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
from sklearn.metrics import r2_score

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
<pre>
LinearModel = sm.OLS(y_train,x_train1).fit()
LinearModel.summary()
</pre>
<table class="simpletable">
<caption>OLS Regression Results</caption>
<tbody><tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.904</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.904</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>4.255e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 22 May 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>00:14:18</td>     <th>  Log-Likelihood:    </th> <td>-2.3022e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 26970</td>      <th>  AIC:               </th>  <td>4.604e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 26963</td>      <th>  BIC:               </th>  <td>4.605e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</tbody></table>

<table class="simpletable">
<tbody><tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P&gt;|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>-2977.5273</td> <td>  549.910</td> <td>   -5.415</td> <td> 0.000</td> <td>-4055.380</td> <td>-1899.674</td>
</tr>
<tr>
  <th>carat</th>   <td> 8788.0993</td> <td>   18.068</td> <td>  486.393</td> <td> 0.000</td> <td> 8752.685</td> <td> 8823.513</td>
</tr>
<tr>
  <th>cut</th>     <td>  115.6685</td> <td>    8.228</td> <td>   14.058</td> <td> 0.000</td> <td>   99.541</td> <td>  131.796</td>
</tr>
<tr>
  <th>color</th>   <td>  312.8370</td> <td>    4.679</td> <td>   66.863</td> <td> 0.000</td> <td>  303.666</td> <td>  322.008</td>
</tr>
<tr>
  <th>clarity</th> <td>  513.7157</td> <td>    4.995</td> <td>  102.839</td> <td> 0.000</td> <td>  503.925</td> <td>  523.507</td>
</tr>
<tr>
  <th>depth</th>   <td>  -54.3952</td> <td>    6.019</td> <td>   -9.037</td> <td> 0.000</td> <td>  -66.193</td> <td>  -42.597</td>
</tr>
<tr>
  <th>table</th>   <td>  -29.3051</td> <td>    4.243</td> <td>   -6.906</td> <td> 0.000</td> <td>  -37.622</td> <td>  -20.988</td>
</tr>
</tbody></table>
With all variables having a meaningful effect, the model achieves a 0.904 R-squared; more that 90% of all price variance is explained by these variables in this model. This is a very good score. Let's see if we can get something better.
<h2>GAM</h2>
<pre>
gam = LinearGAM(terms='auto').fit(x_train1, y_train)
gam.summary()
</pre>
<table>
<tbody>
<tr>
<td colspan="8" width="623">
<p><b>LinearGAM*</b></p>
</td>
</tr>
<tr>
<td colspan="2" width="176">
<p>Distribution:</p>
<p>Link Function:</p>
<p>Number of Samples:</p>
</td>
<td colspan="2" width="147">
<p>Normal Dist</p>
<p>IdentityLink</p>
<p>26970</p>
</td>
<td colspan="2" width="156">
<p>Effective DoF:</p>
<p>Log Likelihood:</p>
<p>AIC:</p>
<p>AICc:</p>
<p>GCV:</p>
<p>Scale:</p>
<p>Pseudo</p>
</td>
<td colspan="2" width="144">
<p>49.4536</p>
<p>-399277.4379</p>
<p>798655.783</p>
<p>798655.9759</p>
<p>1076116.4033</p>
<p>1072565.1844</p>
<p>R-Squared:</p>
</td>
</tr>
<tr>
<td width="121">
<p>Feature Function</p>
</td>
<td colspan="2" width="97">
<p>Lambda</p>
</td>
<td width="105">
<p>Rank</p>
</td>
<td width="66">
<p>EDoF</p>
</td>
<td colspan="2" width="128">
<p>P &gt; x</p>
</td>
<td width="106">
<p>Sig. Code</p>
</td>
</tr>
<tr>
<td width="121">
<p>s(0)</p>
<p>s(1)</p>
<p>s(2)</p>
<p>s(3)</p>
<p>s(4)</p>
<p>s(5)</p>
<p>s(6)</p>
<p>intercept</p>
</td>
<td colspan="2" width="97">
<p>[0.6]</p>
<p>[0.6]</p>
<p>[0.6]</p>
<p>[0.6]</p>
<p>[0.6]</p>
<p>[0.6]</p>
<p>[0.6]</p>
</td>
<td width="105">
<p>20</p>
<p>20</p>
<p>20</p>
<p>20</p>
<p>20</p>
<p>20</p>
<p>20</p>
<p>1</p>
</td>
<td width="68">
<p>1.0</p>
<p>12.3</p>
<p>4.0</p>
<p>6.0</p>
<p>7.0</p>
<p>10.5</p>
<p>8.7</p>
<p>0.0</p>
</td>
<td colspan="2" width="126">
<p>1.11e-16</p>
<p>1.11e-16</p>
<p>1.67e-10</p>
<p>1.11e-16</p>
<p>1.11e-16</p>
<p>1.11e-16</p>
<p>1.49e-09</p>
<p>1.11e-16</p>
</td>
<td width="106">
<p>***</p>
<p>***</p>
<p>***</p>
<p>***</p>
<p>***</p>
<p>***</p>
<p>***</p>
<p>***</p>
</td>
</tr>
<tr>
<td colspan="8" width="623">
<p>Significance codes:&nbsp; 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1</p>
</td>
</tr>
</tbody>
</table>
* LinearGAM doesn't produce a beautiful output table. You can copy the output to Word, create a table and save it as an html to get this kind of a table.<br><br>
The R-squared has risen to 0.933. This model seems to have better predictive power than the previous one.
<h2>Polynomial Regression</h2>
<pre>
PolynomModel = PolynomialFeatures(degree=3)
x_train_model = PolynomModel.fit_transform(x_train)
x_test_model = PolynomModel.fit_transform(x_test)
PolyReg = LinearRegression()
PolyReg.fit(x_train_model,y_train)
y_train_pred = PolyReg.predict(x_train_model)
</pre>
<pre>
r2_score(y_train, y_train_pred)
0.972253547295967
</pre>
This is only getting better! Our Polynomial Regression model has achieved an R-squared of 0.972.<br>
Here is a Regression up to the 3rd degree is used. Using a higher degree would cause overfitting, which in turn would reduce the predictive power of the model on other databases.
<h2>Artifical Neural Network</h2>
Here comes the challenger.
<pre>
scaler = MinMaxScaler()
scaler.fit(x)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
</pre>
Data is scaled to make the ANN work faster.
<pre>
model = Sequential()
model.add(Dense(240, activation='selu', kernel_initializer='VarianceScaling', input_shape=(x_train.shape[1],)))
model.add(Dense(200, activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(160, activation='selu'))
model.add(Dense(1))
</pre>
Our model has 3 dense layers and one dropout layer. More information about these parameters can be found <a href="https://keras.io/">here</a>.

