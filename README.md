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
You can find more information on these methods <a href="https://www.google.com/search?ei=IsTkXLLOEfODk74Ps4uV6Ac&q=regression+types">all over the internet</a>.<br>
Let's start.
<h2>Loading Modules and Dataset in Jupyter</h2>
<pre>
# General tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Linear Regression
import statsmodels.api as sm
from statsmodels.stats import outliers_influence

# GAM
from pygam import LinearGAM, s, f, te

# Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Artifical Neural Network
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import initializers
</pre>

<link rel="stylesheet" href="/static/components/codemirror/lib/codemirror.css?v=288352df06a67ee35003b0981da414ac">

<div class="CodeMirror-code" role="presentation" style=""><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># General tools</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">pandas</span> <span class="cm-keyword">as</span> <span class="cm-variable">pd</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">numpy</span> <span class="cm-keyword">as</span> <span class="cm-variable">np</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">matplotlib</span>.<span class="cm-property">pyplot</span> <span class="cm-keyword">as</span> <span class="cm-variable">plt</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">model_selection</span> <span class="cm-keyword">import</span> <span class="cm-variable">train_test_split</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Linear Regression</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">import</span> <span class="cm-variable">statsmodels</span>.<span class="cm-property">api</span> <span class="cm-keyword">as</span> <span class="cm-variable">sm</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">statsmodels</span>.<span class="cm-property">stats</span> <span class="cm-keyword">import</span> <span class="cm-variable">outliers_influence</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># GAM</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">pygam</span> <span class="cm-keyword">import</span> <span class="cm-variable">LinearGAM</span>, <span class="cm-variable">s</span>, <span class="cm-variable">f</span>, <span class="cm-variable">te</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Polynomial Regression</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">linear_model</span> <span class="cm-keyword">import</span> <span class="cm-variable">LinearRegression</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">preprocessing</span> <span class="cm-keyword">import</span> <span class="cm-variable">PolynomialFeatures</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span cm-text="">​</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-comment"># Artifical Neural Network</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">sklearn</span>.<span class="cm-property">preprocessing</span> <span class="cm-keyword">import</span> <span class="cm-variable">MinMaxScaler</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">models</span> <span class="cm-keyword">import</span> <span class="cm-variable">Sequential</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">layers</span> <span class="cm-keyword">import</span> <span class="cm-variable">Dense</span>, <span class="cm-variable">Dropout</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span>.<span class="cm-property">callbacks</span> <span class="cm-keyword">import</span> <span class="cm-variable">EarlyStopping</span></span></pre><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-keyword">from</span> <span class="cm-variable">keras</span> <span class="cm-keyword">import</span> <span class="cm-variable">initializers</span></span></pre></div>
