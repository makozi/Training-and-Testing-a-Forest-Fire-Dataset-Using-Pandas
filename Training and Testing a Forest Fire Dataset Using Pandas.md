

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data=pd.read_csv('forestfires.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>temp</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>5</td>
      <td>mar</td>
      <td>fri</td>
      <td>86.2</td>
      <td>26.2</td>
      <td>94.3</td>
      <td>5.1</td>
      <td>8.2</td>
      <td>51</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>4</td>
      <td>oct</td>
      <td>tue</td>
      <td>90.6</td>
      <td>35.4</td>
      <td>669.1</td>
      <td>6.7</td>
      <td>18.0</td>
      <td>33</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>4</td>
      <td>oct</td>
      <td>sat</td>
      <td>90.6</td>
      <td>43.7</td>
      <td>686.9</td>
      <td>6.7</td>
      <td>14.6</td>
      <td>33</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>6</td>
      <td>mar</td>
      <td>fri</td>
      <td>91.7</td>
      <td>33.3</td>
      <td>77.5</td>
      <td>9.0</td>
      <td>8.3</td>
      <td>97</td>
      <td>4.0</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>6</td>
      <td>mar</td>
      <td>sun</td>
      <td>89.3</td>
      <td>51.3</td>
      <td>102.2</td>
      <td>9.6</td>
      <td>11.4</td>
      <td>99</td>
      <td>1.8</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y=data.temp
x=data.drop('temp',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>156</th>
      <td>2</td>
      <td>4</td>
      <td>sep</td>
      <td>sat</td>
      <td>93.4</td>
      <td>145.4</td>
      <td>721.4</td>
      <td>8.1</td>
      <td>27</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>1.61</td>
    </tr>
    <tr>
      <th>482</th>
      <td>3</td>
      <td>4</td>
      <td>aug</td>
      <td>sun</td>
      <td>94.9</td>
      <td>130.3</td>
      <td>587.1</td>
      <td>14.1</td>
      <td>40</td>
      <td>5.8</td>
      <td>0.0</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>488</th>
      <td>4</td>
      <td>4</td>
      <td>aug</td>
      <td>tue</td>
      <td>95.1</td>
      <td>141.3</td>
      <td>605.8</td>
      <td>17.7</td>
      <td>71</td>
      <td>7.6</td>
      <td>0.0</td>
      <td>46.70</td>
    </tr>
    <tr>
      <th>221</th>
      <td>3</td>
      <td>4</td>
      <td>sep</td>
      <td>fri</td>
      <td>93.3</td>
      <td>141.2</td>
      <td>713.9</td>
      <td>13.9</td>
      <td>49</td>
      <td>3.6</td>
      <td>0.0</td>
      <td>35.88</td>
    </tr>
    <tr>
      <th>299</th>
      <td>6</td>
      <td>5</td>
      <td>jun</td>
      <td>sat</td>
      <td>53.4</td>
      <td>71.0</td>
      <td>233.8</td>
      <td>0.4</td>
      <td>90</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train.shape
```




    (413, 12)




```python
x_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>month</th>
      <th>day</th>
      <th>FFMC</th>
      <th>DMC</th>
      <th>DC</th>
      <th>ISI</th>
      <th>RH</th>
      <th>wind</th>
      <th>rain</th>
      <th>area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>295</th>
      <td>7</td>
      <td>5</td>
      <td>jun</td>
      <td>sun</td>
      <td>93.1</td>
      <td>180.4</td>
      <td>430.8</td>
      <td>11.0</td>
      <td>48</td>
      <td>1.3</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>234</th>
      <td>4</td>
      <td>5</td>
      <td>sep</td>
      <td>sat</td>
      <td>92.5</td>
      <td>121.1</td>
      <td>674.4</td>
      <td>8.6</td>
      <td>25</td>
      <td>3.1</td>
      <td>0.0</td>
      <td>154.88</td>
    </tr>
    <tr>
      <th>130</th>
      <td>4</td>
      <td>6</td>
      <td>feb</td>
      <td>sat</td>
      <td>68.2</td>
      <td>21.5</td>
      <td>87.2</td>
      <td>0.8</td>
      <td>40</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>188</th>
      <td>6</td>
      <td>4</td>
      <td>mar</td>
      <td>sat</td>
      <td>90.8</td>
      <td>41.9</td>
      <td>89.4</td>
      <td>7.9</td>
      <td>42</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>7.40</td>
    </tr>
    <tr>
      <th>432</th>
      <td>8</td>
      <td>6</td>
      <td>aug</td>
      <td>thu</td>
      <td>94.8</td>
      <td>222.4</td>
      <td>698.6</td>
      <td>13.9</td>
      <td>38</td>
      <td>6.7</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



The line test_size=0.2 suggests that the test data should be 20% of the dataset and the rest should be train data. With the outputs of the shape() functions, you can see that we have 104 rows in the test data and 413 in the training data.



```python
x_test.shape
```




    (104, 12)




```python

```
