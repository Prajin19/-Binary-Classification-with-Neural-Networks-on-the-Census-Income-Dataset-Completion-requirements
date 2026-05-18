# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset-Completion-requirements

## Aim
We'll perform a binary classification on the Census Income dataset available from the UC Irvine Machine Learning Repository
The goal is to determine if an individual earns more than $50K based on a set of continuous and categorical variables.
## Algorithmic Steps (Theoretical Framework)
### Census Income Dataset
For this exercises we're using the Census Income dataset available from the <a href='http://archive.ics.uci.edu/ml/datasets/Adult'>UC Irvine Machine Learning Repository</a>.

The full dataset has 48,842 entries. For this exercise we have reduced the number of records, fields and field entries, and have removed entries with null values. The file <strong>income.csv</strong> has	30,000 entries

Each entry contains the following information about an individual:
* <strong>age</strong>: the age of an individual as an integer from 18 to 90 (continuous)
* <strong>sex</strong>: Male or Female (categorical)
* <strong>education</strong>: represents the highest level of education achieved by an individual (categorical)
* <strong>education_num</strong>: represents education as an integer from 3 to 16 (categorical)
<div><table style="display: inline-block">
<tr><td>3</td><td>5th-6th</td><td>8</td><td>12th</td><td>13</td><td>Bachelors</td></tr>
<tr><td>4</td><td>7th-8th</td><td>9</td><td>HS-grad</td><td>14</td><td>Masters</td></tr>
<tr><td>5</td><td>9th</td><td>10</td><td>Some-college</td><td>15</td><td>Prof-school</td></tr>
<tr><td>6</td><td>10th</td><td>11</td><td>Assoc-voc</td><td>16</td><td>Doctorate</td></tr>
<tr><td>7</td><td>11th</td><td>12</td><td>Assoc-acdm</td></tr>
</table></div>
* <strong>marital-status</strong>: marital status of an individual (categorical)
<div><table style="display: inline-block">
<tr><td>Married</td><td>Divorced</td><td>Married-spouse-absent</td></tr>
<tr><td>Separated</td><td>Widowed</td><td>Never-married</td></tr>
</table></div>
* <strong>workclass</strong>: a general term to represent the employment status of an individual (categorical)
<div><table style="display: inline-block">
<tr><td>Local-gov</td><td>Private</td></tr>
<tr><td>State-gov</td><td>Self-emp</td></tr>
<tr><td>Federal-gov</td></tr>
</table></div>
* <strong>occupation</strong>: the general type of occupation of an individual (categorical)
<div><table style="display: inline-block">
<tr><td>Adm-clerical</td><td>Handlers-cleaners</td><td>Protective-serv</td></tr>
<tr><td>Craft-repair</td><td>Machine-op-inspct</td><td>Sales</td></tr>
<tr><td>Exec-managerial</td><td>Other-service</td><td>Tech-support</td></tr>
<tr><td>Farming-fishing</td><td>Prof-specialty</td><td>Transport-moving</td></tr>
</table></div>
* <strong>hours-per-week</strong>: the hours an individual has reported to work per week as an integer from 20 to 90 (continuous)
* <strong>income</strong>: whether or not an individual makes more than \\$50,000 annually (label)
* <strong>label</strong>: income represented as an integer (0: <=\\$50K, 1: >\\$50K) (optional label)

#### 1. Separate continuous, categorical and label column names
You should find that there are 5 categorical columns, 2 continuous columns and 1 label.
In the case of education and education-num it doesn't matter which column you use. For the label column, be sure to use label and not income.
Assign the variable names "cat_cols", "cont_cols" and "y_col" to the lists of names.
#### 2. Convert categorical columns to category dtypes
#### 3. Set the embedding sizes
Create a variable "cat_szs" to hold the number of categories in each variable.
Then create a variable "emb_szs" to hold the list of (category size, embedding size) tuples.
#### 4. Create an array of categorical values
Create a NumPy array called "cats" that contains a stack of each categorical column .cat.codes.values
Note: your output may contain different values. Ours came after performing the shuffle step shown above.
#### 5. Convert "cats" to a tensor
Convert the "cats" NumPy array to a tensor of dtype int64
#### 6. Create an array of continuous values
Create a NumPy array called "conts" that contains a stack of each continuous column.
Note: your output may contain different values. Ours came after performing the shuffle step shown above.
#### 7. Convert "conts" to a tensor
Convert the "conts" NumPy array to a tensor of dtype float32
#### 8. Create a label tensor
Create a tensor called "y" from the values in the label column. Be sure to flatten the tensor so that it can be passed into the CE Loss function.
#### 9. Create train and test sets from cats, conts, and y
We use the entire batch of 30,000 records, but a smaller batch size will save time during training.
We used a test size of 5,000 records, but you can choose another fixed value or a percentage of the batch size.
Make sure that your test records remain separate from your training records, without overlap.
To make coding slices easier, we recommend assigning batch and test sizes to simple variables like "b" and "t".
#### 10.Define the model class
#### 11. Set the random seed
To obtain results that can be recreated, set a torch manual_seed (we used 33).
#### 12. Create a TabularModel instance
Create an instance called "model" with one hidden layer containing 50 neurons and a dropout layer p-value of 0.4
#### 13.Train the model
Run the cell below to train the model through 300 epochs. Remember, results may vary!
After completing the exercises, feel free to come back to this section and experiment with different parameters.
#### 14. Define the loss and optimization functions
Create a loss function called "criterion" using CrossEntropyLoss
Create an optimization function called "optimizer" using Adam, with a learning rate of 0.001
#### 15. Plot the Cross Entropy Loss against epochs
Results may vary. The shape of the plot is what matters.
#### 16. Evaluate the test set
With torch set to no_grad, pass cat_test and con_test through the trained model. Create a validation set called "y_val". Compare the output to y_test using the loss function defined above. Results may vary.
#### 17. Calculate the overall percent accuracy
Using a for loop, compare the argmax values of the y_val validation set to the y_test set.
