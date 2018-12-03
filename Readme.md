# Data Mining: Naive Bayes and Decision Tree Classifiers
- **Naive Bayes and Decision Tree Classifiers Implemented with Scikit-Learn and Graphviz Visualization**
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_NEWS.png)
- Datasets:
    - News (subset of 20 Newsgroups dataset, with testing label)
    - Mushroom (with testing label)
    - Income (UCI Adult Income dataset, with no testing label)


## Environment
* **< scikit-learn 0.20.1 >**
* **< numpy 1.15.4 >**
* **< pandas 0.23.4 >**
* **< Python 3.7 >**
* **< tqdm 4.28.1 >** (optional - progress bar)
* **< graphviz 0.10.1 >** (optional - visualization)
 

## File Description
```
.
├── src/
|   ├── classifiers.py ----------> Implementation of the naive bayes and decision tree classifiers
|   ├── data_loader.py ----------> Data loader that handles the reading and preprocessing of all 3 datasets
|   └── runner.py ---------------> Runner that runs all modes: train + evaluate, search optimal model, visualize model, etc.
├── data/ -----------------------> unzip data.zip
|   ├── income
|   |   ├── income_test.csv
|   |   ├── income_train.csv
|   |   ├── income.names
|   |   └── sample_output.csv
|   ├── mushroom
|   |   ├── mushroom_test.csv
|   |   ├── mushroom_train.csv
|   |   ├── mushroom.names
|   |   └── sample_output.csv
|   └── news
|       ├── news_test.csv
|       ├── news_train.csv
|       └── sample_output.csv
├── image/ ----------------------> visualization and program output screen shots
├── result/ ---------------------> model prediction output
├── problem_description.pdf -----> Work spec
└── Readme.md -------------------> This file
```


## Usage
### Data
- Unzip `data.zip` with: `unzip data.zip`

### Naive Bayes Classifier
- Train and test with the best **alpha** parameter for the **best** distribution assumption of the Naive Bayes classifier:
    - News dataset: `python3 runner.py --naive_bayes --data_news`
    - Mushroom dataset: `python3 runner.py --naive_bayes --data_mushroom`
    - Income dataset: `python3 runner.py --naive_bayes --data_income`

- Search for the best **alpha** parameter for **each** distribution assumption of the Naive Bayes classifier:
    - Add the `--search_opt` argument
    - News dataset (validated on the testing set): 
    ```
    python3 runner.py --naive_bayes --search_opt --data_news
    ``` 
    - Mushroom dataset (validated on the testing set):
    ```
    python3 runner.py --naive_bayes --search_opt --data_mushroom
    ```
    - Income dataset (Using N-fold cross-validation on the training set): 
    ```
    python3 runner.py --naive_bayes --search_opt --data_income
    ``` 

- Compare **all** distribution assumption of the Naive Bayes classifier with their own best **alpha** parameter:
    - Add the `--run_all` argument
    - News dataset: `python3 runner.py --naive_bayes --run_all --data_news`
    - Mushroom dataset: `python3 runner.py --naive_bayes --run_all --data_mushroom`
    - Income dataset: `python3 runner.py --naive_bayes --run_all --data_income`


### Decision Tree Classifier
- Train and test with the best **max depth** parameter for the Decision Tree classifier:
    - News dataset: `python3 runner.py --decision_tree --data_news`
    - Mushroom dataset: `python3 runner.py --decision_tree --data_mushroom`
    - Income dataset: `python3 runner.py --decision_tree --data_income`

- Search the best **max depth** parameter for the Decision Tree classifier:
    - Add the `--search_opt` argument
    - News dataset (validated on the testing set): 
    ```
    python3 runner.py --decision_tree --search_opt --data_news
    ```
    - Mushroom dataset (validated on the testing set): 
    ```
    python3 runner.py --decision_tree --search_opt --data_mushroom
    ``` 
    - Income dataset (Using N-fold cross-validation on the training set): 
    ```
    python3 runner.py --decision_tree --search_opt --data_income
    ``` 

- Visualize the Decision Tree classifier with the best **max depth** parameter:
    - Add the `--visualize_tree` argument
    - News dataset: `python3 runner.py --decision_tree --visualize_tree --data_news`
    - Mushroom dataset: `python3 runner.py --decision_tree --visualize_tree --data_mushroom`
    - Income dataset: `python3 runner.py --decision_tree --visualize_tree --data_income`


## Result - Naive Bayes Performance
### News Dataset - Testing Set Acc
- naive_bayes.GaussianNB() => 0.80979 (baseline)
- **naive_bayes.MultinomialNB(alpha=0.065)** => **0.89511**
- naive_bayes.ComplementNB(alpha=0.136) => 0.88811
- naive_bayes.BernoulliNB(alpha=0.002) => 0.82727
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_news.png width="546" height="70">

### Mushroom Dataset - Testing Set Acc
- naive_bayes.GaussianNB() => 0.95505 (baseline)
- **naive_bayes.MultinomialNB(alpha=0.0001)** => **0.99569**
- naive_bayes.ComplementNB(alpha=0.0001) => 0.99507
- naive_bayes.BernoulliNB(alpha=0.0001) => 0.98830
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_mushroom.png width="547" height="68">

### Income Dataset - N-Fold Cross-Validation Acc
- naive_bayes.GaussianNB() => 0.58602 (baseline)
- **naive_bayes.MultinomialNB(alpha=0.959)** => **0.79148**
- naive_bayes.ComplementNB(alpha=0.16) => 0.74992
- naive_bayes.BernoulliNB(alpha=0.001) => 0.75760
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_income.png width="547" height="70">

## Result - Decision Tree Performance
### News Dataset - Testing Set Acc
- tree.DecisionTreeClassifier(criterion='gini', splitter='random', random_state=1337, **max_depth=64**) => **0.64895**
- <img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_news.png width="458" height="20">
- decision tree visualization with the graphviz toolkit:
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_NEWS.png>

### Mushroom Dataset - Testing Set Acc
- tree.DecisionTreeClassifier(criterion='gini', splitter='random', random_state=1337, **max_depth=64**) => **1.0**
- <img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_mushroom.png width="345" height="20">
- decision tree visualization with the graphviz toolkit:
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_MUSHROOM_.png>

### Income Dataset - N-Fold Cross-Validation Acc
- tree.DecisionTreeClassifier(criterion='entropy', **max_depth=15**, min_impurity_decrease=2e-4) => **0.83554**
- <img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_income.png width="459" height="20">
- decision tree visualization with the graphviz toolkit:
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_INCOME_2.png>


## Data Preprocessing
### News Dataset Preprocessing
- None, raw input

### Mushroom Dataset Preprocessing
- 22 categorical attributes are transformed into a 117 dimension one-hot feature vector
- Resulting data shape:
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/mushroom_preprocessing.png width="394" height="86">

### Income Dataset Preprocessing
- Specify each entry to either one of the data type: (int, str)
- Identify all missing entries `'?'` and replace them with `np.nan`
- Impute and estimate all missing entries:
    - If dtype is `int`: impute with mean value of the feature column
    - If dtype is `str`: impute with most frequent item in the feature column
- Split data into categorical and continuous and process them separately:
    - categorical features index = [1, 3, 5, 6, 7, 8, 9, 13]
    - continuous features index = [0, 2, 4, 10, 11, 12]
- For categorical data:
    - 8 categorical attributes are transformed into a 99 dimension one-hot feature vector
- For continuous data:
    - Normalize with maximum norm of that feature column
- Re-concatenate categorical features and continuous features, the resulting data shape:
<img src=https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/income_preprocessing.png width="423" height="70">

