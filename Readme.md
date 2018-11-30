# Data Mining: Naive Bayes and Decision Tree Classifiers
- **Naive Bayes and Decision Tree Classifiers implemented with Scikit-Learn**
- Datasets:
    - News (subset of 20 Newsgroups dataset, with testing label)
    - Mushroom (with testing label)
    - Income (UCI Adult Income dataset, with no testing label)


## Environment
* **< scikit-learn 0.20.1 >**
* **< numpy 1.15.4 >**
* **< pandas 0.23.4 >**
* **< Python 3.7 >**
* **< tqdm 4.28.1 >** (optional)
* **< graphviz 0.10.1 >** (optional)
 

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

## Data Preprocessing
### News Dataset Preprocessing
- None, raw input

### Mushroom Dataset Preprocessing
- 22 categorical attributes are transformed into a 117 dimension one-hot vector
- Resulting data shape:
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/mushroom_preprocessing.png)

### Income Dataset Preprocessing
- Specify each entry to either one of the data type: (int, str)
- Identify all missing entries `'?'` and replace them with `np.nan`
- Impute and estimate all missing entries:
    - If dtype is `int`: impute with mean value of the feature column
    - If dtype is `str`: impute with most frequent item in the feature column
- Split data into categorical and continues and process them separately:
    - categorical features index = [1, 3, 5, 6, 7, 8, 9, 13]
    - continues features index = [0, 2, 4, 10, 11, 12]
- For categorical data:
    - 8 categorical attributes are transformed into a 99 dimension one-hot vector
- For continues data:
    - Normalize with maximum norm of that feature column
- Re-concatenate categorical features and continues features, the resulting data shape:
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/income_preprocessing.png)


## Usage

### Naive Bayes Classifier
- Train and test with the best **alpha** parameter for the **best** distribution assumption of the Naive Bayes classifier:
    - News dataset: `python3 runner.py --naive_bayes --data_news`
    - Mushroom dataset: `python3 runner.py --naive_bayes --data_mushroom`
    - Income dataset: `python3 runner.py --naive_bayes --data_income`

- Search for the best **alpha** parameter for **each** distribution assumption of the Naive Bayes classifier:
    - Add the `--search_opt` argument
    - News dataset (validated on the testing set): 
    ```python3 runner.py --naive_bayes --search_opt --data_news``` 
    - Mushroom dataset (validated on the testing set):
    ```python3 runner.py --naive_bayes --search_opt --data_mushroom```
    - Income dataset (Using N-fold cross-validation on the training set): 
    ```python3 runner.py --naive_bayes --search_opt --data_income``` 

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
    ```python3 runner.py --decision_tree --search_opt --data_news```
    - Mushroom dataset (validated on the testing set): 
    ```python3 runner.py --decision_tree --search_opt --data_mushroom``` 
    - Income dataset (Using N-fold cross-validation on the training set): 
    ```python3 runner.py --decision_tree --search_opt --data_income``` 

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
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_news.png)

### Mushroom Dataset - Testing Set Acc
- naive_bayes.GaussianNB() => 0.95505 (baseline)
- **naive_bayes.MultinomialNB(alpha=0.0001)** => **0.99569**
- naive_bayes.ComplementNB(alpha=0.0001) => 0.99507
- naive_bayes.BernoulliNB(alpha=0.0001) => 0.98830
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_mushroom.png)

### Income Dataset - N-Fold Cross-Validation Acc
- naive_bayes.GaussianNB() => 0.58602 (baseline)
- **naive_bayes.MultinomialNB(alpha=0.959)** =>**0.79148**
- naive_bayes.ComplementNB(alpha=0.16) => 0.74992
- naive_bayes.BernoulliNB(alpha=0.001) => 0.75760
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/nb_on_income.png)

## Result - Decision Tree Performance
### News Dataset - Testing Set Acc
- tree.DecisionTreeClassifier(criterion='gini', splitter='random', seed=1337, **max_depth=64**) => **0.64895**
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_news.png)
- decision tree visualization with the graphviz toolkit:
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_NEWS.png)

### Mushroom Dataset - Testing Set Acc
- tree.DecisionTreeClassifier(criterion='gini', splitter='random', seed=1337, **max_depth=64**) => **1.0**
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_mushroom.png)
- decision tree visualization with the graphviz toolkit:
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_MUSHROOM_.png)

### Income Dataset - N-Fold Cross-Validation Acc
- tree.DecisionTreeClassifier(criterion='gini', splitter='random', seed=1337, **max_depth=11**) => **0.83554**
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/dt_on_income.png)
- decision tree visualization with the graphviz toolkit:
![](https://github.com/andi611/Naive-Bayes-and-Decision-Tree-Classifiers/blob/master/image/TREE_INCOME.png)