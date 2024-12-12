# finn-valuation-tool

## Instructions

## Valuation Tool Case

This is a case study for a data science position at FINN.no. In this study we'll do some simple machine learning, and explore some areas related to having machine learning in production systems.


## Hands on

We are, among other tasks, trying to build components for price suggestions on marketplaces, e.g. for cars. The first step here is to use historical car ads to predict prices of new car ads. Attached you will find a file called `train.csv` containing car ads comprising the training set, and a file called `test.csv` containing ads which will be used as a test set. In these two datasets, the column `asking_price` is your target variable. When programming is needed, use your preferred language to answer. A notebook of some sort (e.g. Jupyter) is a good way to communicate your work, but you are free to choose.

## Assignments

1. Train a model of your choice on the training data, but don't spend too much time on it. It is more important that you understand the task than that you try to reach a great score. Predict the price for all ads in the training set, and for all ads in the test set. Calculate the mean absolute percentage error (MAPE*) for both sets  
   a. Reflect on the MAPE, what does it tell you? Is this score good?  
   b. What does the difference in MAPE on the training set and test set tell you? Is this good?  

2. If you were to train an even more sophisticated model, with orders of magnitude more training data at your disposal, which model would you pick, and why? How do you think the new model would compare to the model you created for part 1?

3. You decide to put the more sophisticated model into production. Outline how you would go about doing that.  
   a. What do you see as some challenges of running this model in production?  
   b. How could you solve them?


## Solution

### For all notebooks but CatBoost

Creating a virtual environment with venv.

```bash
python -m venv .venv
```

Activating the virtual environment (on Windows).

```bash
.venv\Scripts\Activate.ps1
```

Upgrading pip.

```bash
python -m pip install --upgrade pip
```

My python version 

```bash
python --version
``` 

```plaintext
Python 3.10.10
```

Installing the required packages.

```bash
pip install -r requirements.txt
```

Running the notebook.


### Catboost notebook

Because of dependency conflicts I decided to create a new virtual environment for this notebook. 

```bash
python -m venv .catboost
```

Activating the virtual environment (on Windows).

```bash	
.catboost\Scripts\Activate.ps1
```

Etc.