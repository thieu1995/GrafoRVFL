============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/graforvfl />`_

.. code-block:: bash

	$ pip install graforvfl==2.1.0


* Install directly from source code.

.. code-block:: bash

   $ git clone https://github.com/thieu1995/GrafoRVFL.git
   $ cd GrafoRVFL
   $ python setup.py install

* In case, you want to install the development version from Github

.. code-block:: bash

   $ pip install git+https://github.com/thieu1995/GrafoRVFL


After installation, you can check the version of installed GrafoRVFL::

   $ python
   >>> import graforvfl
   >>> graforvfl.__version__

=========
Tutorials
=========

In this section, we will explore the usage of the GrafoRVFL model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage.


Provided classes
----------------

Classes that hold Models and Dataset

.. code-block:: python

	from graforvfl import DataTransformer, Data
	from graforvfl import RvflRegressor, RvflClassifier
	from graforvfl import GfoRvflCV


`DataTransformer` class
-----------------------

We provide many scaler classes that you can select and make a combination of transforming your data via
`DataTransformer` class. For example: scale data by `Loge` and then `Sqrt` and then `MinMax`.

.. code-block:: python

	from graforvfl import DataTransformer
	import pandas as pd
	from sklearn.model_selection import train_test_split

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values
	X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

	dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
	X_train_scaled = dt.fit_transform(X_train)
	X_test_scaled = dt.transform(X_test)


`Data` class
------------

+ You can load your dataset into Data class
+ You can split dataset to train and test set
+ You can scale dataset without using DataTransformer class
+ You can scale labels using LabelEncoder

.. code-block:: python

	from graforvfl import Data
	import pandas as pd

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values

	data = Data(X, y, name="position_salaries")

	#### Split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

	#### Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)




`Neural Network` class
----------------------

.. code-block:: python

	from graforvfl import RvflRegressor, RvflClassifier, GfoRvflCV, IntegerVar, StringVar

	## 1. Use standard RVFL model for regression problem
	model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", alpha=0.5)

	## 2. Use standard RVFL model for classification problem
	model = RvflClassifier(size_hidden=10, act_name='sigmoid', weight_initializer="random_normal", alpha=0)


	## 3. Use Gradient Free Optimization to fine-tune the hyper-parameter of RVFL network for regression problem
	# Design the boundary (parameters)
	my_bounds = [
	    IntegerVar(lb=2, ub=1000, name="size_hidden"),
	    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu",
	                          "elu", "selu", "rrelu", "tanh", "sigmoid"), name="act_name"),
	    StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform",
	                           "glorot_normal", "lecun_uniform", "lecun_normal", "random_uniform",
	                           "random_normal"), name="weight_initializer")
	]
	opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
	model = GfoRvflCV(problem_type="regression", bounds=my_bounds,
	                optim="OriginalWOA", optim_params=opt_paras,
	                scoring="MSE", cv=3, seed=42, verbose=True)


Supported functions in `model` object
-------------------------------------

.. code-block:: python

	from graforvfl import RvflRegressor, Data

	data = Data()       # Assumption that you have provided this object like above

	model = RvflRegressor(size_hidden=10, act_name='sigmoid', weight_initializer="random_uniform", alpha=0.5)

	## Train the model
	model.fit(data.X_train, data.y_train)

	## Predicting a new result
	y_pred = model.predict(data.X_test)

	## Calculate metrics using score or scores functions.
	print(model.score(data.X_test, data.y_test, method="MAE"))
	print(model.scores(data.X_test, data.y_test, list_metrics=["MAPE", "NNSE", "KGE", "MASE", "R2", "R", "R2S"]))

	## Calculate metrics using evaluate function
	print(model.evaluate(data.y_test, y_pred, list_metrics=("MSE", "RMSE", "MAPE", "NSE")))

	## Save performance metrics to csv file
	model.save_metrics(data.y_test, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv")

	## Save training loss to csv file
	model.save_loss_train(save_path="history", filename="loss.csv")

	## Save predicted label
	model.save_y_predicted(X=data.X_test, y_true=data.y_test, save_path="history", filename="y_predicted.csv")

	## Save model
	model.save_model(save_path="history", filename="traditional_mlp.pkl")

	## Load model
	trained_model = RvflRegressor.load_model(load_path="history", filename="traditional_mlp.pkl")




A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
