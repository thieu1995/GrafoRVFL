# Version 2.2.0

### 🔧 Core Updates
- Enhanced `Data`, `DataTransformer`, and `BaseRVFL` to support both `pandas.DataFrame` and `pandas.Series`.

### 🧠 Incremental Learning
- Added `partial_fit()` method for: `RvflRegressor` and `RvflClassifier`
  - ✅ Supports **batch training**
  - ✅ Supports **online (real-time) learning**
  - ✅ Supports **hybrid training** (batch + online)
- 👉 [See streaming examples](/examples/real_time_streaming)

### 📈 Model Explainability
- Updated `BaseRVFL` to support **SHAP-based model explainability**. 
- 👉 [See SHAP examples](/examples/shap)

### 🧪 Maintenance
- Updated examples, tests, and documentation.

------------------------------------------------------------------------------

# Version 2.1.0

+ Update `setup.py` for more robust and maintainable package management.
+ Update `mealpy` dependency to **v3.0.2** for better performance and compatibility.
+ Update GitHub Actions workflows for testing and publishing.
+ Update `scorer` module.
+ Update `OneHotEncoder` and `LabelEncoder` classes.
+ Update `TimeSeriesDifferencer`, `FeatureEngineering`, and `DataTransformer` classes.
+ Add more parameters to `GfoRvflCV`, `GfoRvflTuner`, and `GfoRvflComparator` classes include `mode`, `n_workers`, and `termination`.
+ Update examples, tests, and documentation.

------------------------------------------------------------------------------

# Version 2.0.0

+ Fix bugs check digit, floating, number, sequence in `boundary_controller` module.
+ Remove `trainer` parameter from RVFL-based models
+ Rename `GfoRvflTuner` class to `GfoRvflCV` class
+ Add `GfoRvflTuner` class that can be used to tune optimizer-parameter of GFO-RVFL network
+ Add `GfoRvflComparator` class that can be used to compare the performance of multiple GFO-RVFL networks
+ Update examples, tests, and documentation.

------------------------------------------------------------------------------

# Version 1.2.0

+ Fix bugs in `seed` value, `OneHotEncoder` class, and `hard_shrink` function.
+ Update `trainer` parameter for model
+ Rename `list_methods` parameter to `list_metrics` parameter
+ Update examples, tests, and documentation.

------------------------------------------------------------------------------

# Version 1.1.0

+ Rename `MhaTuneRvfl` class to `GfoRvflTuner` class
+ Add seed parameter to all classes.
+ Add examples with GridSearchCV
+ Update PerMetrics and Mealpy dependencies
+ Update documents

------------------------------------------------------------------------------

# Version 1.0.0 (First version)

+ This library uses Numpy only - Gradient Free (No Pytorch, No Tensorflow)
+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, requirements.txt)
+ Add shared modules (activator, boundary_controller, data_processor, randomer, scaler, and scorer)
+ Add `RvflRegressor` and `RvflClassifier` classes
+ Add `GfoRvflTuner` class that can be used to tune RVFL network
+ Add publish workflow
+ Add examples and tests folders

------------------------------------------------------------------------------

# Version 0.0.1

+ Test version to get Zonodo DOI
