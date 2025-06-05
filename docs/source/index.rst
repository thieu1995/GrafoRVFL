.. GrafoRVFL documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GrafoRVFL's documentation!
=====================================

.. image:: https://img.shields.io/badge/release-2.0.0-yellow.svg
   :target: https://github.com/thieu1995/graforvfl/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/graforvfl

.. image:: https://badge.fury.io/py/graforvfl.svg
   :target: https://badge.fury.io/py/graforvfl

.. image:: https://img.shields.io/pypi/pyversions/graforvfl.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/graforvfl.svg
   :target: https://img.shields.io/pypi/status/graforvfl.svg

.. image:: https://img.shields.io/pypi/dm/graforvfl.svg
   :target: https://img.shields.io/pypi/dm/graforvfl.svg

.. image:: https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yml/badge.svg
   :target: https://github.com/thieu1995/graforvfl/actions/workflows/publish-package.yml

.. image:: https://pepy.tech/badge/graforvfl
   :target: https://pepy.tech/project/graforvfl

.. image:: https://img.shields.io/github/release-date/thieu1995/graforvfl.svg
   :target: https://img.shields.io/github/release-date/thieu1995/graforvfl.svg

.. image:: https://readthedocs.org/projects/graforvfl/badge/?version=latest
   :target: https://graforvfl.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/graforvfl.svg
   :target: https://img.shields.io/github/contributors/thieu1995/graforvfl.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10258280.svg
  :target: https://doi.org/10.5281/zenodo.10258280

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


GrafoRVFL is an open-source library in Python that employs gradient-free optimization (GA, PSO, WOA, TLO, DE, ...) to
optimize Random Vector Functional Link Networks. It is entirely implemented based on Numpy and fully compatible
with the interfaces of the Scikit-Learn library. With GrafoRVFL, you can fine-tune the hyper-parameters of network
or optimize weights in the network using gradient-free optimizers.


* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: `RvflRegressor`, `RvflClassifier`, `GfoRvflCV`, `GfoRvflTuner`, `GfoRvflComparator`
* **Total Gradient Free based RVFL Regressor**: > 200 Models
* **Total Gradient Free based RVFL Classifier**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://graforvfl.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, matplotlib


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/graforvfl.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
