PypeCast
==============================

PypeCast:
Python package for fast time series analysis and forecasting
with artificial neural networks. PypeCast consists of four main
modules: Data, Descriptor, Features and Models. The package
comes with simple predefined ANN forecasters, such as LSTM
based models, which are known to perform well in time series
forecasting. It also contains deep regression networks forecasters
with uncertainty estimation, which can be useful in situations
when making a wrong forecast is specially costly.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Recomended naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download and generate HFT data from bovespa's FTP server.
    │   │   └── make_dataset.py
    │   │   └── download_data.sh
    │   │
    │   ├── features       <- Scripts to turn raw data into features for ANN models.
    │   │   └── build_features.py
    │   │   └── features_supervised.py
    │   │
    │   ├── descriptor         <- Scripts to exploratory data analysis, with tools for time-series.
    │   │   │        
    │   │   └── series_description.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions.
    │   │   ├── Model.py
    │   │   └── mlp.py
    │   │   └── simple_lstm.py
    │   │   └── simple_rnn.py
    │   │
    │   ├── utils         <- Auxiliar functions or utilities.
    │   │   │        
    │   │   └── utils.py
    │   │
    │   └── metrics  <- Scripts to create exploratory and results oriented visualizations
    │       └── metrics.py
    │       └── uncertainty_losses.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
