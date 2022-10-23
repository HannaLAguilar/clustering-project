Clustering
==============================

Analyze the behavior of different clustering algorithms in 3 different data sets.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for analysis.
    │   └── raw            <- The original, immutable data dump.
    ││
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── clustering         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes clustering a Python module
    │   │
    │   ├── get_data       <- Scripts  to preprocessing data
    │   │   └── preprocessing.py
    │   │
    │   ├── cluster_algorithm    <- Scripts to cluster algorithms from scratch 
    │   │   │                 
    │   │   ├── bisecting_km.py
    │   │   ├── fuzzy_mean.py
    │   │   ├── kmeans.py
    │   │   └── kmedians.py
    │   │
    │   ├── analysis    <- Scripts to analyze cluster metrics and plot and save figures 
    │   │   │                 
    │   │   ├── bisecting_km.py
    │   │   ├── fuzzy_mean.py
    │   │   ├── kmeans.py
    │   │   └── kmedians.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
