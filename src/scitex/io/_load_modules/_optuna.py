#!/usr/bin/env python3
# Time-stamp: "2024-11-14 07:55:45 (ywatanabe)"
# File: ./scitex_repo/src/scitex/io/_load_modules/_optuna.py

from ._yaml import _load_yaml

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


def load_yaml_as_an_optuna_dict(fpath_yaml, trial):
    """
    Load a YAML file and convert it to an Optuna-compatible dictionary.

    This function reads a YAML file containing hyperparameter configurations
    and converts it to a dictionary suitable for use with Optuna trials.

    Parameters:
    -----------
    fpath_yaml : str
        The file path to the YAML configuration file.
    trial : optuna.trial.Trial
        The Optuna trial object to use for suggesting hyperparameters.

    Returns:
    --------
    dict
        A dictionary containing the hyperparameters with values suggested by Optuna.

    Raises:
    -------
    FileNotFoundError
        If the specified YAML file does not exist.
    ValueError
        If the YAML file contains invalid configuration for Optuna.
    ImportError
        If Optuna is not installed.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Please install with: pip install optuna"
        )

    _d = _load_yaml(fpath_yaml)

    for k, v in _d.items():
        dist = v["distribution"]

        if dist == "categorical":
            _d[k] = trial.suggest_categorical(k, v["values"])

        elif dist == "uniform":
            _d[k] = trial.suggest_int(k, float(v["min"]), float(v["max"]))

        elif dist == "loguniform":
            _d[k] = trial.suggest_loguniform(k, float(v["min"]), float(v["max"]))

        elif dist == "intloguniform":
            _d[k] = trial.suggest_int(k, float(v["min"]), float(v["max"]), log=True)

    return _d


def load_study_rdb(study_name, rdb_raw_bytes_url):
    """
    Load an Optuna study from a RDB (Relational Database) file.

    This function loads an Optuna study from a given RDB file URL.

    Parameters:
    -----------
    study_name : str
        The name of the Optuna study to load.
    rdb_raw_bytes_url : str
        The URL of the RDB file, typically in the format "sqlite:///*.db".

    Returns:
    --------
    optuna.study.Study
        The loaded Optuna study object.

    Raises:
    -------
    optuna.exceptions.StorageInvalidUsageError
        If there's an error loading the study from the storage.
    ImportError
        If Optuna is not installed.

    Example:
    --------
    >>> study = load_study_rdb(
    ...     study_name="YOUR_STUDY_NAME",
    ...     rdb_raw_bytes_url="sqlite:///path/to/your/study.db"
    ... )
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Please install with: pip install optuna"
        )

    storage = optuna.storages.RDBStorage(url=rdb_raw_bytes_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Loaded: {rdb_raw_bytes_url}")
    return study


# EOF
