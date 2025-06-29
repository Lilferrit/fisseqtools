import functools
import itertools
import json
import pathlib
import pickle
from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple

import fire
import numpy as np
import pandas as pd
import shap
import sklearn
import sklearn.metrics
import sklearn.utils
import tqdm
import xgboost as xgb

TrainFun = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray | None]],
    sklearn.base.BaseEstimator,
]


np.random.seed(42)


def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    """
    Trains an XGBoost classifier on the provided training data.

    Args:
        x_train (np.ndarray):
            Feature matrix for training data.
        y_train (np.ndarray):
            Labels for training data.
        x_eval (np.ndarray):
            Feature matrix for evaluation data.
        y_eval (np.ndarray):
            Labels for evaluation data.
        sample_weight (Optional[np.ndarray | None], optional):
            Sample weights for training, default is None.

    Returns:
        sklearn.base.BaseEstimator:
            The trained XGBoost classifier.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        colsample_bynode=0.7,
        subsample=0.5,
        early_stopping_rounds=5,
        n_estimators=100,
        eval_metric="auc",
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_eval, y_eval)],
        sample_weight=sample_weight,
        verbose=True,
    )


def train_single_feature_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sample_weight: Optional[np.ndarray | None] = None,
) -> sklearn.base.BaseEstimator:
    """
    Trains an XGBoost classifier that splits examples on only one feature.

    Args:
        x_train (np.ndarray):
            Feature matrix for training data.
        y_train (np.ndarray):
            Labels for training data.
        x_eval (np.ndarray):
            Feature matrix for evaluation data.
        y_eval (np.ndarray):
            Labels for evaluation data.
        sample_weight (Optional[np.ndarray | None], optional):
            Sample weights for training, default is None.

    Returns:
        sklearn.base.BaseEstimator:
            The trained XGBoost classifier.
    """
    return xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=1,
        n_estimators=1,
        eval_metric="auc",
        early_stopping_rounds=1,
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_eval, y_eval)],
        sample_weight=sample_weight,
        verbose=True,
    )


def get_mask_features(
    target_column: str,
    feature_columns: List[str],
    wt_key: str,
    curr_split: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a mask for the wild-type entries and extracts the feature matrix.

    Args:
        target_column (str):
            The name of the target column.
        feature_columns (List[str]):
            List of feature column names.
        wt_key (str):
            The wild-type key to identify the wild-type entries.
        curr_split (pd.DataFrame):
            The dataset split to be processed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the wild-type mask (wt_mask) and the feature
            matrix.
    """
    wt_mask = (curr_split[target_column] == wt_key).to_numpy(dtype=bool)
    feature_matrix = curr_split[feature_columns].to_numpy(dtype=np.float64)
    return wt_mask, feature_matrix


def get_train_data_labels(
    wt_mask: np.ndarray,
    variant_mask: np.ndarray,
    feature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the feature data and labels for wild-type and variant samples from
    the feature matrix.

    Args:
        wt_mask (np.ndarray):
            Mask for the wild-type entries.
        variant_mask (np.ndarray):
            Mask for the variant entries.
        feature_matrix (np.ndarray):
            The matrix containing feature values.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            A tuple containing the combined feature matrix and the
            corresponding labels.
    """
    wt_features = feature_matrix[wt_mask]
    variant_features = feature_matrix[variant_mask]
    combined_features = np.vstack((wt_features, variant_features))
    labels = np.zeros(len(combined_features), dtype=bool)
    labels[len(wt_features) :] = True
    return combined_features, labels


def get_metrics(
    model: sklearn.base.ClassifierMixin,
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: Optional[str] = "",
    verbose: Optional[str] = True,
) -> Tuple[float, float]:
    """
    Computes and returns the ROC AUC and accuracy.

    Args:
        model (sklearn.base.ClassifierMixin):
            The trained classifier.
        features (np.ndarray):
            Feature matrix for the dataset.
        labels (np.ndarray):
            True labels for the dataset.
        dataset_name (Optional[str], optional):
            The name of the dataset, default is an empty string.
        verbose (Optional[str], optional):
            Whether to print the metrics, default is True.

    Returns:
        Tuple[float, float]:
            The ROC AUC and accuracy of the classifier on the dataset.
    """
    y_prob = model.predict_proba(features)[:, 1].flatten()
    y_pred = y_prob >= 0.5
    roc_auc = sklearn.metrics.roc_auc_score(labels, y_prob)
    accuracy = sklearn.metrics.accuracy_score(labels, y_pred)

    if verbose:
        if dataset_name != "":
            dataset_name += " "

        print(f"{dataset_name}ROC AUC: {roc_auc:.2f}", flush=True)
        print(f"{dataset_name}Accuracy: {accuracy:.2f}", flush=True)

    return roc_auc, accuracy


def train_ovwt(
    train_fun: TrainFun,
    train_split: pd.DataFrame,
    eval_one_split: pd.DataFrame,
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
    test_split: Optional[pd.DataFrame] = None,
    permutate_labels: bool = False,
) -> Tuple[Dict[str, sklearn.base.BaseEstimator], pd.DataFrame]:
    """
    Trains models for different variants of a target column.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        train_split (pd.DataFrame):
            DataFrame containing the training data.
        eval_one_split (pd.DataFrame):
            DataFrame containing the first evaluation split.
        test_split (pd.DataFrame):
            DataFrame containing the second evaluation split.
        meta_data (Dict[str, str | List[str]]):
            Metadata containing column names for target and features.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".

    Returns:
        Tuple[Dict[str, sklearn.base.BaseEstimator], pd.DataFrame]:
            A tuple containing the trained models and the metrics DataFrame.
    """
    target_column: str = meta_data["target_column"]
    feature_columns: List[str] = meta_data["feature_columns"]

    if permutate_labels:
        train_split[target_column] = (
            train_split[target_column].sample(frac=1).reset_index(drop=True)
        )
        eval_one_split[target_column] = (
            eval_one_split[target_column].sample(frac=1).reset_index(drop=True)
        )

        if test_split is not None:
            test_split[target_column] = (
                test_split[target_column].sample(frac=1).reset_index(drop=True)
            )

    get_features = functools.partial(
        get_mask_features, target_column, feature_columns, wt_key
    )
    train_wt_mask, train_features = get_features(train_split)
    eval_one_wt_mask, eval_one_features = get_features(eval_one_split)

    datasets = ["eval", "train"]
    if test_split is not None:
        datasets.append("test")

    stats = ["roc_auc", "accuracy"]
    accuracy_roc = {
        f"{dataset}_{stat}": list()
        for dataset, stat in itertools.product(datasets, stats)
    }
    accuracy_roc[target_column] = list()
    models = dict()

    for curr_variant in train_split[target_column].unique():
        if curr_variant == wt_key:
            print("WT Key, Skipping\n", flush=True)
            continue

        print(f"Training classifier {curr_variant}", flush=True)

        # Get data and labels
        get_var_mask = lambda split: (split[target_column] == curr_variant).to_numpy()
        curr_train_features, curr_train_labels = get_train_data_labels(
            train_wt_mask, get_var_mask(train_split), train_features
        )
        curr_eval_one_features, curr_eval_one_labels = get_train_data_labels(
            eval_one_wt_mask, get_var_mask(eval_one_split), eval_one_features
        )
        weights = sklearn.utils.compute_sample_weight("balanced", curr_train_labels)

        model = train_fun(
            curr_train_features,
            curr_train_labels,
            curr_eval_one_features,
            curr_eval_one_labels,
            weights,
        )

        curr_datasets = [
            ("Train", curr_train_features, curr_train_labels),
            ("Eval", curr_eval_one_features, curr_eval_one_labels),
        ]

        if test_split is not None:
            test_wt_mask, test_features = get_features(test_split)
            curr_test_features, curr_test_labels = get_train_data_labels(
                test_wt_mask, get_var_mask(test_split), test_features
            )
            curr_datasets.append(("Test", curr_test_features, curr_test_labels))

        for name, features, labels in curr_datasets:
            roc_auc, accuracy = get_metrics(model, features, labels, dataset_name=name)
            name = name.lower().replace(" ", "_")
            accuracy_roc[name + "_roc_auc"].append(roc_auc)
            accuracy_roc[name + "_accuracy"].append(accuracy)

        accuracy_roc[target_column].append(curr_variant)
        models[curr_variant] = model
        print("", flush=True)

    return models, pd.DataFrame(accuracy_roc)


def test_ovwt(
    model_dict: Dict[str, sklearn.base.BaseEstimator],
    train_split: pd.DataFrame,
    eval_one_split: pd.DataFrame,
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
    test_split: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Evaluates models for different variants of a target column.

    Args:
        model_dict (Dict[str, sklearn.base.BaseEstimator]):
            Dictionary mapping variant names to pre-trained models.
        train_split (pd.DataFrame):
            DataFrame containing the training data.
        eval_one_split (pd.DataFrame):
            DataFrame containing the first evaluation split.
        meta_data (Dict[str, str | List[str]]):
            Metadata containing column names for target and features.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
        test_split (Optional[pd.DataFrame], optional):
            DataFrame containing the second evaluation split.

    Returns:
        pd.DataFrame:
            DataFrame containing ROC AUC and accuracy metrics.
    """
    target_column: str = meta_data["target_column"]
    feature_columns: List[str] = meta_data["feature_columns"]

    get_features = functools.partial(
        get_mask_features, target_column, feature_columns, wt_key
    )
    train_wt_mask, train_features = get_features(train_split)
    eval_one_wt_mask, eval_one_features = get_features(eval_one_split)

    datasets = ["eval", "train"]
    if test_split is not None:
        datasets.append("test")

    stats = ["roc_auc", "accuracy"]
    accuracy_roc = {
        f"{dataset}_{stat}": list()
        for dataset, stat in itertools.product(datasets, stats)
    }
    accuracy_roc[target_column] = list()

    for curr_variant, model in model_dict.items():
        if curr_variant == wt_key:
            print("WT Key, Skipping\n", flush=True)
            continue

        print(f"Evaluating classifier {curr_variant}", flush=True)

        get_var_mask = lambda split: (split[target_column] == curr_variant).to_numpy()

        curr_train_features, curr_train_labels = get_train_data_labels(
            train_wt_mask, get_var_mask(train_split), train_features
        )
        curr_eval_one_features, curr_eval_one_labels = get_train_data_labels(
            eval_one_wt_mask, get_var_mask(eval_one_split), eval_one_features
        )

        curr_datasets = [
            ("Train", curr_train_features, curr_train_labels),
            ("Eval", curr_eval_one_features, curr_eval_one_labels),
        ]

        if test_split is not None:
            test_wt_mask, test_features = get_features(test_split)
            curr_test_features, curr_test_labels = get_train_data_labels(
                test_wt_mask, get_var_mask(test_split), test_features
            )
            curr_datasets.append(("Test", curr_test_features, curr_test_labels))

        for name, features, labels in curr_datasets:
            roc_auc, accuracy = get_metrics(model, features, labels, dataset_name=name)
            name = name.lower().replace(" ", "_")
            accuracy_roc[name + "_roc_auc"].append(roc_auc)
            accuracy_roc[name + "_accuracy"].append(accuracy)

        accuracy_roc[target_column].append(curr_variant)
        print("", flush=True)

    return pd.DataFrame(accuracy_roc)


def get_shap_values(
    dataset: pd.DataFrame,
    models: Dict[str, sklearn.base.BaseEstimator],
    meta_data: Dict[str, str | List[str]],
    wt_key: Optional[str] = "WT",
    dset_name: Optional[str | None] = None,
) -> pd.DataFrame:
    """
    Computes SHAP values for the given dataset over all variants.

    Args:
        dataset (pd.DataFrame):
            The dataset for which to compute SHAP values.
        models (Dict[str, sklearn.base.BaseEstimator]):
            The models to use for SHAP value computation.
        meta_data (Dict[str, str | List[str]]):
            Metadata containing column names for target and features.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
        dset_name (Optional[str | None], optional):
            The name of the dataset, used for progress display, default is None.

    Returns:
        pd.DataFrame:
            A DataFrame containing SHAP values for each sample and feature.
    """
    target_column: str = meta_data["target_column"]
    feature_columns: List[str] = meta_data["feature_columns"]
    dataset = dataset[dataset[target_column] != wt_key]
    all_features = dataset[feature_columns].to_numpy()

    # Initialize shap value dataframe
    filtered_dataset = dataset[dataset[target_column] != wt_key]
    shap_columns = [target_column] + ["p_is_var"] + feature_columns
    shap_df = pd.DataFrame(
        [[None] * len(shap_columns)] * len(dataset), columns=shap_columns
    )
    shap_df.loc[:, target_column] = filtered_dataset[target_column].to_numpy()

    pbar_desc = (
        None if dset_name is None else f"Computing Shap Values Over: {dset_name}"
    )
    for curr_variant in tqdm.tqdm(shap_df[target_column].unique(), desc=pbar_desc):
        curr_model = models[curr_variant]
        variant_mask = (shap_df[target_column] == curr_variant).to_numpy()
        curr_features = all_features[variant_mask]
        curr_explainer = shap.TreeExplainer(curr_model)
        curr_shap_vals = curr_explainer.shap_values(curr_features)

        if len(curr_shap_vals.shape) == 3:
            curr_shap_vals = curr_shap_vals[:, :, 1]

        shap_df.loc[variant_mask, feature_columns] = curr_shap_vals
        shap_df.loc[variant_mask, target_column] = curr_variant
        curr_predict_probas = curr_model.predict_proba(curr_features)[:, 1]
        shap_df.loc[variant_mask, "p_is_var"] = curr_predict_probas

    return shap_df


def ovwt(
    train_fun: TrainFun,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    test_data_path: Optional[PathLike] = None,
    wt_key: Optional[str] = "WT",
    permutate_labels: bool = False,
) -> Tuple[
    pd.DataFrame,
    Dict[str, sklearn.base.BaseEstimator],
    Dict[str, str | List[str]],
    List[str],
]:
    """
    Orchestrates the training, evaluation, and SHAP value computation.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        train_data_path (PathLike):
            Path to the training data file.
        eval_one_data_path (PathLike):
            Path to the first evaluation data file.
        test_data_path (PathLike):
            Path to the second evaluation data file.
        meta_data_json_path (PathLike):
            Path to the metadata JSON file.
        output_dir (PathLike):
            Directory where results and models are saved.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
    """
    train_split = pd.read_parquet(train_data_path)
    eval_one_split = pd.read_parquet(eval_one_data_path)
    test_split = None
    if test_data_path is not None:
        test_split = pd.read_parquet(test_data_path)

    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    output_dir = pathlib.Path(output_dir)
    models, accuracy_roc = train_ovwt(
        train_fun=train_fun,
        train_split=train_split,
        eval_one_split=eval_one_split,
        meta_data=meta_data,
        test_split=test_split,
        wt_key=wt_key,
        permutate_labels=permutate_labels,
    )

    target_column = meta_data["target_column"]
    example_counts = train_split[target_column].value_counts()
    accuracy_roc["Example Count"] = accuracy_roc[target_column].map(example_counts)
    accuracy_roc = accuracy_roc.sort_values(by=target_column, ascending=False)
    accuracy_roc.to_csv(output_dir / "train_results.csv")

    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    shap_targets = [
        ("train", train_split),
        ("eval", eval_one_split),
    ]

    if test_split is not None:
        shap_targets.append(("test", test_split))

    for name, split in shap_targets:
        curr_shap_df = get_shap_values(split, models, meta_data, wt_key, name)
        curr_shap_df.to_parquet(output_dir / f"{name}_shap.parquet")

    feature_columns = eval_one_split[meta_data["feature_columns"]].columns
    return models, meta_data, list(feature_columns)


def ovwt_single_feature(
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    test_data_path: Optional[PathLike] = None,
    wt_key: Optional[str] = "WT",
) -> None:
    output_dir = pathlib.Path(output_dir)
    models, meta_data, columns = ovwt(
        train_single_feature_xgboost,
        train_data_path,
        eval_one_data_path,
        meta_data_json_path,
        output_dir,
        test_data_path,
        wt_key,
    )

    features = {
        meta_data["target_column"]: list(),
        "feature_name": list(),
    }

    for variant, stump in models.items():
        tree_dump = stump.get_booster().get_dump()[0]
        split_line = tree_dump.split("\n")[0]
        feature_part = split_line.split("[")[1].split("]")[0]
        feature_index = int(feature_part.split("f")[1].split("<")[0])
        feature_name = columns[feature_index]
        features[meta_data["target_column"]].append(variant)
        features["feature_name"].append(feature_name)

    pd.DataFrame(features).to_csv(output_dir / "selected_features.csv")


def ovwt_shap_only(
    model_pkl_path: PathLike,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    test_data_path: Optional[PathLike] = None,
    wt_key: Optional[str] = "WT",
) -> None:
    """
    Compute shap values over training data

    Args:
        model_pkl_path (PathLike):
            Path to the pickled model dictionary
        train_data_path (PathLike):
            Path to the training data file.
        eval_one_data_path (PathLike):
            Path to the first evaluation data file.
        test_data_path (PathLike):
            Path to the second evaluation data file.
        meta_data_json_path (PathLike):
            Path to the metadata JSON file.
        output_dir (PathLike):
            Directory where results and models are saved.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
    """
    train_split = pd.read_parquet(train_data_path)
    eval_one_split = pd.read_parquet(eval_one_data_path)
    test_split = None
    if test_data_path is not None:
        test_split = pd.read_parquet(test_data_path)

    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    output_dir = pathlib.Path(output_dir)

    with open(model_pkl_path, "rb") as f:
        models = pickle.load(f)

    shap_targets = [
        ("train", train_split),
        ("eval", eval_one_split),
    ]

    if test_split is not None:
        shap_targets.append(("test", test_split))

    for name, split in shap_targets:
        curr_shap_df = get_shap_values(split, models, meta_data, wt_key, name)
        curr_shap_df.to_parquet(output_dir / f"{name}_shap.parquet")


def ovwt_stratified(
    train_fun: TrainFun,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    stratify_column: str,
    wt_key: Optional[str] = "WT",
    permutate_labels: bool = False,
    test_data_path: Optional[PathLike] = None,
) -> None:
    """
    Trains models separately for each unique value in the stratification column
    and evaluates them independently.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        train_data_path (PathLike):
            Path to the training data file.
        eval_one_data_path (PathLike):
            Path to the first evaluation data file.
        meta_data_json_path (PathLike):
            Path to the metadata JSON file.
        output_dir (PathLike):
            Directory where results and models are saved.
        stratify_column (str):
            The column used for stratification; models will be trained
            separately for each unique value in this column.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
        permutate_labels (bool, optional):
            Whether to permute labels for control experiments. Default is False.
    """
    train_split = pd.read_parquet(train_data_path)
    eval_split = pd.read_parquet(eval_one_data_path)
    test_split = None if test_data_path is None else pd.read_parquet(test_data_path)

    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    output_dir = pathlib.Path(output_dir)
    all_models = dict()
    result_tables = list()
    for curr_value in eval_split[stratify_column].unique():
        print("-" * 10 + f" Training over {stratify_column} = {curr_value} " + "-" * 10)
        curr_train_split = train_split[train_split[stratify_column] == curr_value]
        curr_eval_split = eval_split[eval_split[stratify_column] == curr_value]

        if test_split is not None:
            curr_test_split = test_split[test_split[stratify_column] == curr_value]
        else:
            curr_test_split = None

        models, accuracy_roc = train_ovwt(
            train_fun=train_fun,
            train_split=curr_train_split,
            eval_one_split=curr_eval_split,
            meta_data=meta_data,
            wt_key=wt_key,
            permutate_labels=permutate_labels,
            test_split=curr_test_split,
        )

        accuracy_roc[stratify_column] = curr_value
        result_tables.append(accuracy_roc)
        all_models[curr_value] = models

    pd.concat(result_tables, ignore_index=True).to_csv(output_dir / "train_results.csv")
    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(all_models, f)


def ovwt_stratified_test(
    models_path: PathLike,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    stratify_column: str,
    wt_key: Optional[str] = "WT",
    permutate_labels: bool = False,
    test_data_path: Optional[PathLike] = None,
) -> None:
    """
    Trains models separately for each unique value in the stratification column
    and evaluates them independently.

    Args:
        train_fun (TrainFun):
            Function used to train the model.
        models_path (PathLike):
            Path to pickled models dictionary.
        train_data_path (PathLike):
            Path to the training data file.
        eval_one_data_path (PathLike):
            Path to the first evaluation data file.
        meta_data_json_path (PathLike):
            Path to the metadata JSON file.
        output_dir (PathLike):
            Directory where results and models are saved.
        stratify_column (str):
            The column used for stratification; models will be trained
            separately for each unique value in this column.
        wt_key (Optional[str], optional):
            The key used for wild-type samples, default is "WT".
        permutate_labels (bool, optional):
            Whether to permute labels for control experiments. Default is False.
    """
    train_split = pd.read_parquet(train_data_path)
    eval_split = pd.read_parquet(eval_one_data_path)
    test_split = None if test_data_path is None else pd.read_parquet(test_data_path)

    with open(models_path, "rb") as f:
        models = pickle.load(f)

    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    output_dir = pathlib.Path(output_dir)
    result_tables = list()
    for curr_value in eval_split[stratify_column].unique():
        print("-" * 10 + f" Training over {stratify_column} = {curr_value} " + "-" * 10)
        curr_train_split = train_split[train_split[stratify_column] == curr_value]
        curr_eval_split = eval_split[eval_split[stratify_column] == curr_value]
        curr_models = models[curr_value]

        if test_split is not None:
            curr_test_split = test_split[test_split[stratify_column] == curr_value]
        else:
            curr_test_split = None

        accuracy_roc = test_ovwt(
            model_dict=curr_models,
            train_split=curr_train_split,
            eval_one_split=curr_eval_split,
            meta_data=meta_data,
            wt_key=wt_key,
            test_split=curr_test_split,
        )

        accuracy_roc[stratify_column] = curr_value
        result_tables.append(accuracy_roc)

    pd.concat(result_tables, ignore_index=True).to_csv(output_dir / "train_results.csv")


def sample_and_change_rows(
    data_df: pd.DataFrame,
    sample_proportion: float,
    target_column: str,
    modifier_string: str,
) -> pd.DataFrame:
    data_df = data_df.copy()
    sampled_idx = data_df.sample(n=int(len(data_df) * sample_proportion)).index
    mask = data_df.index.isin(sampled_idx)
    data_df.loc[mask, target_column] += modifier_string

    return data_df


def wtvwt_control(
    train_fun: TrainFun,
    train_data_path: PathLike,
    eval_one_data_path: PathLike,
    meta_data_json_path: PathLike,
    output_dir: PathLike,
    test_data_path: Optional[PathLike] = None,
    wt_key: Optional[str] = "WT",
    num_iters: int = 100,
    sample_proportion: float = 0.5,
) -> None:
    with open(meta_data_json_path) as f:
        meta_data = json.load(f)

    target_column = meta_data["target_column"]
    train_split = pd.read_parquet(train_data_path)
    train_split = train_split[train_split[target_column] == wt_key]
    eval_one_split = pd.read_parquet(eval_one_data_path)
    eval_one_split = eval_one_split[eval_one_split[target_column] == wt_key]

    test_split = None
    if test_data_path is not None:
        test_split = pd.read_parquet(test_data_path)
        test_split = test_split[test_split[target_column] == wt_key]

    output_dir = pathlib.Path(output_dir)
    models = dict()
    accuracy_roc = list()
    train_shap = list()
    eval_shap = list()
    test_shap = list()

    for i in range(num_iters):
        print("=" * 10 + f" WT vs. WT Iteration: {i + 1} " + "=" * 10)
        curr_train_split = sample_and_change_rows(
            train_split, sample_proportion, target_column, f"_{i}"
        )
        curr_eval_split = sample_and_change_rows(
            eval_one_split, sample_proportion, target_column, f"_{i}"
        )
        curr_test_split = None

        if test_split is not None:
            curr_test_split = sample_and_change_rows(
                test_split, sample_proportion, target_column, f"_{i}"
            )

        next_models, next_accuracy_roc = train_ovwt(
            train_fun=train_fun,
            train_split=curr_train_split,
            eval_one_split=curr_eval_split,
            meta_data=meta_data,
            test_split=curr_test_split,
            wt_key=wt_key,
        )

        train_shap.append(
            get_shap_values(
                curr_train_split,
                next_models,
                meta_data,
                wt_key=wt_key,
                dset_name="train",
            )
        )
        eval_shap.append(
            get_shap_values(
                curr_eval_split, next_models, meta_data, wt_key=wt_key, dset_name="eval"
            )
        )

        if test_split is not None:
            test_shap.append(
                get_shap_values(
                    curr_test_split,
                    next_models,
                    meta_data,
                    wt_key=wt_key,
                    dset_name="test",
                )
            )

        accuracy_roc.append(next_accuracy_roc)
        models.update(next_models)

    accuracy_roc = pd.concat(accuracy_roc, axis=0)
    accuracy_roc = accuracy_roc.sort_values(by=target_column, ascending=False)
    accuracy_roc.to_csv(output_dir / "train_results.csv")

    with open(output_dir / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    pd.concat(train_shap, axis=0).to_parquet(output_dir / "train_shap.parquet")
    pd.concat(eval_shap, axis=0).to_parquet(output_dir / "eval_shap.parquet")

    if test_split is not None:
        pd.concat(test_shap, axis=0).to_parquet(output_dir / "test_shap.parquet")

    feature_columns = eval_one_split[meta_data["feature_columns"]].columns
    return models, meta_data, list(feature_columns)


def main():
    """Setup CLI"""
    fire.Fire(
        {
            "xgb": functools.partial(ovwt, train_xgboost),
            "xgb-stratified": functools.partial(ovwt_stratified, train_xgboost),
            "xgb-stratified-test": ovwt_stratified_test,
            "shap_only": ovwt_shap_only,
            "single_feature": ovwt_single_feature,
            "xgb-wtvwt-control": functools.partial(wtvwt_control, train_xgboost),
        }
    )


if __name__ == "__main__":
    main()
