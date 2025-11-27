# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
import lightgbm as lgb
import dalex as dx

import sys, os

# path to ps3 folder
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from ps3.data import create_sample_split, load_transform
from ps3.evaluation import compute_metrics

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?
# 
# Exposure (how long the insurance need to cover the client) is indeed a deterministic factor of insurance claim.
# This is because longer the duration of insurance coverage, higher the liklihood that a severe accident happens and the client request a claim.
# However, the exposure does not really impact the risk per unit time (how likely the motor driver is a dangerous driver etc.).
# The latter is what we want to model as it reflects the unit price of insurance.
# Therefore, is is better to take claim/exposure as target variable, rather than taking claim as target variable and exposure as feature.

# TODO: use your create_sample_split function here
df = create_sample_split(df, "IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)


pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

#add_constant = FunctionTransformer(lambda X: np.ones((X.shape[0],1)))

preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
        ("num", numeric_transformer, numeric_cols)
        #,("addcons", add_constant, [numeric_cols[0]])
    ]
)

preprocessor.set_output(transform="pandas")
model_pipeline_GLM = Pipeline([
    # TODO: Define pipeline steps here
    ("preprocessor", preprocessor),
    ("GLM_model", GeneralizedLinearRegressor(family=TweedieDistribution(1.5), l1_ratio=1, fit_intercept=True))
])

# let's have a look at the pipeline
model_pipeline_GLM

# let's check that the transforms worked
model_pipeline_GLM[:-1].fit_transform(df_train)

model_pipeline_GLM.fit(df_train, y_train_t, GLM_model__sample_weight=w_train_t)

pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline_GLM[-1].intercept_], model_pipeline_GLM[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline_GLM[-1].feature_names_,
).T

df_test["pp_t_glm2"] = model_pipeline_GLM.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline_GLM.predict(df_train)

print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.
model_pipeline = Pipeline([
    # TODO: Define pipeline steps here
    ("preprocessor", preprocessor),
    ("GBM_model", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5))
])

model_pipeline.fit(X_train_t, y_train_t, GBM_model__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators
param_grid = {
    "GBM_model__n_estimators":[100,200,300],
    "GBM_model__learning_rate":[0.01,0.02,0.05,0.1,0.2]
}

cv = GridSearchCV(model_pipeline, param_grid, cv=5)

cv.fit(X_train_t, y_train_t, GBM_model__sample_weight=w_train_t)
best_lgbm = cv.best_estimator_

df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%
#####################################################################################################
################################################ PS4 ################################################
#####################################################################################################
# TODO: Train a constrained LGBM by introducing a monotonicity constraint for BonusMalus into the LGBMRegressor
# Steps
# 1: Create a plot of the average claims per BonusMalus group, make sure to weigh them by exposure.
group_data = df.copy(deep = True)
group_data["claim_x_exposure"] = group_data["ClaimAmountCut"] * group_data["Exposure"]
group_data = group_data.groupby("BonusMalus").agg(
    total_claim = ("claim_x_exposure", "sum"),
    total_exposure = ("Exposure", "sum"),
    )

group_data["weighted_claim"] = group_data["total_claim"]/group_data["total_exposure"]
group_data = group_data.sort_index()

plt.figure(figsize=(8,5))
plt.scatter(group_data.index, group_data["weighted_claim"])
plt.xlabel("Bonus Malus Group")
plt.ylabel("Exposure Weighted Average Claim")
plt.title("Average Claims per Bonus-Malus Group")
plt.show()

# %%
# 2: Create a new model pipeline or estimator called constrained_lgbm. 
# Introduce an increasing monotonicity constrained for BonusMalus
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        # TODO: Add numeric transforms here
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals),
        ("num", numeric_transformer, ["Density"]),
        ("scalar", StandardScaler(), ["BonusMalus"])
    ]
)

preprocessor.set_output(transform="pandas")
constrained_model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("GBM_model", LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, monotone_constraint = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
])

# %%
# 3. Cross-validate and predict using the best estimator
param_grid = {
    "GBM_model__n_estimators":[50,100,200,300],
    "GBM_model__learning_rate":[0.01,0.02,0.05,0.1,0.2]
}

cv = GridSearchCV(constrained_model_pipeline, param_grid, cv=5)

cv.fit(X_train_t, y_train_t, GBM_model__sample_weight=w_train_t)

df_test["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv.best_estimator_.predict(X_train_t)

print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Based on the cross-validated constrained LGBMRegressor object, plot a learning curve which is showing the convergence of the score on the train and test set.
# Steps
# 1. Re-fit the best constrained lgbm estimator from the cross-validation and provide the tuples of the test and train dataset to the estimator via `eval_set`
best_lgbm_pipe = cv.best_estimator_
fitted_preprocess = best_lgbm_pipe[:-1]          
X_train_trans = fitted_preprocess.transform(X_train_t)
X_test_trans  = fitted_preprocess.transform(X_test_t)
best_lgbm_pipe.fit(X_train_t, y_train_t, 
                   GBM_model__sample_weight=w_train_t, 
                   GBM_model__eval_set=[(X_train_trans, y_train_t), (X_test_trans, y_test_t)],
                   GBM_model__eval_names=["train", "test"]
                   )

# %%
# 2. Plot the learning curve by running lgb.plot_metric on the estimator
lgb.plot_metric(best_lgbm_pipe[-1])

# %%
# TODO Write a function evaluate_predictions within the evaluation module, which computes various metrics given the true outcome values and the model’s predictions.
c_LGBM = compute_metrics(df_test["pp_t_lgbm_constrained"], y_test_t, w_test_t, model = "constrained_LGBM")
u_LGBM = compute_metrics(df_test["pp_t_lgbm"], y_test_t, w_test_t, model = "LGBM")
result = pd.concat([c_LGBM, u_LGBM])
result

# %%
# Plots the PDPs of all features and compare the PDPs between the unconstrained and constrained LGBM. 
# Steps
# 1. Define an explainer object using the constrained lgbm model, data and features.
# 2. Compute the marginal effects using model_profile and plot the PDPs.
# 3. Compare the PDPs between the unconstrained and constrained LGBM
exp_c = dx.Explainer(best_lgbm_pipe, X_train_t, y_train_t)
exp_c.model_profile().plot()

exp_u = dx.Explainer(best_lgbm, X_train_t, y_train_t)
exp_u.model_profile().plot()

# %%
# TODO Compare the decompositions of the predictions for some specific row (e.g. the first row of the test set) for the constrained LGBM and our initial GLM.
# Step
# 1. Define DALEX Explainer objects for both.
# 2. Call the method `predict_parts` for each and provide one observation as data point and `type="shap"`.
# 3. Plot both decompositions and compare where they might deviate.
exp_GLM = dx.Explainer(model_pipeline_GLM, X_train_t, y_train_t)
result_GLM = exp_GLM.predict_parts(new_observation = X_test_t.loc[[1]], type = 'shap', B=10, label = "GLM")
result_LGBM = exp_c.predict_parts(new_observation = X_test_t.loc[[1]], type = 'shap', B=10, label = "LGBM")
# %%
result_LGBM.plot(result_GLM)


