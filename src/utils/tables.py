import pandas as pd


df = pd.read_csv("results/results.csv")
df = df.drop_duplicates(
    subset=["model_name", "style", "method", "temperature", "magnitude", "r", "lbd", "seed"], keep="last"
)

style = "ce"
metric = "fpr"
r = 2
pretty_names = {
    "resnet34": "ResNet-34",
    "densenet121": "DenseNet-121",
    "tv": "ResNet-50",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "resnet50": "ImageNet",
    "ce": "CE",
    "lognorm": "LogNorm",
    "mixup": "Mixup",
    "regmixup": "RegMixup",
    "openmix": "OpenMix",
    "ensemble": "Ensemble",
    "mc_dropout": "MC-Dropout",
    "odin": "ODIN",
    "doctor": "Doctor",
    "mlp": "MLP",
    "metric_lagrange": "Ours",
    "msp": "MSP",
}

res = df
res = res.query("r == @r and style == @style")
res = res.sort_values(["model_name", metric, "method"], ascending=True if metric == "auc" else False)
res = res.groupby(["model_name", "method"]).tail(5)
# compute mean and std
res = res.groupby(["model_name", "method"]).agg(
    {"fpr": ["mean", "std"], "auc": ["mean", "std"], "accuracy": ["mean", "std"], "aurc": ["mean", "std"]}
)
# flatten columns
res.columns = ["_".join(x) for x in res.columns.ravel()]
res = res.reset_index()
k1 = "auc_mean"
k2 = "auc_std"
res = res[["model_name", "method", k1, k2]]
res = res.pivot(index="model_name", columns="method", values=[k1,k2])
res = res.reset_index()

# flatten mean and std columns
res.columns = ["_".join(x) for x in res.columns.ravel()]
# reorder columns
res["dataset"] = res["model_name_"].apply(lambda x: x.split("_")[1])
res["model_name"] = res["model_name_"].apply(lambda x: x.split("_")[0])
print(res)
method_names = ["msp","doctor","metric_lagrange"]
print(res.columns)
res = res[
    ["model_name", "dataset"]
    + [f"{k1}_{method}" for method in method_names]
    + [f"{k2}_{method}" for method in method_names]
]
# format std
for col in res.columns:
    if "std" in col:
        res[col] = res[col].apply(lambda x: f"({x*100:.2f})")
# format mean
for col in res.columns:
    if "mean" in col:
        res[col] = res[col].apply(lambda x: f"{x*100:.2f}")

# rename models
res["Model"] = res["model_name"].apply(lambda x: pretty_names[x])
# drop model_name column
res = res.drop(columns=["model_name"])
# rename datasets
res["Dataset"] = res["dataset"].apply(lambda x: pretty_names[x])
res = res.drop(columns=["dataset"])

# unite mean and std
for col in res.columns:
    print(col)
    if "mean" in col:
        res[pretty_names[col.replace(f"{k1}_", "")]] = res[col] + " " + res[col.replace("mean", "std")]
        # drop mean and std columns
        res = res.drop(columns=[col, col.replace("mean", "std")])
# table 1
print(
    res.to_markdown(index=False).replace("scriptsize", "\scriptsize")
    # .replace("DenseNet-121", "")
    # .replace("ResNet-34", "")
    # .replace("ResNet-50", "")
)
