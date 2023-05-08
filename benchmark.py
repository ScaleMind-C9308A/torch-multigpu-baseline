import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob

if __name__ == "__main__":
    
    palette = sns.color_palette("rocket_r")
    
    parquet_files = glob(os.getcwd() + "/*.parquet")
    dfs = []
    
    for par_file in parquet_files:
        ex_ref = par_file.split("/")[-1][:-8]
        info_split = ex_ref.split("_")
        bs, lr, warm = int(info_split[0]), float(info_split[1]), info_split[2]
        base_df = pd.read_parquet(par_file).iloc[:100, :]
        base_df["batch_size"] = [bs]*100
        base_df["learning_rate"] = [0.3]*100 if warm == "True" else [lr]*100
        base_df["warm_up"] = [warm]*100
        base_df["epoch"] = range(100)
        dfs.append(base_df)
    
    main_df = pd.concat(dfs)
    
    # train_loss
    plt.figure()
    sns.relplot(
        data=main_df,
        x="epoch", y="train_loss",
        hue="batch_size", size="learning_rate", col="warm_up",
        kind="line", 
        # size_order=["T1", "T2"], 
        palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )
    plt.savefig("train_loss")
    
    # test_loss
    plt.figure()
    sns.relplot(
        data=main_df,
        x="epoch", y="test_loss",
        hue="batch_size", size="learning_rate", col="warm_up",
        kind="line", 
        # size_order=["T1", "T2"], 
        palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )
    plt.savefig("test_loss")
    
    # train_acc
    plt.figure()
    sns.relplot(
        data=main_df,
        x="epoch", y="train_acc",
        hue="batch_size", size="learning_rate", col="warm_up",
        kind="line", 
        # size_order=["T1", "T2"], 
        palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )
    plt.savefig("train_acc")
    
    # test_acc
    plt.figure()
    sns.relplot(
        data=main_df,
        x="epoch", y="test_acc",
        hue="batch_size", size="learning_rate", col="warm_up",
        kind="line", 
        # size_order=["T1", "T2"], 
        palette=palette,
        height=5, aspect=.75, facet_kws=dict(sharex=False),
    )
    plt.savefig("test_acc")