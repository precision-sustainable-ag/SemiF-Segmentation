import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
import pandas as pd

jsonpath = "/home/mkutuga/mmsegmentation/demo/mmsegmentation/work_dirs/test23473/None.log.json"
dfs = []
spedfs = []
with open(jsonpath, "r") as j:
    data = json.load(j)
    reslist = data["results"]
    for idx, resdict in enumerate(reslist):
        if resdict["mode"] == "val":
            df = pd.DataFrame.from_dict(resdict, orient='index').T
            df = df.iloc[-1:]
            spedf = pd.DataFrame(
                df.iloc[:, 7:17].T.reset_index(names="speciesIoU"))
            spedf.columns = spedf.columns.astype(str)
            spedf["epoch"] = resdict["epoch"]
            spedf = spedf.rename(columns={"0": "speciesIoU_values"})
            spedf['epoch'] = resdict['epoch']
            spedf['iter'] = resdict['iter']
            spedf['lr'] = resdict['lr']
            spedf['aAcc'] = resdict['aAcc']
            spedf['mIoU'] = resdict['mIoU']
            spedf['mAcc'] = resdict['mAcc']
            # spedf[]
            spedfs.append(spedf)

            dfs.append(df.reset_index())
df = pd.concat(spedfs)

sns.set_theme(style="darkgrid")
# Plot the responses for different events and regions
sns.lineplot(x="epoch", y="mAcc", data=df, legend="brief")
#place legend outside top right corner of plot
plt.savefig("test.png")
