import csv
from collections import OrderedDict

import wandb

try:
    import wandb
except ImportError:
    pass


def update_summary(
    epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False, resume=""
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([("train_" + k, v) for k, v in train_metrics.items()])
    rowd.update([("eval_" + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        if resume != "":
            with open(filename, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k, v in row.items():
                        row[k] = float(v)
                    wandb.log(row)
        wandb.log(rowd)
    with open(filename, mode="a") as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header and (resume == ""):  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
