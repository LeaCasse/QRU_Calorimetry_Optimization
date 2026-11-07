"""Reproduce the manuscript's detailed plots and its data and tuning tables."""
import argparse
from pathlib import Path
import shutil

from build_outputs import FIG, TAB, ROOT, json, load_data, np, plt, QRU, runs, save, table


def read_result(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text())


def metrics(group):
    return [r["validation"]["mse"] for r in group], [r["validation"]["accuracy"] for r in group]


def curves(ax, group, metric, label, x=None, **kwargs):
    traces = [r["fit"]["trace"] for r in group]
    values = np.array([[point[metric] for point in trace] for trace in traces])
    epochs = np.array([point["epoch"] for point in traces[0]])
    axis = epochs if x is None else x
    return ax.plot(axis, values.mean(axis=0), label=label, **kwargs)[0]


def rate_label(rate):
    if rate >= .001:
        return f"{rate:g}"
    mantissa, exponent = f"{rate:.0e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def estimate(values):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def percentage(mean, sd):
    return rf"${100*mean:.2f}\pm {100*sd:.2f}$"


def main(overleaf=None):
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    sw = runs("sweeps")
    extended = read_result("extended_sweeps")

    def group(sweep, key, value):
        return [r for r in sw if r["config"]["group"] == sweep and r["config"][key] == value]

    data = load_data()
    partitions = ("train", "val", "test")
    counts = [[np.count_nonzero(data[f"y_{part}"] == code) for part in partitions] for code in (-1, 0, 1)]
    table("paper_counts", "Dataset and class balance", "tab:dataset",
          ["Class code", "Training", "Validation", "Test"],
          [[str(code), *row] for code, row in zip((-1, 0, 1), counts)]
          + [["Total", *np.sum(counts, axis=0)]])

    depths = [1, 2, 3, 4, 6, 10, 14]
    for metric, label, key in (("val_mse", "Validation MSE", "mse"),
                                ("val_accuracy", "Validation accuracy", "accuracy")):
        fig, ax = plt.subplots(figsize=(7, 3.15))
        for depth in depths:
            curves(ax, group("depth", "depth", depth), metric, str(depth))
        ax.set(xlabel="Completed epochs", ylabel=label)
        if key == "mse":
            ax.set_yscale("log")
        ax.legend(title="Depth", loc="lower center", bbox_to_anchor=(.5, 1.02),
                  ncol=len(depths), frameon=False, fontsize=10, title_fontsize=10)
        save(fig, f"paper_depth_{key}")

    timing = read_result("runtime_depth")["runs"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.7))
    for i, ax in enumerate(axs[:2]):
        points = [estimate(metrics(group("depth", "depth", z))[i]) for z in depths]
        mean, sd = np.array(points).T
        ax.errorbar(depths, mean, yerr=sd, fmt="o-", capsize=3)
    times = np.array([[r["train_seconds"] for r in timing if r["model"] == f"QRU{z}"] for z in depths])
    med = np.median(times, axis=1)
    axs[2].errorbar(depths, med, yerr=[med-times.min(1), times.max(1)-med], fmt="o-", capsize=3)
    for ax, label in zip(axs, ("Validation MSE", "Validation accuracy", "Training (s)")):
        ax.set(xlabel="Depth", ylabel=label, xticks=depths)
    save(fig, "paper_depth_cost")

    model = QRU(10, dimensions=1)
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.7), sharey=True)
    for ax, low, high, labels in zip(axs, [-np.pi, 0], [np.pi, 2*np.pi],
                                   [(r"$-\pi$", "0", r"$\pi$"), ("0", r"$\pi$", r"$2\pi$")]):
        x = np.linspace(low, high, 401)
        for seed in range(9):
            ax.plot(x, model.forward(model.initial(seed), x[:, None]), lw=1)
        ax.set(xlabel="Input angle", xticks=[low, (low+high)/2, high], xticklabels=labels, ylim=(-1.05, 1.05))
    axs[0].set_ylabel(r"$\langle Z\rangle$")
    save(fig, "paper_scaling_functions")

    templates = ["XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.7))
    for i, (ax, label) in enumerate(zip(axs, ["Validation MSE", "Validation accuracy"])):
        mean, sd = np.array([estimate(metrics(group("axes", "axes", template))[i])
                             for template in templates]).T
        ax.errorbar(range(len(templates)), mean, yerr=sd, fmt="o", capsize=3)
        ax.set(ylabel=label, xticks=range(len(templates)), xticklabels=templates)
    save(fig, "paper_two_axis")

    def trajectory_pair(settings, name, legend_title):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.1))
        for ax, metric, ylabel in zip(axs, ["val_mse", "val_accuracy"], ["Validation MSE", "Validation accuracy"]):
            for label, values in settings:
                ax.plot(range(len(values[metric])), values[metric], label=label)
            ax.set(xlabel="Completed epochs", ylabel=ylabel)
        axs[0].set_yscale("log")
        axs[1].legend(title=legend_title, frameon=False, loc="lower center",
                      bbox_to_anchor=(.5, 1), ncol=min(len(settings), 3))
        save(fig, name)

    def means(group):
        return {m: np.mean([[p[m] for p in r["fit"]["trace"]] for r in group], axis=0)
                for m in ("val_mse", "val_accuracy")}

    trajectory_pair([(axis, means(group("axes", "axes", axis))) for axis in ("XYX", "XYZ")],
                    "paper_triplet", "Block")

    fig, axs = plt.subplots(2, 2, figsize=(8.3, 5.4))
    for row, metric in enumerate(("val_mse", "val_accuracy")):
        for batch in [1, 5, 20, 100, 256, "full"]:
            batch_runs = group("batch", "batch_size", batch)
            label = "Full (4,096)" if batch == "full" else str(batch)
            updates = batch_runs[0]["fit"]["steps"]
            steps = np.arange(len(batch_runs[0]["fit"]["trace"])) * (updates // 10)
            for col, axis in enumerate((None, steps)):
                line = curves(axs[row, col], batch_runs, metric, label, axis,
                              marker="o", markevery=[-1], ms=4)
            if batch == "full":
                axs[row, 1].annotate("10 updates", (steps[-1], line.get_ydata()[-1]),
                                     xytext=(7, 0), textcoords="offset points", fontsize=9)
        axs[row, 1].set_xscale("symlog", linthresh=1)
        axs[row, 1].set(xticks=[0, 1, 10, 100, 1000, 10000],
                        xticklabels=["0", "1", "10", "100", "1,000", "10,000"])
    for ax in axs[0]:
        ax.set(ylabel="Validation MSE", yscale="log")
    for ax in axs[1]:
        ax.set_ylabel("Validation accuracy")
    for ax in axs[:, 0]:
        ax.set_xlabel("Completed epochs")
    for ax in axs[:, 1]:
        ax.set_xlabel("Optimizer updates")
    axs[1, 0].legend(title="Batch size", frameon=False, loc="center right", ncol=2)
    save(fig, "paper_batch")

    base_optimizers = ["SGD", "RMSprop", "Adam", "Adagrad", "Adadelta"]
    optimizer_stats = {}
    for name in base_optimizers:
        subset = group("optimizer", "optimizer", name)
        mse, accuracy = metrics(subset)
        optimizer_stats[name] = (subset[0]["config"]["lr"], *estimate(mse), *estimate(accuracy))
    for row in extended:
        if row["optimizer"] != "Adam":
            mse, acc = row["val_mse"], row["val_accuracy"]
            optimizer_stats[row["setting"]] = (row["rate"], mse["mean"], mse["sd"], acc["mean"], acc["sd"])
    order = ["Adam", "Adamax", "NAdam", "AdamW", "Adagrad", "Adadelta"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.9))
    for ax, idx, ylabel in zip(axs, [1, 3], ["Validation MSE", "Validation accuracy"]):
        ax.errorbar(range(len(order)), [optimizer_stats[n][idx] for n in order],
                    yerr=[optimizer_stats[n][idx+1] for n in order], fmt="o", capsize=3)
        ax.set(ylabel=ylabel, xticks=range(len(order)), xticklabels=order)
        ax.tick_params(axis="x", labelrotation=24)
    save(fig, "paper_optimizers")
    rows = []
    for name, (rate, mse, _, acc, sd) in optimizer_stats.items():
        rows.append([name, rate_label(rate), f"{mse:.4f}", percentage(acc, sd)])
    table("paper_optimizers", "Optimizer settings", "tab:optimizers",
          ["Optimizer", "Learning rate", "Validation MSE", "Validation accuracy (\\%)"], rows)

    residual = np.linspace(-2, 2, 501)
    fig, ax = plt.subplots(figsize=(5.3, 2.8))
    ax.plot(residual, abs(residual), label="L1")
    ax.plot(residual, residual**2, label="L2")
    ax.plot(residual, np.where(abs(residual)<=1, .5*residual**2, abs(residual)-.5), label="Huber")
    ax.set(xlabel="Residual", ylabel="Loss per example")
    ax.legend(frameon=False)
    save(fig, "paper_loss_shape")
    trajectory_pair([(name, means(group("loss", "loss", name))) for name in ("L1", "L2", "Huber")],
                    "paper_loss_curves", "Training loss")

    extra_rates = {r["rate"]: r for r in extended if r["optimizer"] == "Adam"}
    rates = sorted({r["config"]["lr"] for r in sw if r["config"]["group"] == "learning_rate"} | set(extra_rates))
    rate_traces, rows, rate_stats = {}, [], []
    for rate in rates:
        if rate in extra_rates:
            result = extra_rates[rate]
            rate_traces[rate] = result["trace"]
            train, val, mse = [[result[key][k] for k in ("mean", "sd")]
                               for key in ("train_accuracy", "val_accuracy", "val_mse")]
        else:
            subset = group("learning_rate", "lr", rate)
            rate_traces[rate] = means(subset)
            train = estimate([r["training"]["accuracy"] for r in subset])
            mse, val = map(estimate, metrics(subset))
        rate_stats.append([*mse, *val])
        rows.append([rate_label(rate), f"{mse[0]:.4f}", percentage(*train), percentage(*val)])
    shown_rates = [5e-7, 5e-6, 5e-5, .03, .5]
    trajectory_pair([(rate_label(rate), rate_traces[rate]) for rate in shown_rates], "paper_rate_curves", "Adam rate")
    table("paper_rates", "Learning rate sweep", "tab:rates",
          ["Rate", "Validation MSE", "Training acc. (\\%)", "Validation acc. (\\%)"], rows)
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    vals = np.array(rate_stats)
    for ax, i, ylabel in zip(axs, [0, 2], ["Validation MSE", "Validation accuracy"]):
        ax.errorbar(range(len(rates)), vals[:, i], yerr=vals[:, i+1], fmt="o", capsize=3)
        ax.set(ylabel=ylabel, xticks=range(len(rates)), xticklabels=[rate_label(r) for r in rates])
        ax.tick_params(axis="x", labelrotation=38)
    save(fig, "paper_rate_endpoints")

    fig, ax = plt.subplots(figsize=(6, 2.8))
    for name, start in (("Adam", .003), ("SGD", .03)):
        ax.step([0, 25, 37, 50], [start, start/10, start/100, start/100],
                where="post", label=name)
    ax.set(xlabel="Completed epochs", ylabel="Scheduled learning rate", yscale="log", xticks=[0, 25, 37, 50])
    ax.legend(frameon=False)
    save(fig, "paper_rate_schedule")
    schedule_runs = [r for r in sw if r["config"]["group"] == "schedule"]
    trajectory_pair([(name+(" scheduled" if mode else " fixed"),
                      means([r for r in schedule_runs if r["config"]["optimizer"] == name
                             and r["config"]["schedule"] == mode]))
                     for name in ("Adam", "SGD") for mode in (False, True)],
                    "paper_schedule_curves", None)

    for name, stem in [("bayesian_search", "paper_bayesian"), ("hyperband_search", "paper_hyperband")]:
        rr = read_result(name)
        fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
        for ax, key, ylabel in zip(axs, ["mse", "accuracy"], ["Validation MSE", "Validation accuracy"]):
            if name == "bayesian_search":
                values = [r["result"]["validation"][key] for r in rr]
                trials = [r["trial"] for r in rr]
                ax.plot(trials, values, "o", ms=3, alpha=.55)
                incumbent = np.minimum.accumulate if key == "mse" else np.maximum.accumulate
                ax.plot(trials, incumbent(values), lw=1.4)
                ax.axvline(10.5, ls=":", color="gray", lw=.9)
                ax.set_xlabel("Bayesian trial")
            else:
                for budget in [1, 3, 9, 27]:
                    results = {r["result"]["id"]: r["result"]["validation"][key] for r in rr
                               if r["result"]["config"]["epochs"] == budget}
                    ax.scatter([budget]*len(results), list(results.values()), alpha=.55, s=17)
                ax.set(xlabel="Hyperband epochs", xscale="log", xticks=[1, 3, 9, 27], xticklabels=[1, 3, 9, 27])
            ax.set_ylabel(ylabel)
        axs[0].set_yscale("log")
        save(fig, stem)

    compare = runs("comparison")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for kind, depth, label in [("QRU", 4, "QRU, L=4"), ("QRU", 10, "QRU, L=10"),
                               ("VQC", 10, "VQC, L=10"), ("MLP", 10, "MLP, scalar")]:
        subset = [r for r in compare if r["config"]["kind"] == kind and r["config"]["depth"] == depth]
        for ax, metric in zip(axs, ("val_mse", "train_accuracy", "val_accuracy")):
            line = curves(ax, subset, metric, label)
            values = np.array([[p[metric] for p in r["fit"]["trace"]] for r in subset])
            sd = values.std(axis=0, ddof=1)
            ax.fill_between(line.get_xdata(), values.mean(0)-sd, values.mean(0)+sd, alpha=.12,
                            color=line.get_color(), linewidth=0)
    for ax, ylabel in zip(axs, ["Validation MSE", "Training accuracy", "Validation accuracy"]):
        ax.set(xlabel="Completed epochs", ylabel=ylabel)
    axs[0].set_yscale("log")
    axs[2].legend(frameon=False, loc="lower right")
    save(fig, "paper_comparison")

    if overleaf:
        for source, destination, extension in [(FIG, "Images", "pdf"), (TAB, "tables", "tex")]:
            target = overleaf / destination
            target.mkdir(parents=True, exist_ok=True)
            for file in source.glob(f"paper_*.{extension}"):
                shutil.copy2(file, target / file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overleaf", type=Path)
    main(parser.parse_args().overleaf)
