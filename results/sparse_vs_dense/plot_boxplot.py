import json
from pathlib import Path
import matplotlib
# Force non-interative backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#The base code was generated using AI.

def main():

    json_files = [
        Path("./sparsity_behaviour_results/chain_length_5.json"),
        Path("./sparsity_behaviour_results/chain_length_10.json"),
        Path("./sparsity_behaviour_results/chain_length_15.json"),
        Path("./sparsity_behaviour_results/chain_length_20.json")
    ]

    stats_list = []
    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            stats_list.append({
                "label": data.get("label", file_path.stem),
                "med": data["med"],
                "q1": data["q1"],
                "q3": data["q3"],
                "whislo": data["whislo"],
                "whishi": data["whishi"],
                "fliers": data.get("fliers", [])
            })

    fig, ax = plt.subplots(figsize=(8,6))

    box_props = dict(linestyle='-', linewidth=1.5, color='darkblue')
    median_props = dict(linestyle='-', linewidth=2, color='red')

    ax.bxp(
        stats_list,
        showfliers=True,
        boxprops=box_props,
        medianprops=median_props
    )

    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_ylabel("Density", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.set_xlabel("Chain Length", fontsize=16)

    plt.tight_layout()

    # Save the figure to disk alongside displaying
    plt.savefig("box_plot_chain_length_density_last_jacobian.png", dpi=400, bbox_inches="tight")

if __name__ == "__main__":
    main()
