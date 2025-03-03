from itertools import combinations
from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from uncertainty_engine.client import Client
from uncertainty_engine.nodes.sensor_designer import (
    BuildSensorDesigner,
    ScoreSensorDesign,
    SuggestSensorDesign,
)
from utils import LOSPlotter, InteractiveHistogram
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
from scipy.stats import norm
import seaborn as sns

# Interactive plot imports
from ipywidgets import (
    IntSlider,
    interactive_output,
    VBox,
    HBox,
    Play,
    jslink,
    fixed,
)
import sys

# Check if running in Google Colab
if "google.colab" not in sys.modules:
    from IPython.display import display


# def enable_plotly_in_cell():
#     display(
#         IPython.core.display.HTML(
#             """<script src="/static/components/requirejs/require.js"></script>"""
#         )
#     )
#     init_notebook_mode(connected=False)


# class InteractiveHistogram:
#     def __init__(self, df, title="EIG Distribution", n_bins=20, bar_color="#EBF38B"):
#         self.df = self.interpret_df(df)
#         self.title = title
#         self.n_bins = n_bins
#         self.bar_color = bar_color
#         self.fig = None
#         self.plot()

#     def interpret_df(self, df):

#         # Sample DataFrame with continuous values
#         return pd.DataFrame(
#             {"Sets": df.keys(), "value": [df[k]["mean_score"] for k in df.keys()]}
#         )

#     def process_data(self):
#         # Define bins
#         bin_size = (self.df["value"].max() - self.df["value"].min()) / self.n_bins
#         bin_edges = np.arange(
#             self.df["value"].min(), self.df["value"].max() + bin_size, bin_size
#         )
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # Assign values to bins
#         self.df["EIG"] = pd.cut(self.df["value"], bins=bin_edges, right=False)

#         # Group by bins and concatenate names
#         df_grouped = (
#             self.df.groupby("EIG")
#             .agg({"value": "count", "Sets": lambda x: ", ".join(x)})
#             .reset_index()
#         )

#         df_grouped["EIG"] = df_grouped["EIG"].astype(
#             str
#         )  # Convert bins to string for display
#         df_grouped["Frequency"] = df_grouped["value"]

#         # Create hover text
#         hover_texts = [
#             f"EIG range: [{round(bin_edges[i], 2)}, {round(bin_edges[i+1], 2)})<br>Sets: {row_name}"
#             for i, row_name in enumerate(df_grouped["Sets"].values)
#         ]

#         return (
#             bin_centers,
#             df_grouped["Frequency"].values,
#             hover_texts,
#             np.diff(bin_edges),
#         )

#     def plot(self):
#         bin_centers, bin_counts, hover_texts, bin_widths = self.process_data()

#         fig = go.Figure()
#         fig.add_trace(
#             go.Bar(
#                 x=bin_centers,  # Continuous x-axis
#                 y=bin_counts,
#                 hoverinfo="text",
#                 hovertext=hover_texts,
#                 width=bin_widths,  # Bin widths
#                 marker=dict(color=self.bar_color, line=dict(width=1, color="black")),
#             )
#         )

#         # Update layout
#         fig.update_layout(
#             title=self.title,
#             xaxis_title="Expected Information Gain (higher is better)",
#             yaxis_title="Frequency",
#             xaxis=dict(tickmode="linear"),
#             bargap=0,  # No gaps between bins
#         )

#         self.fig = fig
#         fig.show()


class Designer:
    def __init__(
        self,
        email: str,
        observables: pd.DataFrame,
        quantities_of_interest: Optional[pd.DataFrame] = None,
        sigma: Optional[Union[float, list[float]]] = None,
    ):
        self.client = Client(
            email=email,
            deployment="https://13qg20i4yc.execute-api.eu-west-2.amazonaws.com/dev/api",
        )

        self.observables = observables
        observables_dict = self.observables.to_dict(orient="list")

        self.quantities_of_interest = quantities_of_interest
        # if quantities_of_interest is not None:
        #     quantities_of_interest = quantities_of_interest.to_dict(orient="list")

        self.sigma = sigma

        builder = BuildSensorDesigner(
            sensor_data=observables_dict,
            quantities_of_interest_data=None,
            sigma=self.sigma,
        )

        response = self.client.run_node(builder)
        if response['status'] == 'FAILURE':
            msg = response['error']['message']
            raise ValueError(msg)
        self.designer = response["output"]["sensor_designer"]

    def visualise_data(self, selected_lines=None, print_str=None):
        plotter = LOSPlotter(self.observables, qoi_df=self.quantities_of_interest)
        output = plotter.display_plot(selected_lines, print_str)
        return output

    visualize_dataset = visualise_data

    def select_design(self):
        self.plotter = LOSPlotter(self.observables, qoi_df=self.quantities_of_interest)
        output = self.plotter.display_interactive_plot()
        return output

    def suggest(self, num_sensors: int, num_eval: int):
        if num_eval > 100:
            raise ValueError("This demo is limited to 100 evaluations")

        suggest_design = SuggestSensorDesign(
            sensor_designer=self.designer,
            num_sensors=num_sensors,
            num_eval=num_eval,
        )

        response = self.client.run_node(suggest_design)

        self.designer = response["output"]["sensor_designer"]

        if num_sensors >= 25:
            objective = "Exact"
        else:
            objective = "GP-Based"

        # Disable GP-based objective function if no quantities of interest are provided
        if (objective == "GP-Based") and (
            len(self.designer["bed"]["quantities_of_interest_df"]) == 0
        ):
            objective = "Exact"

        n_sensor_cache = {}
        for k, v in self.designer["bed"]["cache"]["Exact"].items():
            if len(eval(k)) != num_sensors:
                continue
            for sub_k, sub_v in v.items():
                if sub_k in ["mean_score", "score_var"]:
                    indices = eval(k)  # Ensure this is a list or tuple
                    if isinstance(indices, list):
                        indices = tuple(indices)  # Convert list to tuple

                    new_key = tuple(
                        [
                            list(self.designer["bed"]["sensor_df"][0].keys())[i]
                            for i in indices
                        ]
                    )
                    if new_key in n_sensor_cache:
                        n_sensor_cache[new_key].update({sub_k: sub_v})
                    else:
                        n_sensor_cache[new_key] = {sub_k: sub_v}

        top_5 = dict(
            sorted(
                n_sensor_cache.items(), key=lambda x: x[1]["mean_score"], reverse=True
            )[: min(5, len(n_sensor_cache))]
        )

        print("Suggested sensor combinations:")
        for i, (suggestion, details) in enumerate(top_5.items()):
            print(i, suggestion, f"(EIG: {details['mean_score']:.2f})")
        # return top_5

    def score(self, design: list):
        score_design = ScoreSensorDesign(
            sensor_designer=self.designer,
            design=design,
        )

        response = self.client.run_node(score_design)

        self.designer = response["output"]["sensor_designer"]

        return response["output"]["score"]
    
    def score_design(self, design: list):
        raw_score = self.score(design)
        print(f"EIG for {design}: {raw_score:.2f}")

    def redundancy_analysis(
        self, designs: list, num_dropout: list = [1], objective="Exact", num_iter=1
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for design in designs:
                for length in len(design) - np.array(num_dropout):
                    for combo in combinations(design, length):
                        for _ in range(num_iter):
                            self.score(list(combo))

        redundancy_results = {}
        for design in designs:
            redundancy_results[str(design)] = {}
            for length in len(design) - np.array(num_dropout):
                length = int(length)
                overall_mean = 0
                means = []
                vars = []
                combs = list(combinations(design, length))
                for combo in combs:
                    sensor_ids = list(self.designer["bed"]["sensor_df"][0].keys())
                    combo_inds = [sensor_ids.index(sensor) for sensor in combo]
                    if (
                        str(list(combo_inds))
                        not in self.designer["bed"]["cache"][objective]
                    ):
                        continue
                    means.append(
                        self.designer["bed"]["cache"][objective][str(list(combo_inds))][
                            "mean_score"
                        ]
                    )
                    vars.append(
                        self.designer["bed"]["cache"][objective][str(list(combo_inds))][
                            "score_var"
                        ]
                    )
                overall_mean = np.mean(means)
                overall_var = np.var(means) + np.mean(vars)
                redundancy_results[str(design)][length] = {
                    "mean": overall_mean,
                    "var": overall_var,
                }

        for length in len(design) - np.array(num_dropout):
            x_labels = list(redundancy_results.keys())
            means = [value[length]["mean"] for value in redundancy_results.values()]
            std_devs = [
                np.sqrt(value[length]["var"]) for value in redundancy_results.values()
            ]
            x_indices = range((len(x_labels)))

            # plt.figure(figsize=(10, 10))
            fig, axs = plt.subplots(nrows=1, ncols=len(x_labels))

            # # Plot scatter points for the mean
            # plt.scatter(x_indices, means, color="purple", label="Mean", zorder=3)

            # Plot error bars for 1st and 2nd standard deviations
            for x, mean, std in zip(x_indices, means, std_devs):
                output = LOSPlotter(self.observables)

                sensor_inds = [sensor_ids.index(sensor) for sensor in eval(x_labels[x])]
                mean_eig = self.designer["bed"]["cache"][objective][
                    str(sorted(sensor_inds))
                ]["mean_score"]
                output.display_plot(
                    eval(x_labels[x]),
                    print_str="Full-set :{x_labels[x]}, EIG : {mean_eig}",
                )

                combs = list(combinations(eval(x_labels[x]), length))
                for combo in combs:
                    sensor_inds = [sensor_ids.index(sensor) for sensor in combo]
                    mean_eig = self.designer["bed"]["cache"][objective][
                        str(sorted(sensor_inds))
                    ]["mean_score"]
                    output.display_plot(
                        combo, print_str=f"Sub-set :{combo}, EIG : {mean_eig}"
                    )
            #     # 1st standard deviation error bars
            #     plt.errorbar(
            #         x,
            #         mean,
            #         yerr=std,
            #         fmt="o",
            #         color="purple",
            #         label="1st Std Dev" if x == 0 else "",
            #         zorder=2,
            #         elinewidth=3,
            #     )
            #     # 2nd standard deviation error bars
            #     plt.errorbar(
            #         x,
            #         mean,
            #         yerr=2 * std,
            #         fmt="o",
            #         color="darkorange",
            #         label="2nd Std Dev" if x == 0 else "",
            #         zorder=1,
            #         capsize=5,
            #         elinewidth=3,
            #     )

            # # Customize the plot
            # plt.xticks(x_indices, x_labels, rotation=90)
            # plt.xlabel("Designs")
            # plt.ylabel("GP-EIG")
            # plt.title(f"GPEIG Distributions With {len(design) - length} Sensor Dropout")
            # plt.legend()

            # # Show the plot
            # plt.tight_layout()
            # plt.show()
            # plt.close()

    def visualise_score_distribution(self):
        for objective in self.designer["bed"]["cache"].keys():
            if len(self.designer["bed"]["cache"][objective]) == 0:
                continue

            # Determine the length of design keys and plot separate histograms
            design_lengths = {}
            for design_key in self.designer["bed"]["cache"][objective].keys():
                length = len(eval(design_key))
                if length not in design_lengths:
                    design_lengths[length] = []
                design_lengths[length].append(
                    self.designer["bed"]["cache"][objective][design_key]["mean_score"]
                )

            for length, scores in design_lengths.items():
                plt.figure()
                plt.hist(scores, bins=50, color="#3ab09e", edgecolor="black")
                plt.xlabel(f"EIG")
                plt.ylabel("Frequency")
                plt.title(
                    f"EIG Distribution for {objective} Objective Function (Design Length: {length})"
                )
                plt.show()
                plt.close()

    def visualise_score_distribution_dg(self):
        for objective in self.designer["bed"]["cache"].keys():
            if len(self.designer["bed"]["cache"][objective]) == 0:
                continue
            # for design_key in list(self.designer["bed"]["cache"][objective].keys())[0]:
            #     length = len(eval(design_key))
            #     title = f"EIG Distribution for {objective} Objective Function (Design Length: {length})"
            InteractiveHistogram(df=self.designer["bed"]["cache"][objective])

    visualise_score_distribution = visualise_score_distribution_dg

    visualize_score_distribution = visualise_score_distribution

    def visualise_posterior(
        self, design: list, sigma: Optional[Union[float, list[float]]] = None
    ):
        # If sigma is not specified, use the default value
        if sigma is None:
            sigma = self.sigma

        # Calculate the likelihoods
        data = self.observables.values
        indices = [self.observables.columns.get_loc(sensor) for sensor in design]

        # Set the sigma values for the chosen design
        if isinstance(sigma, list):
            design_sigma = (np.array(sigma)[list(sorted(set(indices)))]).tolist()
        else:
            design_sigma = sigma

        sensor_data = data[:, indices]

        # calculate the likelihoods
        qois = list(self.quantities_of_interest.columns)
        means = np.zeros((len(self.observables), len(qois)))
        stds = np.zeros((len(self.observables), len(qois)))
        for i in range(len(self.observables)):
            sample_data = data[i, indices]
            noise = norm(loc=0, scale=design_sigma).rvs(len(sample_data))
            likelihood = np.exp(
                self._log_likelihood(sensor_data, sample_data + noise, design_sigma)
            )
            likelihood /= likelihood.sum()
            means[i] = np.dot(likelihood, self.quantities_of_interest.values)
            expected_sqr = np.dot(likelihood, self.quantities_of_interest.values**2)
            interim = expected_sqr - means[i] ** 2
            interim[interim < 0] = 0
            stds[i] = np.sqrt(interim)

        print(f"Average Relative Uncertainty = {np.mean(stds / means):.5f}")

        # Plot the posterior
        slider = IntSlider(
            min=0,
            max=len(self.observables) - 1,
            step=1,
            value=0,
            continuous_update=True,
        )
        play = Play(
            interval=50,
            value=0,
            min=0,
            max=len(self.observables) - 1,
            step=1,
            description="Press play",
            disabled=False,
            repeat=True,
        )

        # Link the play button to the slider
        jslink((play, "value"), (slider, "value"))
        controls = HBox([play, slider])

        # Make the interactive plot link and return
        out = interactive_output(
            self.plot_zeff, {"run": slider, "means": fixed(means), "stds": fixed(stds)}
        )

        # Display the combined controls and the output
        display(VBox([controls, out]))

    def plot_zeff(self, run, means, stds):
        mean = means[run]
        std = stds[run]

        rhos = np.linspace(0, 1, len(mean))
        plt.plot(rhos, mean, label="Mean", color="#000000")
        plt.plot(
            rhos,
            self.quantities_of_interest.iloc[run].values,
            label="True",
            linestyle="--",
            color="#EBF38B",
        )
        plt.fill_between(
            rhos,
            mean - 2 * std,
            mean + 2 * std,
            color="#8AA0AD",
            label=r"$2^{nd}$ Std Dev",
        )
        plt.fill_between(
            rhos,
            mean - std,
            mean + std,
            color="#45687C",
            label=r"$1^{st}$ Std Dev",
        )
        plt.xlim(0, 1)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [1, 0, 3, 2]
        plt.legend([handles[i] for i in order], [labels[i] for i in order])

        plt.xlabel(r"ρ")
        plt.ylabel(r"$Z_{eff}$")
        plt.savefig("high_res_plot.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    # def visualise_posterior(
    #     self,
    #     design: list,
    #     sigma: Optional[Union[float, list[float]]] = None,
    #     sample: Optional[int] = None,
    #     functional_qoi: Optional[bool] = False,
    # ):
    #     # If sample is not specified, select a random sample
    #     if sample is None:
    #         sample = np.random.randint(0, len(self.observables))

    #     # If sigma is not specified, use the default value
    #     if sigma is None:
    #         sigma = self.sigma

    #     # Calculate the likelihoods
    #     data = self.observables.values
    #     indices = [self.observables.columns.get_loc(sensor) for sensor in design]

    #     # Set the sigma values for the chosen design
    #     if isinstance(sigma, list):
    #         design_sigma = (np.array(sigma)[list(sorted(set(indices)))]).tolist()
    #     else:
    #         design_sigma = sigma

    #     sensor_data = data[:, indices]
    #     sample_data = data[sample, indices]

    #     noise = norm(loc=0, scale=design_sigma).rvs(len(sample_data))
    #     likelihood = np.exp(
    #         self._log_likelihood(sensor_data, sample_data + noise, design_sigma)
    #     )
    #     likelihood /= likelihood.sum()

    #     # If there is no quantities of interest, plot a histogram
    #     if self.quantities_of_interest is None:
    #         colours = [
    #             "#3ab09e" if i != sample else "#111122" for i in range(len(data))
    #         ]
    #         fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    #         axs.bar(range(len(self.observables)), likelihood, color=colours, width=4)
    #         axs.set_title(f"Normalised Likelihoods for Sample {sample}")
    #         axs.set_ylabel("Likelihood")
    #         axs.set_xlabel("Sample")

    #         custom_labels = ["Sample Likelihood", "True Sample Likelihood"]
    #         custom_colors = ["#3ab09e", "#111122"]
    #         axs.legend(
    #             handles=[
    #                 plt.Line2D([0], [0], color=color, lw=4) for color in custom_colors
    #             ],
    #             labels=custom_labels,
    #         )
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()

    #     # elif functional_qoi:
    #     if functional_qoi:
    #         qois = list(self.quantities_of_interest.columns)
    #         trues = self.quantities_of_interest.iloc[sample].values
    #         means = np.zeros(len(qois))
    #         stds = np.zeros(len(qois))

    #         for i, qoi in enumerate(qois):
    #             means[i] = np.sum(self.quantities_of_interest[qoi].values * likelihood)
    #             expected_sqr = np.sum(
    #                 self.quantities_of_interest[qoi].values ** 2 * likelihood
    #             )
    #             stds[i] = np.sqrt(expected_sqr - means[i] ** 2)

    #         rhos = np.linspace(0, 1, len(qois))
    #         plt.plot(rhos, means, label="Mean", color="#000000")
    #         plt.plot(
    #             rhos,
    #             trues,
    #             label="True",
    #             linestyle="--",
    #             color="#EBF38B",
    #         )
    #         plt.fill_between(
    #             rhos,
    #             means - 2 * stds,
    #             means + 2 * stds,
    #             color="#8AA0AD",
    #             label=r"$2^{nd}$ Std Dev",
    #         )
    #         plt.fill_between(
    #             rhos,
    #             means - stds,
    #             means + stds,
    #             color="#45687C",
    #             label=r"$1^{st}$ Std Dev",
    #         )
    #         plt.xlim(0, 1)

    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         order = [0, 1, 3, 2]
    #         plt.legend([handles[i] for i in order], [labels[i] for i in order])

    #         plt.xlabel(r"ρ")
    #         plt.ylabel(r"$Z_{eff}$")
    #         plt.savefig("high_res_plot.png", dpi=300, bbox_inches="tight")
    #         plt.show()
    #         plt.close()

    #     else:
    #         rows = np.random.choice(
    #             a=np.arange(len(likelihood)), size=10_000, p=likelihood
    #         )

    #         # Plot the posterior
    #         prior_pair_plot = sns.pairplot(
    #             self.quantities_of_interest.iloc[rows],
    #             kind="kde",
    #             diag_kind="hist",
    #             diag_kws={"bins": 20},
    #         )
    #         prior_pair_plot.figure.suptitle(
    #             f"Quantity of Interest Posterior, Sample {sample}, True {', '.join([f'{x:.2f}' for x in self.quantities_of_interest.iloc[sample].values.tolist()])}",
    #             fontsize=8,
    #         )
    #         prior_pair_plot.figure.tight_layout()
    #         plt.show()
    #         plt.close()

    visualize_posterior = visualise_posterior

    def _log_likelihood(self, f, d, sigma):
        """Calculate the log likelihood of the model outputs."""
        return norm(loc=f, scale=sigma).logpdf(d).sum(axis=1)

    @property
    def best_design(self):
        best_ind = eval(
            list(
                dict(
                    sorted(
                        self.designer["bed"]["cache"]["Exact"].items(),
                        key=lambda x: x[1]["mean_score"],
                        reverse=True,
                    )
                ).keys()
            )[0]
        )

        return np.array(list(self.designer["bed"]["sensor_df"][0].keys()))[best_ind]

    @property
    def worst_design(self):
        worst_ind = eval(
            list(
                dict(
                    sorted(
                        self.designer["bed"]["cache"]["Exact"].items(),
                        key=lambda x: x[1]["mean_score"],
                    )
                ).keys()
            )[0]
        )
        return np.array(list(self.designer["bed"]["sensor_df"][0].keys()))[worst_ind]

    @property
    def chosen_design(self, lim_los=4):
        return self.plotter.chosen_design[:lim_los]
