import glob
import json
import os
import random
import time

import choicemodels
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_create_dir(directory):
    """Provided a directory path, creates that directory if necessary."""
    if "/" in str(directory):
        parts = os.path.split(directory)
        if os.path.isfile(directory) or "." in parts[1]:
            outfile = directory
            print("Getting directory from filepath: {}".format(outfile))
            directory = parts[0]
        if not os.path.exists(directory):
            # still do a try/except because multiprocessing
            try:
                os.makedirs(directory)
                print("Created directory for: {}".format(directory))
            except FileExistsError:
                print("File already exists: {}".format(directory))
    else:
        print("Incomplete path given: {}".format(directory))


def read_predicted_model(model_dir, county, modelname):
    """Utility function reads MNL model results from JSON file"""
    modelfile = model_dir.joinpath("results", f"{county}-{modelname}.json")
    with open(modelfile, "r") as openfile:
        model = json.load(openfile)
    model_exp = model["exp"]
    data = model[modelname]
    data["fit_parameters"] = pd.read_json(data["fit_parameters"], orient="split")
    results = choicemodels.MultinomialLogitResults(model_expression=model_exp, results=data)
    return results


def run_location_choice_mnl(obs, alts, mdexp, taz_sample_size=None):
    """Runs MNL on ready obs and alternatives data and outputs ChoiceModelResults() object"""
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, "chosenTAZ", sample_size=taz_sample_size)
    model = choicemodels.MultinomialLogit(mct, mdexp)
    return model.fit()


def summarize_results(modeltype):
    results_list = glob.glob(f"../data/intermediate/LocationChoice/{modeltype}/Results/*.json")
    output = []

    def p_value_categorizer(x):
        if x < 0.001:
            return "***"
        elif x < 0.01:
            return "**"
        elif x < 0.1:
            return "*"
        else:
            return ""

    for model in results_list:
        with open(model, "r") as openfile:
            result = json.load(openfile)
        county = model.rsplit("/")[-1].rsplit("-")[0]
        sector = model.rsplit("/")[-1].rsplit("-")[1][:-5]
        model_coef = pd.read_json(result[sector]["fit_parameters"], orient="split")
        model_coef["P-Values"] = model_coef["P-Values"].apply(p_value_categorizer)
        model_vars = result["exp"].rsplit("+")
        model_vars.insert(0, "constant")
        model_vars = [i.strip() for i in model_vars]
        model_coef["Coefficient"] = (
            model_coef["Coefficient"].round(3).astype(str) + " " + model_coef["P-Values"]
        )
        model_coef = model_coef.assign(vars=model_vars)
        model_coef = model_coef.filter(items=["vars", "Coefficient"]).transpose()
        model_coef.columns = model_coef.iloc[0]
        model_coef = model_coef.iloc[1:]
        model_coef = model_coef.assign(
            county=county,
            sector=sector,
            rho_squared=result[sector]["log_likelihood"]["rho_squared"],
            model_converged=result[sector]["log_likelihood"]["model_converged"],
        )
        output.append(model_coef)
    output = pd.concat(output, axis=0)
    output = output.set_index(["sector", "county"]).sort_index()
    return output


# UTILS from https://github.com/AZMAG/smartpy_core/blob/master/smartpy_core/wrangling.py


class HowLong:
    """
    Context manager for for measuring time it takes to do something.
    Wrap a block of code into the with statement, after leaving the
    block the time will be printed.

    Modified from https://github.com/AZMAG/smartpy_core/blob/master/smartpy_core/wrangling.py

    Usage:
    ------
    t = HowLong()
    with t:
        # logic to time
    """

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        e = time.perf_counter()
        delta = e - self.start
        if delta <= 60:
            # minute or less
            tqdm.write(f"{self.message} - {delta:.2f} seconds")
        elif delta <= 3600:
            # hour or less
            minutes = int(delta / 60)
            secs = int(delta - (minutes * 60))
            tqdm.write(f"{self.message} - {minutes} minutes, {secs} seconds")
        else:
            hours = int(delta / 3600)
            minutes = int((delta - (hours * 3600)) / 60)
            secs = int(delta - (hours * 3600) - (minutes * 60))
            tqdm.write(
                f"{self.message} - {hours} hours, {minutes} minutes {secs}".format(hours, minutes)
            )


class Seeded(object):
    """
    Context manager for handling reproduceable random seed sequences. Mangages
    both python and numpy random states.
    Parameters:
    -----------
    seed: int
        Seed to initialize the sequence.
    """

    def _get_states(self):
        return random.getstate(), np.random.get_state()

    def _set_states(self, py_state, np_state):
        random.setstate(py_state)
        np.random.set_state(np_state)

    def __init__(self, seed):

        # store the provided seed so we can fetch later if needed
        self.__provided_seed = seed

        # temporarily capture the current state
        orig_py_state, orig_np_state = self._get_states()

        # init new states based on the provided seed
        random.seed(seed)
        np.random.seed(seed)

        self._py_state = random.getstate()
        self._np_state = np.random.get_state()

        # revert back to the original state
        self._set_states(orig_py_state, orig_np_state)

    def __enter__(self):
        """
        Temporarily sets the random state to the current seed state,
        called when entering a `with` block.
        """
        self._old_py_state, self._old_np_state = self._get_states()
        self._set_states(self._py_state, self._np_state)
        return self

    def __exit__(self, *args):
        """
        Reverts the random state to the default random environment,
        called when leaving a `with` block.
        """
        self._py_state, self._np_state = self._get_states()
        self._set_states(self._old_py_state, self._old_np_state)

    def __call__(self, func, *args, **kwargs):
        """
        Executes the provided function and arguments with the current seed state,
        accepts function arguments as either args or named keyword args.
        Parameters:
        -----------
        func: callable
            Function to call using the current seeded state
        Returns:
        --------
        Results of the function
        """
        with self:
            results = func(*args, **kwargs)
        return results

    def get_seed(self):
        """
        Returns the provided seed value.
        """
        return self.__provided_seed


# TODO: Make this compatible with different county models
# def make_shiny_modelparams():
#     sectors = ["AGREMPN", "OTHEMPN", "MWTEMPN", "RETEMP", "FPSEMPN", "HEREMPN"]
#     for sector in sectors:
#         model_params = read_predicted_model("06081", "%s" % sector).report_fit()
#         with open(
#             Path.home().joinpath(
#                 "projects/24182-bart-link21-modeling/shinyapp/ELCM/model_pars/%s.txt" % sector
#             ),
#             "w",
#         ) as f:
#             f.write(model_params)
#     return None


def summarize_mnl_monte(mnl_out):
    plot_labels = mnl_out.iloc[:, 0].to_list()
    mnl_out = mnl_out.transpose().reset_index()
    all_data = {}
    for data_type in ["Beta", "Tscore"]:
        plot_data = {}
        temp_data = mnl_out[mnl_out["index"].str.contains(data_type)].iloc[:, 1:]
        for i in range(len(mnl_out.columns.to_list()) - 1):
            plot_data[i] = temp_data[i]
        # labels, data = [*zip(*plot_data.items())]
        data = [*zip(*plot_data.items())[1]]
        all_data[data_type] = data
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    # rectangular box plot
    ax1.boxplot(
        all_data["Beta"],
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=plot_labels,
    )  # will be used to label x-ticks
    ax1.set_title("LCM Coefficientss")
    # notch shape box plot
    ax2.boxplot(
        all_data["Tscore"],
        notch=False,  # notch shape
        vert=True,  # vertical box alignment
        patch_artist=True,  # fill with color
        labels=plot_labels,
    )  # will be used to label x-ticks
    ax2.set_title("LCM T-Score")
    for ax in [ax1, ax2]:
        ax.yaxis.grid(True)
        ax.tick_params(labelrotation=90)
    return plt.show()
