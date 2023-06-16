from pathlib import Path

import numpy as np
import pandas as pd
from mappings import ind_prof_map, occ_det_soc_map, occ_to_census_upd
from pandas.api.types import CategoricalDtype
from utils import agebreaker2

# import matplotlib.pyplot as plt


# plt.style.use("ggplot")

INPUT_REMI_PATH = Path(
    "~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Data/LandEcon Group REMI"
)
background_path = Path("~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Background")
INPUT_BASEDATA_PATH = background_path.joinpath(
    "MTC REMI Data/MTC Shared Folder/Household Forecast/base_data"
)
INPUT_MAPPINGS_PATH = background_path.joinpath(
    "MTC REMI Data/MTC Shared Folder/Household Forecast/mappings"
)

breaks_5 = list(range(0, 86, 5)) + [np.inf]


def easer(
    target_series=None,
    source_series=None,
    t_0=7,
    t_1=15,
    envelope_year_start=2015,
    envelope_year_end=2050,
):
    """MTC HHPROJ-A specific function that sinusoidally eases from one value to another over time t"""
    # We 'ease out' the difference between the series of interest and the target series
    # when the easing out of the difference is done (difference is 0) the two are identical

    # x-coordinates
    index = np.arange(envelope_year_end - envelope_year_start + 1)

    easing_window = t_1 - t_0

    # Use a sinusoidal easing function, set to converge in a relevant range between t_0 and t_1
    x = np.pi * (index - t_0) / easing_window
    easing = 0.5 * np.cos(x) + 0.5

    # To the left of the convergence window, easing should be 1 (i.e. = source series)
    easing[index < t_0] = 1

    # To the right of the convergence window, it should be 0 (i.e. no difference to,
    # or fully transitioned to target series)

    easing[index >= t_1] = 0

    # turn in to a pd.Series
    easing = pd.Series(
        easing, index=pd.Index(np.arange(envelope_year_start, envelope_year_end + 1), name="Year")
    )
    easing.index = easing.index.set_names("Year")
    target_series.index = target_series.index.astype(str)
    source_series.index = source_series.index.astype(str)
    easing.index = easing.index.astype(str)

    source_less_target = source_series - target_series
    output = (
        source_less_target.replace(np.nan, 0).mul(easing, axis=0).replace(np.nan, 0) + target_series
    ).stack()
    output.name = "value"
    return output.unstack("Year").rename(columns=lambda x: str(x))


#########################################
#      Import MTC Background data       #
#########################################
# TODO PBA2040 is by region, update to PBA 2050?
# Can probably be deleted as it should not matter


def import_pba2040_projections():
    """Function imports PBA 2040 population projections and sorts based on gender, race, ethnicity and age group"""
    print("Importing Project Bay Area 2040 projections")
    remipop_pba_binned = pd.read_csv(INPUT_BASEDATA_PATH.joinpath("remi_pop_l_NC3RC1.csv"))

    remipop_pba_binned["age_grp_5"] = pd.cut(
        remipop_pba_binned.Category.str.extract(r"(\d{1,3})", expand=False).astype(np.int32),
        bins=breaks_5,
        labels=agebreaker2(breaks_5),
        include_lowest=True,
        right=False,
    )

    remipop_pba_binned["rac_ethn"] = remipop_pba_binned.hierarchy.str.split("--").map(
        lambda x: x[0]
    )
    remipop_pba_binned["gender"] = remipop_pba_binned.hierarchy.str.split("--").map(lambda x: x[1])
    remipop_pba_binned["runid"] = "PBA_2040"
    remipop_pba_binned = remipop_pba_binned.rename(
        columns={"variable": "Year"}
    )  # DO WE NEED TO MAINTAIN RUNIDS?

    remipop_pba_binned = (
        remipop_pba_binned.groupby(["Year", "runid", "Region", "gender", "rac_ethn", "age_grp_5"])
        .value.sum()
        .unstack("Year")
        .round(0)
        .astype(np.int32)
    )
    return remipop_pba_binned[range(2011, 2041)]


# DOF 1980-2020
def import_dof_observed_hh():
    """Function imports DOF observed households data"""
    print("Importing DOF observed household counts 1990-2020")

    def dof_hh_prep(dofdata):
        dofdata["year"] = dofdata["year"].astype(str)
        dofdata["HR"] = ""
        dofdata["runid"] = "Observed"
        dofdata = dofdata.groupby(["HR", "runid", "county", "year"]).tothh.sum().unstack("year")
        return dofdata

    # 1980
    hh80 = {
        "Alameda": 426092,
        "Contra Costa": 241534,
        "Marin": 88723,
        "Napa": 36624,
        "San Francisco": 298956,
        "San Mateo": 225201,
        "Santa Clara": 458519,
        "Solano": 80426,
        "Sonoma": 114474,
    }

    # 1980
    hh80 = pd.Series(
        pd.Series(hh80).values,
        index=pd.MultiIndex.from_product(
            [pd.Series(hh80).index, [""], ["Observed"], ["1980"]],
            names=["county", "HR", "runid", "year"],
        ),
    ).reset_index(name="value")
    hh80 = hh80.groupby(["HR", "runid", "county", "year"]).value.sum()

    # 1990-2009
    hh90_09 = pd.read_csv(INPUT_BASEDATA_PATH.joinpath("dof_e5_1990_2017.csv")).rename(
        columns={"County": "county"}
    )

    hh90_09 = dof_hh_prep(hh90_09)
    hh90_09 = hh90_09.loc[:, :"2009"]

    # 2010-2020
    hh10_20 = pd.read_csv(
        INPUT_BASEDATA_PATH.joinpath("dof_e5_2010_2020.csv"),
        index_col=["year", "county", "variable", "is_bayarea"],
    ).value

    hh10_20 = hh10_20.loc[:, :, "Households", True].reset_index().rename(columns={"value": "tothh"})
    hh10_20 = dof_hh_prep(hh10_20)

    # shift january 1 to july 1 by taking half the diff from one year to the next and shifting the index back one year so it is still the original year
    hh80_20 = hh80.unstack("year").join(hh90_09).join(hh10_20)
    hh80_20 = (hh80_20 + (hh80_20.diff(axis=1) * 0.5).T.shift(-1).T.fillna(0)).iloc[:, 1:-1]
    hh80_20 = hh80.unstack("year").join(hh80_20)
    return hh80_20


def import_gq_shares():
    """Function import 2010 group quarter shares to apply to REMI population"""
    grp_qtr_shares = pd.read_csv(
        INPUT_BASEDATA_PATH.joinpath("census_2010_pums_grp_qtr_region_age_grp_5_binned_shares.csv"),
        index_col=None,
        names=["Region", "gender", "rac_ethn", "age_grp_5", "value"],
    )
    remiregions = {
        "East Bay": ["Alameda County", "Contra Costa County"],
        "West Bay": ["Marin County", "San Mateo County", "San Francisco County"],
        "North Bay": ["Napa County", "Solano County", "Sonoma County"],
        "South Bay": ["Santa Clara County"],
    }

    grp_qtr_shares["county"] = grp_qtr_shares["Region"].map(remiregions)
    grp_qtr_shares = grp_qtr_shares.explode("county").drop(columns="Region")
    grp_qtr_shares = grp_qtr_shares.sort_values(["county", "gender", "rac_ethn", "age_grp_5"])
    grp_qtr_shares.set_index(["county", "gender", "rac_ethn", "age_grp_5"], inplace=True)
    grp_qtr_shares = grp_qtr_shares.value.fillna(0)
    return grp_qtr_shares


###########
# REMI Data
###########

# TODO: The original function is set up to perform the operation in a loop for all counties (SEE quarregion_replicates).
# This function does one scenario at a time, can loop the function with different scenarios.


def import_remi_population(scenario_name):
    """Function prepares raw REMI inputs for headship model."""
    input_remi_path = INPUT_REMI_PATH.joinpath(scenario_name)
    print("Importing REMI data")
    remi = pd.read_excel(
        input_remi_path.joinpath(
            "REMI 3.0.0 Tables/1_REMI3.0.0_Population- By Ethnicity, Gender, and Age.xls"
        ),
        skiprows=2,
    )
    # remi_id = remi.iloc[0].to_dict()["Forecast"].strip()
    # print("\n", remi_id)
    remi.columns = remi.iloc[2, :].to_list()
    remi = remi.iloc[3:, :]
    remi = remi[remi["Region"] != "All Regions"]
    remi = remi[remi["Race"] != "All Races"]
    remi = remi[remi["Gender"] != "Total"]
    remi = remi[remi["Age"] != "All Ages (0-100)"]
    # remi["runid"] = remi_id
    remi = (
        remi.rename(columns={"Region": "county"})
        .set_index(["county", "Race", "Gender", "Age"])  # "runid",
        .filter(regex=r"\d{4}")
    )
    remi = (remi.stack() * 1000).round(0).astype(np.int32).reset_index(name="persons")
    remi = remi.rename(columns={"Race": "rac_ethn", "level_4": "year"})
    remi["year"] = remi["year"].astype("int32")
    remi["sex"] = remi.Gender.str.lower()
    remi["age_grp_5"] = pd.cut(
        remi.Age.str.extract(r"(\d{1,3})", expand=False).astype(np.int32),
        bins=breaks_5,
        labels=agebreaker2(breaks_5),
        include_lowest=True,
        right=False,
    )
    remi["gender"] = remi.Gender + "s"
    remipop_binned = (
        remi.groupby(["year", "county", "gender", "rac_ethn", "age_grp_5"])  # "runid",
        .persons.sum()
        .unstack("year")
    )
    return remipop_binned


# SECTION TITLED: PBA Convergence Adjustments #


def import_headship_rates():
    """Function imports headship rayes created in preprocess"""
    try:
        headship_alt = pd.read_csv("data/intermediate/HeadshipModel/headship_rates.csv")
    except FileNotFoundError:
        # TODO: it should probably just be able to run it from here
        return print("Run preprocess.py first")

    headship_alt.age_grp_5 = headship_alt.age_grp_5.astype(
        CategoricalDtype(categories=agebreaker2(breaks_5), ordered=True)
    )
    headship_alt["geovintage"] = headship_alt.region + " " + headship_alt.vintage
    headship_alt = headship_alt.set_index(["geovintage", "rac_ethn", "age_grp_5"]).value

    # Makes dictionary of different headship rates
    hsr = {}
    for rgn in headship_alt.index.get_level_values("geovintage").unique():
        hsr["HR %s" % rgn] = headship_alt.loc[rgn]

    # Also add PBA 2040 headship rates, as well as the special phased version, from ACS 14-18 to Census 2000 rates
    hsr["HR PBA 2040"] = pd.read_csv(
        INPUT_BASEDATA_PATH.joinpath("headship_rates_adj.csv"), index_col=["rac_ethn", "age_grp_5"]
    ).PWGTP
    hsr["HR Phased"] = (
        pd.read_csv("data/intermediate/HeadshipModel/headship_rates_eased.csv", dtype={"Year": str})
        .set_index(["Year", "rac_ethn", "age_grp_5"])
        .value
    )
    return hsr


# PREPARING EASING SOURCE SERIES

# create base series for easing function with modest growth assumption
# we ease in the difference between this series and the projected one
def prepare_source_series():
    hhs_observed = (
        import_dof_observed_hh()[map(str, range(2010, 2020))].groupby(level="county").sum()
    )
    envelope_year_start = 2019
    envelope_year_end = 2050
    base_growth_rate = 1.006

    source_series = {}
    for yr in range(envelope_year_start, envelope_year_end + 1):
        source_series[str(yr)] = hhs_observed[str(envelope_year_start)] * (
            base_growth_rate ** (yr - envelope_year_start)
        )

    source_series = pd.concat(source_series, names=["Year"]).unstack("county")
    source_series = pd.concat([hhs_observed.T, source_series.loc["2020":]])
    source_series.index = pd.Index(map(int, source_series.index), dtype=int, name="Year")
    return source_series


# BEGINNING ACTUAL PREDICTION CALCULATIONS


def run_headship_model_reg1(remi, hr):
    # TODO: DEBUGGGGG

    gqpop = (
        remi.stack().mul(import_gq_shares()).unstack("year").replace(np.nan, 0).filter(items=[2050])
    )

    # hhpop = remi.filter(items=[2050]).droplevel(0).stack().sub(gqpop.stack()).unstack()
    hhpop = remi.filter(items=[2050]).stack().sub(gqpop.stack()).unstack()

    import pdb

    pdb.set_trace()

    hhpop_2050 = hhpop.stack().groupby(["county", "rac_ethn", "age_grp_5", "year"]).sum()
    household_projections_age = (
        hr.mul(hhpop_2050)
        .groupby(["county", "rac_ethn", "age_grp_5", "year"])
        .sum()
        .unstack("year")
    )
    household_projections_county = (
        household_projections_age.replace(np.nan, 0).groupby("county").sum()
    )
    # Easing HH data does not matter, we only need 2050 level
    return hhpop_2050, household_projections_county


# Importing model results from MTC income category prediciton models:
cat1_params = {
    "Intercept": 0.9014822089887953,
    "msasize == 'Above 1 million'[T.True]": -0.023106570055203248,
    "cnty_to_us": -0.04162971542053355,
    "occ_det_officeadmin": -1.3479524747945162,
    "occ_det_mgmt": -2.659160871098275,
    "ag_65p": 0.7521756360762499,
    "Swhite_NH": -0.15241392360414863,
    "bin": "cat1",
}


cat2_params = {
    "Intercept": 0.26389977393621145,
    "division == '09'[T.True]": 0.020202255432665656,
    "SLF": 0.16508019244228725,
    "cnty_to_us": -0.020824244926714922,
    "occ_det_officeadmin": 0.1135919277447307,
    "occ_det_mgmt": -0.750691556572288,
    "occ_det_hlthsup": -0.4637412567281163,
    "Shispanic": 0.018512070484060103,
    "Swhite_NH": 0.0615574462408454,
    "ag_25_64": -0.03530171840802021,
    "bin": "cat2",
}


cat3_params = {
    "Intercept": -0.783748070053159,
    "msasize == 'Above 1 million'[T.True]": 0.006109103390653646,
    "np.log(per_capita_inc_adj2009)": 0.07871396861896023,
    "ind_prof": -0.23307918760124222,
    "SLF": 0.4937844365018745,
    "Sother_NH": 0.03306720386369877,
    "occ_det_foodprep": -1.63139476428312,
    "occ_det_hlthsup": -0.8389417061728535,
    "occ_det_biz": -0.9109243906623103,
    "ag_65p": 0.08549779967503035,
    "ag_25_64": 0.03233216840467573,
    "bin": "cat3",
}


cat4_params = {
    "Intercept": -1.908494936500041,
    "cnty_to_us": 0.02263000654481139,
    "occ_det_mgmt": 0.6555272237490307,
    "occ_det_community": -1.384391759437581,
    "np.log(per_capita_inc_adj2009)": 0.2171094628345671,
    "ag_25_64": -0.4523331394668578,
    "bin": "cat4",
}


def preprocess_soc_occ_codes():
    soc = pd.read_excel(
        INPUT_MAPPINGS_PATH.joinpath("soc_structure_2010.xls"),
        skiprows=11,
        names=["Major Group", "Minor Group", "Broad Group", "Detailed Occupation", "Description"],
    ).iloc[1:, :]

    def classifier(df):
        x = df.iloc[:4].tolist()
        out = next(s for s in x if not s is np.NaN)  # noqa : E714
        return out

    def classlevel(s):
        try:
            if s[3:] == "0000":
                return "major"
            elif np.float64(s[3:]) % 100 == 0:
                return "minor"
            elif np.float64(s[3:]) % 10 == 0:
                return "broad"
            else:
                return "detail"
        except:  # noqa : E722
            return "none"

    # soc['class']=soc.iloc[:,0:4].apply(classifier,axis=1)
    # soc["class"] = soc["class"].str[0:2]+"-0000"
    # soc['occup_grp_det']= soc["class"].map(occ_det_soc_map)
    soc["soc_2"] = soc["Major Group"]  # .fillna('').str.split('-').apply(lambda x: x[0])
    soc["class"] = soc.iloc[:, 0:4].apply(classifier, axis=1)
    soc["hierarchy"] = soc["class"].fillna("-1").map(classlevel)
    soc["Description"] = soc.Description.fillna("0").str.lower()
    soc[soc.Description.fillna("").str.contains("chitect")]
    soc = soc.append(
        pd.DataFrame(
            data={
                "Description": [
                    "baggage porters, bellhops, and concierges",
                    "baggage porters, bellhops, and concierges; tour and travel guides",
                    "military enlisted tactical operations and air/weapons specialists and crew members",
                    "first-line enlisted military supervisors",
                    "tour and travel guides",
                    "counselors, social workers, and other community and social service specialists",
                    "counselors and social workers",
                    "miscellaneous community and social service specialists",
                    "military officer special and tactical operations leaders",
                    "military specific occupations",
                ],
                "soc_2": [
                    "39-0000",
                    "39-0000",
                    "55-0000",
                    "55-0000",
                    "39-0000",
                    "21-0000",
                    "21-0000",
                    "21-0000",
                    "55-0000",
                    "55-0000",
                ],
                "class": [
                    "39-0000",
                    "39-0000",
                    "55-0000",
                    "55-0000",
                    "39-0000",
                    "21-0000",
                    "21-0000",
                    "21-0000",
                    "55-0000",
                    "55-0000",
                ],
                "hierarchy": ["minor"] * 10,
            }
        )
    )

    soc["soc_2"] = soc["soc_2"].fillna(method="ffill")
    soc = soc[soc["soc_2"].notnull()]
    soc["DescShort"] = soc.Description.str.slice(0, 8)
    soc["occ_census"] = soc.Description.str.lower().map(occ_to_census_upd)
    soc["occ_census"] = soc["soc_2"].map(
        soc[soc.hierarchy == "major"].set_index(["Major Group"]).occ_census.to_dict()
    )
    soc_minor = soc[soc.hierarchy == "minor"].set_index("Description")["class"]
    return soc, soc_minor


def make_all_remi_vars(scenario_name, year):
    # Race
    pop_data = import_remi_population(scenario_name)
    rv_Shispanic = (pop_data.loc[:, :, :, "Hispanic"].groupby("county").sum().loc[:, year]).div(
        pop_data.groupby("county").sum().loc[:, year]
    )
    rv_Sother_NH = (
        pop_data.loc[:, :, :, "Other-NonHispanic"].groupby("county").sum().loc[:, year]
    ).div(pop_data.groupby("county").sum().loc[:, year])
    rv_Swhite_NH = (
        pop_data.loc[:, :, :, "White-NonHispanic"].groupby("county").sum().loc[:, year]
    ).div(pop_data.groupby("county").sum().loc[:, year])
    rv_ag25_64 = (
        pop_data.loc[
            :,
            :,
            :,
            :,
            [
                "Ages 25-29",
                "Ages 30-34",
                "Ages 35-39",
                "Ages 40-44",
                "Ages 45-49",
                "Ages 50-54",
                "Ages 55-59",
                "Ages 60-64",
            ],
        ]
        .groupby("county")
        .sum()
        .loc[:, year]
    ).div(pop_data.groupby("county").sum().loc[:, year])
    rv_ag65p = (
        pop_data.loc[
            :, :, :, :, ["Ages 65-69", "Ages 70-74", "Ages 75-79", "Ages 80-84", "Ages 85+"]
        ]
        .groupby("county")
        .sum()
        .loc[:, year]
    ).div(pop_data.groupby("county").sum().loc[:, year])
    # Labor force
    lfpr = pd.read_excel(
        INPUT_REMI_PATH.joinpath(scenario_name, "2_Labor- Labor Force - By Gender.xls"),
        sheet_name="All",
        skiprows=5,
    )
    lfpr = lfpr[lfpr["Region"] != "All Regions"]
    lfpr = lfpr[lfpr["Gender"] == "Total"]
    lfpr = lfpr.rename(columns={"Region": "county"}).set_index("county")
    rv_S_lf = lfpr.loc[:, year].mul(1000).div(pop_data.groupby("county").sum().loc[:, year])

    def make_true_catvar(x):
        datafrm = pd.DataFrame(
            data={year: [x, x, x, x, x, x, x, x, x]},
            index=[
                "Alameda County",
                "Contra Costa County",
                "Marin County",
                "Napa County",
                "San Francisco County",
                "San Mateo County",
                "Santa Clara County",
                "Solano County",
                "Sonoma County",
            ],
        )
        datafrm.index.name = "county"
        return datafrm.stack().reset_index(level=1, drop=True)

    rv_is_division_9 = make_true_catvar(1)
    rv_msasize = make_true_catvar(1)

    # Income
    inc = pd.read_excel(
        INPUT_REMI_PATH.joinpath(scenario_name, "6_GDP and Income -Income- Income Profile.xls"),
        sheet_name="All",
        skiprows=5,
    )
    inc = inc[inc["Category"] == "Equals: Disposable personal income"]
    inc = (
        inc[inc["Region"] != "All Regions"].rename(columns={"Region": "county"}).set_index("county")
    )
    real_pers_inc_per_cap = (inc.loc[:, year] * 1e9).div(
        pop_data.groupby("county").sum().loc[:, year]
    )
    real_pers_inc_per_cap = np.log(real_pers_inc_per_cap)

    # Industry
    indust = pd.read_excel(
        INPUT_REMI_PATH.joinpath(
            scenario_name,
            "5_Employment- Employment by Occupation - Employment by Industry and Occupation.xls",
        ),
        sheet_name="All",
        skiprows=5,
    )
    indust = (
        indust[indust["Region"] != "All Regions"]
        .rename(columns={"Region": "county"})
        .set_index("county")
    )
    totjobs = indust[indust["Industry"] == "All Industries"].loc[:, year].groupby("county").sum()
    indust["cat"] = indust["Industry"].map(ind_prof_map)
    indust = indust[indust["cat"].notnull()].loc[:, year].groupby("county").sum()

    rv_S_ind_prof = indust.div(totjobs)

    # Occupations
    soc, soc_minor = preprocess_soc_occ_codes()
    occ = pd.read_excel(
        INPUT_REMI_PATH.joinpath(
            scenario_name,
            "4_Employment- Employment by Occupation -Occupations.xls",
        ),
        sheet_name="All",
        skiprows=5,
    )

    occ = (
        occ[occ["Region"] != "All Regions"].rename(columns={"Region": "county"}).set_index("county")
    )
    occ = occ[occ["Occupation"] != "All Occupations"]
    occ["soc_minor"] = (
        occ.Occupation.str.lower().str.strip().map(soc_minor.groupby(level=0).first())
    )
    occ["soc_major"] = occ.soc_minor.str.slice(0, 2) + "-0000"
    occ["description"] = occ.Occupation.str.strip().str.lower()
    occ["occup_grp"] = occ.soc_major.map(soc.groupby("soc_2").occ_census.first())
    occ["occup_grp_det"] = occ.soc_major.map(occ_det_soc_map)
    occ["occup_grp_det"] = occ["occup_grp_det"].fillna("Other Occupations")
    empoccup_det_pct = (
        occ.loc[:, ["occup_grp_det", year]].groupby(["county", "occup_grp_det"]).sum().stack()
    )
    empoccup_det_pct = empoccup_det_pct / (empoccup_det_pct.groupby(["county"]).sum())

    # empoccup_pct = occ.loc[:, ["occup_grp", year]].groupby(["county", "occup_grp"]).sum().stack()
    # empoccup_pct = empoccup_pct / (empoccup_pct.groupby(["county"]).sum())

    rv_S_occ_det_biz = empoccup_det_pct.loc[:, ["occ_det_biz"]].reset_index([1, 2], drop=True)
    rv_S_occ_det_community = empoccup_det_pct.loc[:, ["occ_det_community"]].reset_index(
        [1, 2], drop=True
    )
    rv_S_occ_det_foodprep = empoccup_det_pct.loc[:, ["occ_det_foodprep"]].reset_index(
        [1, 2], drop=True
    )
    rv_S_occ_det_hlthsup = empoccup_det_pct.loc[:, ["occ_det_hlthsup"]].reset_index(
        [1, 2], drop=True
    )
    rv_S_occ_det_mgmt = empoccup_det_pct.loc[:, ["occ_det_mgmt"]].reset_index([1, 2], drop=True)
    rv_S_occ_det_officeadmin = empoccup_det_pct.loc[:, ["occ_det_officeadmin"]].reset_index(
        [1, 2], drop=True
    )

    # PLCAEHOLDERS
    rv_cnty_to_us = make_true_catvar(1.6)

    param_to_var = {
        "SLF": rv_S_lf,
        "Shispanic": rv_Shispanic,
        "Sother_NH": rv_Sother_NH,
        "Swhite_NH": rv_Swhite_NH,
        "ag_25_64": rv_ag25_64,
        "ag_65p": rv_ag65p,
        "cnty_to_us": rv_cnty_to_us,
        "division == '09'[T.True]": rv_is_division_9,
        "ind_prof": rv_S_ind_prof,
        "msasize == 'Above 1 million'[T.True]": rv_msasize,
        "np.log(per_capita_inc_adj2009)": real_pers_inc_per_cap,
        "occ_det_biz": rv_S_occ_det_biz,
        "occ_det_community": rv_S_occ_det_community,
        "occ_det_foodprep": rv_S_occ_det_foodprep,
        "occ_det_hlthsup": rv_S_occ_det_hlthsup,
        "occ_det_mgmt": rv_S_occ_det_mgmt,
        "occ_det_officeadmin": rv_S_occ_det_officeadmin,
    }
    return param_to_var


def run_inc_cat_predictions(param_to_var):
    # we have variables for each submodel in a list. Loop through them and predict shares using models
    var_x_coef_submodels = {}
    for i, prms in enumerate([cat1_params, cat2_params, cat3_params, cat4_params]):
        var_x_coef = {}
        for p, v in prms.items():
            if p not in ["bin", "Intercept", "np.log(aland)"]:
                if "occ_det_" in p:
                    print(p, param_to_var[p], "\n\n")
                # fetch the corresponding series to the actually estimated model
                param_times_data = param_to_var[p] * v
                var_x_coef[p] = param_times_data
        var_x_coef_submodels[i] = pd.concat(var_x_coef).sum(level=1) + prms["Intercept"]
    hhincproj = pd.concat(var_x_coef_submodels)
    hhincproj.index = hhincproj.index.set_names("bin", level=0)
    hhincproj = hhincproj / (hhincproj.swaplevel(0, 1).groupby("county").sum())
    return hhincproj


def create_remi_job_counts(scenario_name):
    indust = pd.read_excel(
        INPUT_REMI_PATH.joinpath(
            scenario_name,
            "5_Employment- Employment by Occupation - Employment by Industry and Occupation.xls",
        ),
        sheet_name="All",
        skiprows=5,
    )
    indust = (
        indust[indust["Region"] != "All Regions"]
        .rename(columns={"Region": "county"})
        .set_index("county")
    )
    indust = indust[indust["Industry"] != "All Industries"]
    indust = indust.filter(items=["Industry", 2050]).set_index("Industry", append=True)
    remi_occ_maps = pd.read_csv(
        "~/projects/24182-bart-link21-modeling/data/inputs/HeadshipModel/NAICS_Lookup.csv"
    )
    output = (
        indust.reset_index(0)
        .merge(remi_occ_maps, on="Industry")
        .set_index("county")
        .filter(items=[2050, "TM1.5_codes", "Link21_codes", "office_space"])
    )
    output[2050] = output[2050].mul(1000)

    return output


def create_income_indexs(scenario_name):
    remi_vars_base = make_all_remi_vars(scenario_name, 2021)
    remi_vars_horizon = make_all_remi_vars(scenario_name, 2050)
    inc_cat_base = run_inc_cat_predictions(remi_vars_base)
    inc_cat_horizon = run_inc_cat_predictions(remi_vars_horizon)
    index_multiplier = inc_cat_horizon.div(inc_cat_base)
    return index_multiplier


def writeout_remi_summary(scenario_name):
    """Function writes summary of pop and hh to dropbox scenario file"""
    remi_persons = import_remi_population(scenario_name)
    hr_2050 = import_headship_rates()["HR Phased"].loc["2050"]
    hhpop, hhtot = run_headship_model_reg1(remi_persons, hr_2050)
    hhpop = hhpop.groupby("county").sum()
    output = pd.concat([hhpop, hhtot], axis=1).rename(columns={0: "HHPOP", 2050: "TOTHH"})
    output.to_csv(INPUT_REMI_PATH.joinpath(scenario_name, "pop_summary.csv"))
    return print("Summary writen out to DropBox scenerio file")


if __name__ == "__main__":
    remi_persons = import_remi_population("2023-02-02 Outputs")
    dof_hh = import_dof_observed_hh()
    hr_2050 = import_headship_rates()["HR Phased"].loc["2050"]
    scenario_hh = run_headship_model_reg1(remi_persons, hr_2050)
