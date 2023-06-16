import os

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from mappings import (
    bayarearegions,
    indus_to_census,
    occ_det_soc_map,
    occ_to_census_upd,
    qtrmap,
    yr_to_vintage,
)
from utils import (
    adjust_wharton_variable,
    agebreaker,
    agebreaker2,
    agebreaker_mtc,
    classifier,
    classlevel,
)

# from pandas.api.types import CategoricalDtype


# from datetime import datetime

# from pandas.api.types import CategoricalDtype
# import pylab as P


# import matplotlib.cm as cm
# import matplotlib.font_manager as fm
#
# import glob
# import fnmatch

# import patsy
# import collections
# import re
#
# import logging
#
# from matplotlib.backends.backend_pdf import PdfPages

#
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm, matplotlib.font_manager as fm

YEARS_OUT_1 = list(range(2010, 2051, 1))
YEARS_OUT_1_2015 = list(range(2015, 2051, 1))
YEARS_OUT_5 = list(range(2010, 2051, 5))
YEARS_OUT_5_2015 = list(range(2015, 2051, 5))

# set constant for whether to use income shares from PUMS data

USE_PUMS_SHARES = True
USE_HIST_YEARS = True

age_coarse = [0, 5, 15, 25, 65, np.inf]
age_mtc = [0, 5, 20, 45, 65, np.inf]


def pct(x):
    return x / x.sum()


breaks_5 = list(range(0, 86, 5)) + [np.inf]
diffbreaks_custom = [0, 5] + list(range(20, 90, 5)) + [np.inf]
age_coarse = [0, 5, 15, 25, 65, np.inf]
age_mtc = [0, 5, 20, 45, 65, np.inf]


INPUT_REMI_PATH = "~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Background/MTC REMI Data/MTC Shared Folder/Household Forecast/REMI_raw_output"
INPUT_BASEDATA_PATH = "~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Background/MTC REMI Data/MTC Shared Folder/Household Forecast/base_data"
INPUT_MAPPINGS_PATH = "~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Background/MTC REMI Data/MTC Shared Folder/Household Forecast/mappings"
OUTPUT_PATH = "data/intermediate/HeadshipModel"

soc = pd.read_excel(
    os.path.join(INPUT_MAPPINGS_PATH, "soc_structure_2010.xls"),
    skiprows=11,
    names=["Major Group", "Minor Group", "Broad Group", "Detailed Occupation", "Description"],
)
soc["soc_2"] = soc["Major Group"]  # .fillna('').str.split('-').apply(lambda x: x[0])

soc["class"] = soc.iloc[:, :4].apply(classifier, axis=1)
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

soc_minor = soc[soc.hierarchy == "minor"].set_index("Description")["class"]


soc["occ_census"] = soc.Description.str.lower().map(occ_to_census_upd)
soc["occ_census"] = soc["soc_2"].map(
    soc[soc.hierarchy == "major"].set_index(["Major Group"]).occ_census.to_dict()
)


USE_ADJUSTED_Q1 = False
USE_ADJUSTED_Q1 = True


dataformodels = pd.read_csv(os.path.join(INPUT_BASEDATA_PATH, "censusdata.csv"))

if USE_ADJUSTED_Q1:
    dataformodels = pd.read_csv(os.path.join(INPUT_BASEDATA_PATH, "allyears_feb2020upd.csv"))

# dataformodels=pd.read_csv(os.path.join(box,'EDF Shared work/control_totals/base_data/allyears_feb2020upd.csv'))
dataformodels.geoid10 = dataformodels.geoid10.map(lambda x: "{:05d}".format(x))
dataformodels.state = dataformodels.state.map(lambda x: "{:02d}".format(x))
dataformodels.division = dataformodels.division.map(lambda x: "{:02d}".format(x))
dataformodels = dataformodels.set_index(["vintage", "geoid10"])


dataformodels_occdet = pd.read_csv(os.path.join(INPUT_BASEDATA_PATH, "censusdata_occvars.csv"))
dataformodels_occdet.geoid10 = dataformodels_occdet.geoid10.map(lambda x: "{:05d}".format(x))
dataformodels_occdet = dataformodels_occdet.set_index(["vintage", "geoid10"])


# add extra occupation detail
dataformodels = dataformodels.join(dataformodels_occdet)


# load msa classifiers for each census vintage

msaclass = pd.read_csv(
    os.path.join(INPUT_BASEDATA_PATH, "msamappings_1990_2013.csv"), dtype=object, index_col=0
)

msaclass["geoid"] = msaclass.geoid.apply(lambda x: "%05d" % int(x))
msaclass = msaclass.drop_duplicates(["geoid", "vintage"])

msaclass = msaclass.set_index(["vintage", "geoid"]).metroconcept
msaclass.index = msaclass.index.set_names(dataformodels.index.names)

# assign msa code back to county data
dataformodels["MSA_code"] = msaclass


# 'http://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_metro.txt'
# get FHFA housing price indices at metro level

metro = pd.read_csv(
    os.path.join(INPUT_BASEDATA_PATH, "HPI_PO_metro.txt"), sep="\t"
)  # names=['cbsa','cbsa_id','year','qtr','index'])
metro_for_stacking = metro[["metro_name", "cbsa", "yr", "qtr", "index_nsa"]].rename(
    columns={"metro_name": "geo", "yr": "year"}
)
metro_for_stacking["cbsa"] = metro_for_stacking.cbsa.apply(lambda x: "%05d" % int(x))
metro_for_stacking["qtrmo"] = metro_for_stacking.qtr.map(qtrmap)
msaclass2 = msaclass["ACS2010"].copy()

metro_for_stacking["date"] = metro_for_stacking.apply(
    lambda x: pd.to_datetime("%s-%s-%s" % (x["qtrmo"], "01", x.year)), axis=1
)
metro_for_stacking["vintage"] = metro_for_stacking.year.map(yr_to_vintage)
metro_for_stacking_m = metro_for_stacking.merge(
    msaclass2.reset_index(), left_on="cbsa", right_on="metroconcept", how="inner"
).rename(columns={"geoid": "geoid10"})
metro_for_stacking_m = metro_for_stacking_m.loc[
    (metro_for_stacking_m.vintage.notnull()) & (metro_for_stacking_m.qtr == 1)
].set_index(
    ["vintage", "geoid10"]
)  # .unstack(3)

dataformodels["fhfa"] = metro_for_stacking_m.index_nsa


vintage_MSA_sizes = dataformodels.reset_index().groupby(["vintage", "MSA_code"]).population.sum()
vintage_top50_msas = vintage_MSA_sizes.groupby(level="vintage", group_keys=False).nlargest(50)
vintage_top30_msas = vintage_MSA_sizes.groupby(level="vintage", group_keys=False).nlargest(30)
vintage_top70_msas = vintage_MSA_sizes.groupby(level="vintage", group_keys=False).nlargest(70)

dataformodels_top50msas = (
    dataformodels.reset_index()
    .merge(vintage_top50_msas, on=["vintage", "MSA_code"], how="inner")
    .set_index(["vintage", "geoid10"])
)
dataformodels_top70msas = (
    dataformodels.reset_index()
    .merge(vintage_top70_msas, on=["vintage", "MSA_code"], how="inner")
    .set_index(["vintage", "geoid10"])
)
dataformodels_top30msas = (
    dataformodels.reset_index()
    .merge(vintage_top30_msas, on=["vintage", "MSA_code"], how="inner")
    .set_index(["vintage", "geoid10"])
)


depvar = {1: "inc_cat_1_pums", 2: "inc_cat_2_pums", 3: "inc_cat_3_pums", 4: "inc_cat_4_pums"}

mtc_pba2013 = pd.read_csv(
    os.path.join(INPUT_BASEDATA_PATH, "mtc2010_2040_inccats.csv"), index_col=0
)


# observed starting distribution pct

if USE_ADJUSTED_Q1:

    incshares_pums_recent = pd.read_excel(
        os.path.join(INPUT_BASEDATA_PATH, "pums_remireg_income_shares_2006_2018_pct_q1fix.xlsx"),
        dtype={"STCOUNTY": str},
    ).rename(columns={"YEAR": "vintage"})
else:
    incshares_pums_recent = pd.read_excel(
        os.path.join(INPUT_BASEDATA_PATH, "pums_remireg_income_shares_2006_2018_pct.xlsx"),
        dtype={"STCOUNTY": str},
    ).rename(columns={"YEAR": "vintage"})

incshares_pums_recent.vintage = incshares_pums_recent.vintage.map(yr_to_vintage)

bayareashares_pums_2015 = (
    incshares_pums_recent[incshares_pums_recent.vintage == "ACS2015"]
    .set_index(["remi_region"])
    .filter(regex=r"inc_cat")
    .rename(columns=lambda x: x + "_tot")
)


# observed starting distribution absolute

if USE_ADJUSTED_Q1:
    incshares_pums_recent_abs = pd.read_excel(
        os.path.join(INPUT_BASEDATA_PATH, "pums_remireg_income_shares_2006_2018_abs_q1fix.xlsx"),
        dtype={"STCOUNTY": str},
    ).rename(columns={"YEAR": "vintage"})
else:
    incshares_pums_recent_abs = pd.read_excel(
        os.path.join(INPUT_BASEDATA_PATH, "pums_remireg_income_shares_2006_2018_abs.xlsx"),
        dtype={"STCOUNTY": str},
    ).rename(columns={"YEAR": "vintage"})


incshares_pums_recent_abs.vintage = incshares_pums_recent_abs.vintage.map(yr_to_vintage)

bayareainc_pums_2015 = (
    incshares_pums_recent_abs[incshares_pums_recent_abs.vintage == "ACS2015"]
    .set_index(["remi_region"])
    .filter(regex=r"inc_cat")
    .rename(columns=lambda x: x + "_tot")
)


bayareaincomeobs = (
    incshares_pums_recent_abs[incshares_pums_recent_abs.vintage.isin(["ACS2015", "ACS2010"])]
    .set_index("vintage")
    .groupby(lambda x: int(x[-4:]))
    .sum()
    .rename(columns=lambda x: x.replace("inc_cat", "bin"))
)

# scale to decennial household counts

bayareaincomeobs.index = bayareaincomeobs.index.set_names(["YEAR"])


# Load Wharton index from Gyourko. Idea is that more restrictive areas might all other things equal be more top
# heavy in the income distribution

# We made a household-weighted version when aggregating jurisdictions to county
# http://localhost:8888/notebooks/analysis/topical/get%20census%20API%20place%20data%20national.ipynb

wharton_combo = pd.read_excel(
    os.path.join(INPUT_BASEDATA_PATH, "WRLURI_weighted_and_orig.xlsx"), dtype={"geoid10": str}
)
WRLURI_wgt = wharton_combo.set_index("geoid10").weighted
WRLURI_wgt.name = "WRLURI"

# create a frame with the index literally repeated for each census vintage.
# The update method doesn't seem to broadcast without it

WRLURI_expanded = {}
WRLURI_wgt.loc["06001"]
for vint in dataformodels.index.get_level_values(0).unique():
    WRLURI_expanded[vint] = WRLURI_wgt
WRLURI_expanded = pd.concat(WRLURI_expanded, names=["vintage"])

# assign to dataformodels dataframe
dataformodels.update(WRLURI_expanded)

combo_remireg = pd.read_excel(
    os.path.join(INPUT_BASEDATA_PATH, "WRLURI_bayarea_weighted_and_orig.xlsx")
)
combo_remireg

pd.options.display.float_format = "{:,.2f}".format

# create wharton index for *each* year
years = range(2001, 2051) if USE_HIST_YEARS else range(2015, 2051)

rv_wharton = {}
for yr in years:
    rv_wharton[yr] = combo_remireg.set_index("Region").weighted
rv_wharton_geo = pd.concat(rv_wharton, names=["year"]).unstack(0)


# read Mike's occ mapping file, notfully enumerated to detailed occupations.
# Note that SOC 29 is split: Doctors are in "professional", the remainder in "services"
mike_occ_codes = pd.read_csv(
    os.path.join(INPUT_MAPPINGS_PATH, "emp_occ_codes.csv"), index_col=["Variable", "Description"]
).SOC_codes
mike_occ_codes = (
    mike_occ_codes.str.split(",").apply(pd.Series).stack().reset_index(2, drop=True).str.strip()
)


# Start preping future RHS data / variables

# Loading REMI data
temp = pd.read_excel(
    os.path.join(INPUT_REMI_PATH, "population/By Ethnicity, Gender, and Age.xlsx"), skiprows=2
)
hdr = temp.iloc[0].to_dict()
diffbreaks_dave = [0, 15, 25, 65, np.inf]

print(hdr["Forecast"].strip())

outdata = {}
temp = pd.read_excel(
    os.path.join(INPUT_REMI_PATH, "population/By Ethnicity, Gender, and Age.xlsx"), skiprows=5
)
outdata[hdr["Forecast"].strip()] = temp.set_index(["Region", "Race", "Gender", "Ages"]).filter(
    regex=r"\d{4}"
)
outdata = pd.concat(outdata)

outdata.index = outdata.index.set_names(["runid"], level=[0])

remipop = (outdata.stack() * 1000).round(0).astype(np.int32).reset_index(name="value")

remipop = remipop.rename(columns={"Race": "rac_ethn", "level_5": "Year"})
remipop["sex"] = remipop.Gender.str.lower()
remipop["age_fine"] = pd.cut(
    remipop.Ages.str.extract(r"(\d{1,3})", expand=False).astype(int),
    right=False,
    bins=diffbreaks_custom,
    labels=agebreaker(diffbreaks_custom),
).astype(str)
remipop["age_grp_5"] = pd.cut(
    remipop.Ages.str.extract(r"(\d{1,3})", expand=False).astype(np.int32),
    bins=breaks_5,
    labels=agebreaker2(breaks_5),
    include_lowest=True,
    right=False,
)
remipop["age_grp_coarse"] = pd.cut(
    remipop.Ages.str.extract(r"(\d{1,3})", expand=False).astype(np.int32),
    bins=age_coarse,
    labels=agebreaker2(age_coarse),
    include_lowest=True,
    right=False,
)

remipop["age_grp_mtc"] = pd.cut(
    remipop.Ages.str.extract(r"(\d{1,3})", expand=False).astype(np.int32),
    bins=age_mtc,
    labels=agebreaker_mtc(age_mtc),
    include_lowest=True,
    right=False,
)

remipop["gender"] = remipop.Gender + "s"


futurepop = remipop.groupby(["runid", "Region", "Year"]).value.sum().unstack("Year")


# create core demographic frames from which to extract specific series

remi_age = remipop.groupby(["runid", "Region", "Year", "age_grp_coarse"]).value.sum()
remi_race = remipop.groupby(["runid", "Region", "Year", "rac_ethn"]).value.sum()


# subset to relevant race group and turn to percent

rv_Swhite_NH = (
    remi_race.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "White-NonHispanic"]
    .unstack("Year")
)
rv_Sblack_NH = (
    remi_race.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Black-NonHispanic"]
    .unstack("Year")
)
rv_Sother_NH = (
    remi_race.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Other-NonHispanic"]
    .unstack("Year")
)
rv_Shispanic = (
    remi_race.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Hispanic"]
    .unstack("Year")
)

# age specific series
rv_ag15_24 = (
    remi_age.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Ages 15-24"]
    .unstack("Year")
)
rv_ag25_64 = (
    remi_age.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Ages 25-64"]
    .unstack("Year")
)
rv_ag65p = (
    remi_age.groupby(level=["runid", "Region", "Year"])
    .apply(pct)
    .loc[:, :, :, "Ages 65+"]
    .unstack("Year")
)


# Labor force by sex
outlfpr = {}
tmplfpr = pd.read_excel(
    os.path.join(INPUT_REMI_PATH, "labor/lfpr_gender.xlsx"), sheet_name="All", skiprows=5
)

outlfpr["R6H2C_COVID_cross_rev2"] = tmplfpr

outlfpr = pd.concat(outlfpr, names=["runid"]).reset_index(1, drop=True).reset_index()
rv_S_lf_female = (
    outlfpr[outlfpr.Gender == "Female"].set_index(["runid", "Region"]).filter(regex=r"\d{4}")
)

rv_S_lf = outlfpr[outlfpr.Gender == "Total"].set_index(["runid", "Region"]).filter(regex=r"\d{4}")

# Labor force total
outlf = {}

tmplf = pd.read_excel(os.path.join(INPUT_REMI_PATH, "labortot/hhinc 2 labor_age.xlsx"), skiprows=5)
outlf["R6H2C_COVID_cross_rev2"] = tmplf

outlf = pd.concat(outlf, names=["runid"]).reset_index(1, drop=True).reset_index()

outlf_df = (
    outlf.set_index(["runid", "Region", "Ages", "Race", "Gender"]).filter(regex=r"\d{4}").stack()
    * 1000
).reset_index(name="value")
outlf_df = outlf_df.loc[
    outlf_df.Region.isin(["West Bay", "East Bay", "North Bay", "South Bay"])
].rename(columns={"level_5": "year"})

# Occupation

outocc = []
outindus = []

tmpocc = pd.read_excel(
    os.path.join(INPUT_REMI_PATH, "employment/hhinc 4 emp occ indus.xlsx"),
    sheet_name="Occupations",
    skiprows=4,
)
tmpind = pd.read_excel(
    os.path.join(INPUT_REMI_PATH, "employment/hhinc 4 emp occ indus.xlsx"),
    sheet_name="Employment by Industry",
    skiprows=4,
)

outocc.append(tmpocc[tmpocc.Forecast == "R6H2C_COVID_cross_rev2"])
outindus.append(tmpind[tmpind.Forecast == "R6H2C_COVID_cross_rev2"])

outocc = pd.concat(outocc)
outindus = pd.concat(outindus)


outocc.loc[~outocc.Occupations.str.match(r"\s+"), "major_group"] = outocc.loc[
    ~outocc.Occupations.str.match(r"\s+")
].Occupations
outocc["soc_minor"] = (
    outocc.Occupations.str.lower().str.strip().map(soc_minor.groupby(level=0).first())
)
outocc["soc_2"] = outocc["soc_major"] = outocc.soc_minor.str.slice(0, 2) + "-0000"
# outocc['county_name']=outocc.Region
# the detail occupations have whitespace in front. We use that for classification purposes
outocc["description"] = outocc.Occupations.str.strip().str.lower()
outocc.loc[outocc.Occupations.str.match(r"\s+"), "detail"] = outocc.loc[
    outocc.Occupations.str.match(r"\s+")
].Occupations.str.strip()

# drop major groups (which are the REMI headings, but different than soc-2). Drop except military
# as all other have components. Keep military as it has no children.

outocc = outocc.loc[(outocc.major_group.isnull()) | (outocc.major_group == "Military")]

outocc["occup_grp"] = outocc.soc_2.map(soc.groupby("soc_2").occ_census.first())
outocc["occup_grp_det"] = outocc.soc_2.map(occ_det_soc_map)
outocc["occup_grp_det"] = outocc["occup_grp_det"].fillna("Other Occupations")

outocc.major_group = outocc.major_group.fillna(method="ffill")


empoccup_pct = (
    outocc.groupby(["Forecast", "Region", "occup_grp"])
    .sum()
    .stack()
    .groupby(level=[0, 1, 3])
    .apply(pct)
    .unstack(3)
)
empoccup_det_pct = (
    outocc.groupby(["Forecast", "Region", "occup_grp_det"])
    .sum()
    .stack()
    .groupby(level=[0, 1, 3])
    .apply(pct)
    .unstack(3)
)


# Load industry employment by scenario, sub-region

outindus["naics"] = outindus.Industries.str.extract(r"\(([0-9_\-,]+)\)")
outindus["industry_name"] = (
    outindus.Industries.str.split("(").apply(lambda x: x[0]).str.strip().str.title()
)
# keep only records with 2-digit detail
outindus = outindus.loc[outindus.naics.fillna("").str.contains(r"^\d{2}$|^\d{2}-\d{2}$")]
outindus["indus_grp"] = outindus.industry_name.map(
    dict(zip(map(lambda x: x.title(), indus_to_census.keys()), indus_to_census.values()))
)

empindus_pct = (
    outindus.groupby(["Forecast", "Region", "indus_grp"])
    .sum()
    .stack()
    .groupby(level=[0, 1, 3])
    .apply(pct)
    .unstack(3)
)


rv_S_ind_educ = empindus_pct.loc(0)[:, :, "ind_educ"].reset_index(2, drop=True)

rv_S_ind_retail = empindus_pct.loc(0)[:, :, "ind_retail"].reset_index(2, drop=True)
# rv_S_ind_public = empindus_pct.loc(0)[:,:,'ind_public']
rv_S_ind_accom_food_svcs = empindus_pct.loc(0)[:, :, "ind_accom_food_svcs"].reset_index(
    2, drop=True
)
rv_S_ind_health = empindus_pct.loc(0)[:, :, "ind_health"].reset_index(2, drop=True)
rv_S_ind_prof = empindus_pct.loc(0)[:, :, "ind_prof"].reset_index(2, drop=True)

rv_S_occ_mgmt = empoccup_pct.loc(0)[:, :, "occ_mgmt"].reset_index(2, drop=True)
rv_S_occ_nat = empoccup_pct.loc(0)[:, :, "occ_nat"].reset_index(2, drop=True)
rv_S_occ_svcs = empoccup_pct.loc(0)[:, :, "occ_svcs"].reset_index(2, drop=True)
rv_S_occ_prod = empoccup_pct.loc(0)[:, :, "occ_prod"].reset_index(2, drop=True)
rv_S_occ_sls = empoccup_pct.loc(0)[:, :, "occ_sls"].reset_index(2, drop=True)


# get slightly more detailed occupation shares

rv_S_occ_det_mgmt = empoccup_det_pct.loc(0)[:, :, "occ_det_mgmt"].reset_index(2, drop=True)
rv_S_occ_det_biz = empoccup_det_pct.loc(0)[:, :, "occ_det_biz"].reset_index(2, drop=True)
rv_S_occ_det_comp = empoccup_det_pct.loc(0)[:, :, "occ_det_comp"].reset_index(2, drop=True)
rv_S_occ_det_community = empoccup_det_pct.loc(0)[:, :, "occ_det_community"].reset_index(
    2, drop=True
)
rv_S_occ_det_lgl = empoccup_det_pct.loc(0)[:, :, "occ_det_lgl"].reset_index(2, drop=True)
rv_S_occ_det_hlthsup = empoccup_det_pct.loc(0)[:, :, "occ_det_hlthsup"].reset_index(2, drop=True)
rv_S_occ_det_foodprep = empoccup_det_pct.loc(0)[:, :, "occ_det_foodprep"].reset_index(2, drop=True)
rv_S_occ_det_officeadmin = empoccup_det_pct.loc(0)[:, :, "occ_det_officeadmin"].reset_index(
    2, drop=True
)


outincome = []
tmp = pd.read_excel(os.path.join(INPUT_REMI_PATH, "income/misc income.xlsx"), skiprows=5)
outincome.append(tmp[tmp.Forecast == "R6H2C_COVID_cross_rev2"])
outincome = (
    pd.concat(outincome)
    .set_index(["Forecast", "Region", "Category", "Units"])
    .filter(regex=r"\d{4}")
)

outincome.loc(0)[:, :, :, "Thousands"] *= 1000
outincome.loc(0)[:, :, :, "Billions of Fixed (2009) Dollars"] *= 1e9

# Extract wage, income Series
rh_agg_wage_income = outincome.loc(0)[:, :, "Wages and Salaries"].reset_index(
    level=["Category", "Units"], drop=True
)

rv_cnty_to_us = outincome.loc(0)[:, :, "Relative Housing Price"].reset_index(
    level=["Category", "Units"], drop=True
)

real_pers_inc = outincome.loc(0)[:, :, "Personal Income"].reset_index(
    level=["Category", "Units"], drop=True
)
real_pers_inc.head(2)

real_pers_inc_per_cap = outincome.loc(0)[:, :, "Personal Income"].reset_index(
    level=["Category", "Units"], drop=True
) / outincome.loc(0)[:, :, "Total Population"].reset_index(level=["Category", "Units"], drop=True)


# Wharton rates adjustment converting to variabel
regionrates = {"West Bay": 0.995, "South Bay": 1, "East Bay": 1, "North Bay": 1.005}

rv_wharton_geo_adj = adjust_wharton_variable(rv_wharton_geo, regionrates)


# Regions dummies (can be reviewed later)
length = years.__len__()
regions = ["West Bay", "East Bay", "North Bay", "South Bay", "Rest of California"]

mock_geodummies = pd.DataFrame(
    np.zeros([len(regions), len(years)], dtype="int"), index=regions, columns=years
)
mock_geodummies.index.name = "Region"

rv_is_north_bay = mock_geodummies.copy()
rv_is_north_bay.loc["North Bay"] = 1

rv_is_south_bay = mock_geodummies.copy()
rv_is_south_bay.loc["South Bay"] = 1

rv_is_east_bay = mock_geodummies.copy()
rv_is_east_bay.loc["East Bay"] = 1

rv_is_west_bay = mock_geodummies.copy()
rv_is_west_bay.loc["West Bay"] = 1

rv_is_rest_ca = mock_geodummies.copy()
rv_is_rest_ca.loc["Rest of California"] = 1

rv_msasize = rv_is_bay_area = mock_geodummies.copy().replace(0, 1)
yearly = pd.DataFrame(
    [years, years, years, years, years],
    columns=years,
    index=["West Bay", "East Bay", "North Bay", "South Bay", "Rest of California"],
)
yearly.index.name = "Region"

# census division 9 is the pacific states
rv_is_division_9 = pd.DataFrame(
    [
        np.repeat(1, length),
        np.repeat(1, length),
        np.repeat(1, length),
        np.repeat(1, length),
        np.repeat(1, length),
    ],
    columns=years,
    index=["West Bay", "East Bay", "North Bay", "South Bay", "Rest of California"],
)
rv_is_division_9.index.name = "Region"

rv_is_not_division_9 = pd.DataFrame(
    [
        np.repeat(0, length),
        np.repeat(0, length),
        np.repeat(0, length),
        np.repeat(0, length),
        np.repeat(0, length),
    ],
    columns=years,
    index=["West Bay", "East Bay", "North Bay", "South Bay", "Rest of California"],
)
rv_is_not_division_9.index.name = "Region"


# other misc variables

yrseries = pd.Series(years)
county_area = pd.read_csv(
    os.path.join(INPUT_BASEDATA_PATH, "county_area.csv"), dtype={"geoid": object, "geoid.1": object}
).set_index("geoid")
aland = county_area.aland

rv_density = county_area.loc[bayarearegions.keys()].rename(
    columns={"geoid.1": "geoid"}
)  # .map(bayarearegions).reset_index(drop=True)
rv_density["Region"] = rv_density.geoid.map(bayarearegions)
rv_density["population"] = dataformodels["population"].xs("ACS2013")
rv_density = rv_density.groupby(["Region"]).apply(
    lambda x: x["population"].sum() / x["aland"].sum()
)
rv_density = pd.concat(
    [
        pd.DataFrame(
            data={
                "density": rv_density.repeat(years.__len__()),
                "variable": pd.concat([yrseries, yrseries, yrseries, yrseries]).values,
            }
        )
        .set_index("variable", append=True)
        .unstack(1)
        .density,
        pd.DataFrame(rv_is_not_division_9.loc["Rest of California", :]).T,
    ]
)
rv_density.index.name = "Region"
rv_density.head(2)


#
#     READY FOR OLS


# Use a dict to keep track of which regression source data is in memory
# Should be refactored to include runyears, input paths, whether to use PUMA income shares, etc

runstate = {}


def statswrapper(modeloutput):
    statistics = pd.Series({"r2": modeloutput.rsquared, "adj_r2": modeloutput.rsquared_adj})
    # put them togher with the result for each term
    result_df = pd.DataFrame(
        {
            "params": modeloutput.params,
            "pvals": modeloutput.pvalues,
            "std": modeloutput.bse,
            "test_stats": statistics,
        }
    )
    # add the complexive results for f-value and the total p-value
    fisher_df = pd.DataFrame(
        {"params": {"_f_test": modeloutput.fvalue}, "pvals": {"_f_test": modeloutput.f_pvalue}}
    )
    # merge them and unstack to obtain a hierarchically indexed series
    res_series = pd.concat([result_df, fisher_df]).sort_values("test_stats")
    res_series["test_stats"].apply(lambda x: np.nan if x == 0 else x)

    return res_series  # .dropna()


def run_model(modeldata, rhsvars, inc_cat=1):
    cat_rhs = "+".join(rhsvars)
    cat = sm.ols(depvar[inc_cat] + " ~ " + cat_rhs, data=modeldata).fit()
    # cat = sm.wls(dataformodels[depvar[inc_cat]]+' ~ ' +cat_rhs, weights = 'population',data=dataformodels).fit()
    print("*" * 80, "\n")
    print(
        "SUMMARY FOR INCOME BIN %s, (%s)"
        % (inc_cat, "dep var: PUMS based shares" if USE_PUMS_SHARES else "fuzzy dep var")
    )
    print(cat.summary2())
    #     toxls = statswrapper(cat).fillna(0)
    #     toxls.to_excel(xlsmodel,sheet_name='inc_cat_%s'%(inc_cat))
    #     pd.DataFrame(toxls).to_latex(os.path.join(OUTPUT_PATH,'incomeregressions_cat__{}.tex'.format(inc_cat,datetime.datetime.now())),na_rep='',formatters=[f1,f1,f1,f4])
    # pd.DataFrame(toxls).to_latex(os.path.join(r'N:\Planning\ABAG Forecasting Models\Household Module\Documentation','incomeregressions_cat__{}.tex'.format(inc_catdatetime.datetime.now())),na_rep='',formatters=[f1,f1,f1,f4])
    # {:%Y%m%d_%H%M%S}
    cat_params = cat.params.to_dict()
    cat_params["bin"] = "cat" + str(inc_cat)
    return cat_params


# UPDATED OCCUPATIONS and data universe MARCH 2020 - removal of smaller counties which may skew regressions
runstate["datauniv"] = "TOP30_MSAS_ACS2013"


cat1_rhs_pums = [
    "cnty_to_us",
    "msasize=='Above 1 million'",
    "occ_det_officeadmin",
    "occ_det_mgmt",
    "ag_65p",
    "Swhite_NH",
]
cat1_params = run_model(
    dataformodels_top30msas.loc["ACS2013"],
    cat1_rhs_pums,  # if USE_PUMS_SHARES else cat1_rhs_reg,
    inc_cat=1,
)

# Specification 1, early Jan 2016
cat2_rhs_pums = [
    "SLF",
    "cnty_to_us",
    "division=='09'",
    "occ_det_officeadmin",
    "occ_det_mgmt",
    "occ_det_hlthsup",
    "Shispanic",
    "Swhite_NH",
    "ag_25_64",
]

cat2_params = run_model(
    dataformodels_top30msas.loc["ACS2013"],
    cat2_rhs_pums,  # if USE_PUMS_SHARES else cat2_rhs_reg,
    inc_cat=2,
)

cat3_rhs_pums = [
    "np.log(per_capita_inc_adj2009)",
    "ind_prof",
    "SLF",
    "Sother_NH",
    "occ_det_foodprep",
    "occ_det_hlthsup",
    "occ_det_biz",
    "msasize=='Above 1 million'",
    "ag_65p",
    "ag_25_64",
]
cat3_params = run_model(
    dataformodels_top30msas.loc["ACS2013"],
    cat3_rhs_pums,  # if USE_PUMS_SHARES else cat3_rhs_reg,
    inc_cat=3,
)

# Specification 2, early Jan 2016
cat4_rhs_pums = [
    "cnty_to_us",
    "occ_det_mgmt",
    "occ_det_community",
    "np.log(per_capita_inc_adj2009)",
    "ag_25_64",
]
cat4_params = run_model(
    dataformodels_top30msas.loc["ACS2013"],
    cat4_rhs_pums,  # if USE_PUMS_SHARES else cat4_rhs_reg,
    inc_cat=4,
)


set(cat1_rhs_pums + cat2_rhs_pums + cat3_rhs_pums + cat4_rhs_pums)


# Running the actual prediction

# read baseline REFERENCE projection
# Reading this as it is the only one of two scenrios we have)

projhh_reference = pd.read_csv(
    "data/intermediate/HeadshipModel/INCOMEref_PBA2040.csv", index_col=[0]
)
projhh_reference.columns = projhh_reference.columns.map(float).map(int)
projhh_reference = projhh_reference.stack()
projhh_reference.index = projhh_reference.index.set_names("Year", level=1)

STARTYEAR = 2015

xlsoutput = pd.ExcelWriter("data/intermediate/HeadshipModel/IncomeModelOutputs.xlsx")


# determines whether to load eased or 'raw' household projections
# from `REGPROJ-HHPROJ 1 - Total household projection PBA2050 Phased HR.ipynb`

USE_EASED = True
# output containers - so we can store data for each scenario in a loop

incomekeep = {}
outputcontainer = {}
counterfactuals = {}
rhs_plotting = {}
# coef_leverage_multiscenario={}
intercepts = {}
coefficients = {}
# coef_leverage_allscens={}
hhprojs = {}
racepop_combo = {}
lfprdata_combo = {}
agepop_combo = {}
homeprices = {}

# For regression model grab vars at the subregion level
# Grab the relevant region-specific variables from the population and labor force dataframes
# These will be used to plug into REMI's regression equation later.
scenario = "R6H2C_COVID_cross_rev2"
# save for later
incomekeep[(scenario, "inc_RPI")] = real_pers_inc.loc[scenario]
incomekeep[(scenario, "inc_RPIcap")] = real_pers_inc_per_cap.loc[scenario]
incomekeep[(scenario, "inc_wage")] = rh_agg_wage_income.loc[scenario]


# -------------------------------------------------------------------------

# Now, use parameters from each model to project future households by income bin.
# For each parameter from the estimated models, we multiply with the appropriate data points.

# TODO: Consider refactoring. Would be simpler to just use named series instead, and
# then just multiply those with a similarly indexed series indexed on parameters holding estimates

# For now store variable name to dataframe relation in a dict
param_to_var = {
    "WRLURI": rv_wharton_geo_adj,
    "np.log(inc_wage_adj2009)": np.log(rh_agg_wage_income).loc[scenario],
    "SLF": rv_S_lf.loc[scenario],
    "cnty_to_us": rv_cnty_to_us.loc[scenario],
    "Sblack_NH": rv_Sblack_NH.loc[scenario],
    "Shispanic": rv_Shispanic.loc[scenario],
    "Sother_NH": rv_Sother_NH.loc[scenario],
    "Swhite_NH": rv_Swhite_NH.loc[scenario],
    "ag_15_24": rv_ag15_24.loc[scenario],
    "ag_25_64": rv_ag25_64.loc[scenario],
    "msasize == 'Above 1 million'[T.True]": rv_msasize,
    "occ_mgmt:occ_svcs": rv_S_occ_mgmt.loc[scenario] * rv_S_occ_svcs.loc[scenario],
    "occ_mgmt:ind_prof": rv_S_occ_mgmt.loc[scenario] * rv_S_ind_prof.loc[scenario],
    "division == '09'[T.True]": rv_is_division_9,
    "division == '03'[T.True]": rv_is_not_division_9,
    "division == '06'[T.True]": rv_is_not_division_9,
    "division == '04'[T.True]": rv_is_not_division_9,
    "is_bay_area2[T.ouside bay area]": rv_is_bay_area,
    "ag_65p": rv_ag65p.loc[scenario],
    "occ_mgmt": rv_S_occ_mgmt.loc[scenario],
    "occ_prod": rv_S_occ_prod.loc[scenario],
    "occ_nat": rv_S_occ_nat.loc[scenario],
    "occ_sls": rv_S_occ_sls.loc[scenario],
    "occ_svcs": rv_S_occ_svcs.loc[scenario],
    "occ_det_mgmt": rv_S_occ_det_mgmt.loc[scenario],
    "occ_det_biz": rv_S_occ_det_biz.loc[scenario],
    "occ_det_comp": rv_S_occ_det_comp.loc[scenario],
    "occ_det_community": rv_S_occ_det_community.loc[scenario],
    "occ_det_lgl": rv_S_occ_det_lgl.loc[scenario],
    "occ_det_hlthsup": rv_S_occ_det_hlthsup.loc[scenario],
    "occ_det_foodprep": rv_S_occ_det_foodprep.loc[scenario],
    "occ_det_officeadmin": rv_S_occ_det_officeadmin.loc[scenario],
    "ind_educ": rv_S_ind_educ.loc[scenario],
    "ind_retail": rv_S_ind_retail.loc[scenario],
    "ind_accom_food_svcs": rv_S_ind_accom_food_svcs.loc[scenario],
    "ind_health": rv_S_ind_health.loc[scenario],
    "ind_prof": rv_S_ind_prof.loc[scenario],
    "np.log(density)": np.log(rv_density),
    "per_capita_inc_adj2009": real_pers_inc_per_cap.loc[scenario],
    "np.log(per_capita_inc_adj2009)": np.log(real_pers_inc_per_cap).loc[
        scenario
    ],  # the remi variable is adjusted to 2009 dollars out of the box
    "np.log(per_capita_inc_adj2009 + 1)": np.log(real_pers_inc_per_cap + 1).loc[scenario],
    "np.log(population)": np.log(futurepop).loc[scenario],
    "years": yearly,
    "Sother_NH:np.log(per_capita_inc)": np.log(real_pers_inc_per_cap).loc[scenario]
    * rv_Sother_NH.loc[scenario],
}


# do the actual prediction - parameter times (future predicted) variable

# compile output into one list
i = 0
yrs = range(STARTYEAR, 2051)
final = {}
coef_leverage = {}
var_x_coef_submodels = {}
var_x_coef_submodels_detail = {}


# we have variables for each submodel in a list. Loop through them and predict shares using models
for i, prms in enumerate([cat1_params, cat2_params, cat3_params, cat4_params]):
    var_x_coef = {}
    for p, v in prms.items():
        if p not in ["bin", "Intercept", "np.log(aland)"]:

            # fetch the corresponding series to the actually estimated model
            param_times_data = param_to_var[p].loc[:, yrs] * v
            var_x_coef[p] = param_times_data
            # coef_leverage_allscens[(scenario,i+1,p)]=param_times_data
            coefficients[(scenario, i + 1, p)] = v

    # after looping through submodels, we sum variables times coefficients, and add intercept

    var_x_coef_submodels[i] = pd.concat(var_x_coef).sum(level=1) + prms["Intercept"]
    var_x_coef_submodels_detail[i] = pd.concat(var_x_coef)
    intercepts[(scenario, i)] = prms["Intercept"]


hhincproj = pd.concat(var_x_coef_submodels)
hhincproj.index = hhincproj.index.set_names("bin", level=0)
hhincproj = hhincproj.swaplevel(0, 1)

# also keep this scenario's coeff times var influence for global sheet
# coef_leverage_multiscenario[scenario]=pd.concat(var_x_coef_submodels_detail).loc[:,[2020,2030,2040,2050]]

# shares now predicted but we need to scale to 100% within each subregion
hhincproj_scaled = hhincproj.groupby(level="Region").apply(pct).stack().unstack("bin")


# subset to just the Bay Area subregions

hhincproj_scaled_long = hhincproj_scaled.loc(0)[
    ["East Bay", "North Bay", "West Bay", "South Bay"], range(STARTYEAR, 2051)
]

# change column names, set indices
hhincproj_scaled_long.columns = ["inc_cat_%s_tot" % x for x in range(1, 5)]
# hhincproj_scaled_long.columns=hhincproj_scaled_long.columns.map(lambda x: 'inc_cat_{}_tot'.format(x))
# hhincproj_scaled_long.columns=hhincproj_scaled_long.columns.set_names('inc_cat')

hhincproj_scaled_long["what"] = "modeled"
hhincproj_scaled_long = hhincproj_scaled_long.reset_index().set_index(["level_1", "Region", "what"])
hhincproj_scaled_long.index = hhincproj_scaled_long.index.set_names(
    ["vintage", "remi_region", "what"]
)
hhincproj_scaled_long = (
    hhincproj_scaled_long.stack().reset_index(name="share").rename(columns={"level_3": "inc_cat"})
)


# index growth to STARTYEAR levels (we do this using start and end year estimated shares)

hhincproj_scaled_indexed = (
    hhincproj_scaled_long.loc[hhincproj_scaled_long.vintage >= 2015]
    .set_index(["vintage", "remi_region", "what", "inc_cat"])
    .share
)
hhincproj_scaled_indexed = hhincproj_scaled_indexed.unstack(["remi_region", "inc_cat"])

# divide with startyear
hhincproj_scaled_indexed = hhincproj_scaled_indexed / hhincproj_scaled_indexed.xs(STARTYEAR)
hhincproj_scaled_indexed = hhincproj_scaled_indexed.stack(level=["remi_region"]).reset_index(
    level=1, drop=True
)

if USE_PUMS_SHARES:
    # bayareashares_pums_2013 = pd.read_csv(os.path.join(box,'EDF Shared work/control_totals/base_data/pums_bayarea_income_shares_1990_2013.csv'))
    # bayareashares_pums_2013.rename(columns={'Region':'remi_region'},inplace=True)
    # bayareashares_pums_2013=bayareashares_pums_2013.loc[(bayareashares_pums_2013.YEAR==2013)&(bayareashares_pums_2013.remi_region!='outside bay area'),1:].set_index('remi_region')
    existing_shares = bayareashares_pums_2015.copy()
else:
    assert "Provide shares from ACS"
    # existing_shares=incshares_nonpums.xs('ACS2013').stack().xs(2013).unstack(0).xs('observed')


# Then evolve existing income distribution by applying these indexed growth in shares to existing shares

income_predicted_shares = (
    (existing_shares.mul(hhincproj_scaled_indexed, axis=0))
    .stack()
    .groupby(level=[0, 1])
    .apply(pct)
    .reset_index(name="share")
    .rename(columns={"level_2": "inc_cat"})
)

income_predicted_shares["what"] = "PBA2050"
income_predicted_shares = income_predicted_shares.rename(columns={"vintage": "years"})
income_predicted_shares["vintage"] = "projected"

# Do some minor reformating of dataframe
# reshape for easy dataframe x series multiplication, aligning on indices
income_predicted_for_counts = income_predicted_shares.copy()
income_predicted_for_counts["inc_cat"] = income_predicted_for_counts.inc_cat.str.extract(
    r"(\d)", expand=False
).astype(np.int64)
income_predicted_for_counts = (
    income_predicted_for_counts.set_index(["remi_region", "years", "inc_cat"]).unstack(2).share
)
income_predicted_for_counts.index = income_predicted_for_counts.index.set_names(["Region", "Year"])


projhh_current = pd.read_csv("data/intermediate/HeadshipModel/INCOMEref_REMI.csv", index_col=[0])

projhh_current.columns = projhh_current.columns.astype(int)
projhh_current = projhh_current.stack()
projhh_current.index = projhh_current.index.set_names(["Region", "Year"])


# ----------------------------------------------------------------------
# DONE WITH YEARLY INCOME BIN SHARE PREDICTION
# Now, we need to apply to the relevant household forecast to get counts
# ----------------------------------------------------------------------

# read latest projection of the relevant scenario
projhh_current = pd.read_csv("data/intermediate/HeadshipModel/INCOMEref_REMI.csv", index_col=[0])

projhh_current.columns = projhh_current.columns.astype(int)
projhh_current = projhh_current.stack()
projhh_current.index = projhh_current.index.set_names(["Region", "Year"])

hhprojs[scenario] = projhh_current

# Drum roll - predicted households times predicted shares
# scenario specific

hhincproj_count = (
    projhh_current.mul(income_predicted_for_counts.stack(), axis=0)
    .loc[:, YEARS_OUT_1_2015]
    .unstack("inc_cat")
)
hhincproj_count["vintage"] = "Projection"
hhincproj_count.set_index("vintage", append=True, inplace=True)
hhincproj_count.index = hhincproj_count.index.reorder_levels([2, 0, 1])
hhincproj_count.columns = hhincproj_count.columns.map(lambda x: "bin_%s" % x)

# apply to *reference* projection instead - that is, use scenario-specific income shares,
# but use with BASELINE population instead of scenario.

hhincproj_count_reference = (
    projhh_reference.mul(income_predicted_for_counts.stack(), axis=0)
    .loc[:, YEARS_OUT_1_2015]
    .unstack("inc_cat")
)
hhincproj_count_reference["vintage"] = "Projection REF"
hhincproj_count_reference.set_index("vintage", append=True, inplace=True)
hhincproj_count_reference.index = hhincproj_count_reference.index.reorder_levels([2, 0, 1])
hhincproj_count_reference.columns = hhincproj_count_reference.columns.map(lambda x: "bin_%s" % x)


# the 2010, 2015 counts come from one year pums files

# PUMS 2015 is about 20,000 higher than DOF's count. We scale back to DOF levels.
scale_pums_to_2015 = (
    hhincproj_count.sum(level=[2]).loc[2015].sum() / bayareaincomeobs.sum(axis=1).loc[2015]
)
scale_pums_to_2015 = pd.Series(scale_pums_to_2015, index=pd.Index([2015]), name="YEAR")

bayareaincomeobs.update(bayareaincomeobs.loc[[2015]].mul(scale_pums_to_2015, axis=0))

hist_and_projected = pd.concat(
    [
        bayareaincomeobs,  # data2010.unstack(),
        hhincproj_count.sum(level=[2]).loc[range(2016, 2051, 1)],
    ]
)
hist_and_projected.to_excel(xlsoutput, ("REMIcurr_inc_proj")[:30])
outputcontainer[scenario] = hist_and_projected

hist_and_projected_ref = pd.concat(
    [
        bayareaincomeobs,  # data2010.unstack(),
        hhincproj_count_reference.sum(level=[2]).loc[range(2016, 2051, 1)],
    ]
)

counterfactuals[scenario] = hist_and_projected_ref

# RHS Plotting--future remi vars
# TODO: minor refactoring to avoid redundancy - we could use `param_to_var` instead and save this step

for_rhs_plotting = pd.DataFrame(
    data={
        "SLF": rv_S_lf.stack().loc[scenario],
        "real_pers_inc": real_pers_inc_per_cap.stack().loc[scenario],
        "lf_female": rv_S_lf_female.stack().loc[scenario],
        "occ_mgmt": rv_S_occ_mgmt.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_svcs": rv_S_occ_svcs.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_prod": rv_S_occ_prod.loc[:, range(2015, 2051)].stack().loc[scenario],
        "wage_income": rh_agg_wage_income.stack().loc[scenario],
        "wharton_geo": rv_wharton_geo.stack(),
        "wharton_geo_adj": rv_wharton_geo_adj.stack(),
        "msasize": rv_msasize.stack(),
        "density": rv_density.stack(),
        "ind_educ": rv_S_ind_educ.loc[:, range(2015, 2051)].stack().loc[scenario],
        "ind_retail": rv_S_ind_retail.loc[:, range(2015, 2051)].stack().loc[scenario],
        "ind_accom_food_svcs": rv_S_ind_accom_food_svcs.loc[:, range(2015, 2051)]
        .stack()
        .loc[scenario],
        "ind_health": rv_S_ind_health.loc[:, range(2015, 2051)].stack().loc[scenario],
        "ind_prof": rv_S_ind_prof.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_sls": rv_S_occ_sls.loc[:, range(2015, 2051)].stack().loc[scenario],
        "Swhite_NH": rv_Swhite_NH.stack().loc[scenario],
        "Sother_NH": rv_Sother_NH.stack().loc[scenario],
        "Sblack_NH": rv_Sblack_NH.stack().loc[scenario],
        "Shispanic": rv_Shispanic.stack().loc[scenario],
        "ag15_24": rv_ag15_24.stack().loc[scenario],
        "ag25_64": rv_ag25_64.stack().loc[scenario],
        "occ_det_mgmt": rv_S_occ_det_mgmt.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_biz": rv_S_occ_det_biz.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_comp": rv_S_occ_det_comp.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_community": rv_S_occ_det_community.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_lgl": rv_S_occ_det_lgl.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_hlthsup": rv_S_occ_det_hlthsup.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_foodprep": rv_S_occ_det_foodprep.loc[:, range(2015, 2051)].stack().loc[scenario],
        "occ_det_officeadmin": rv_S_occ_det_officeadmin.loc[:, range(2015, 2051)]
        .stack()
        .loc[scenario],
        "ag65p": rv_ag65p.stack().loc[scenario],
    }
)
rhs_plotting[scenario] = for_rhs_plotting

# plot out the counts by bin

fig = plt.figure(figsize=(11, 8))
plt.tight_layout()
hhincproj_count.sum(level=["Year"]).loc[YEARS_OUT_1_2015].plot()
plt.title("Households, by income group\nScenario %s" % (scenario))
plt.tight_layout()

plt.show(block=True)
pd.concat(outputcontainer, axis=0).astype(np.int64).to_excel(xlsoutput, "scenariocombo")
pd.concat(outputcontainer, axis=0).astype(np.int64).stack().reset_index(name="value").rename(
    columns={"level_0": "runid", "level_1": "year", "level_2": "bin"}
).to_csv(os.path.join(OUTPUT_PATH, "incomeproj_for_MSA_ACS"))

# write out *right hand side* data
# pd.concat(rhs_plotting).stack().reset_index(name='value').rename(columns={'level_0':'runid','level_1':'Region','level_2':'year','level_3':'variable'}).to_csv(os.path.join(OUTPUT_PATH,"incomeproj_hhinc_rhs_vars_remi.csv"))

# write out coefficient influence
pd.Series(coefficients).reset_index().rename(
    columns={"level_0": "scenario", "level_1": "bin", "level_2": "variable", 0: "coefficient"}
).to_excel(xlsoutput, "coefficients")
df_intercept = pd.DataFrame(data={"value": pd.Series(intercepts)})
df_intercept["variable"] = "AAintercept"
df_intercept.set_index("variable", append=True, inplace=True)
df_intercept.index = df_intercept.index.set_names(["scenario", "bin", "variable"])

df_coefficient = pd.DataFrame(data={"value": pd.Series(coefficients)})
df_coefficient.index = df_coefficient.index.set_names(["scenario", "bin", "variable"])
pd.concat([df_intercept, df_coefficient]).sort_index().unstack(0).to_excel(
    xlsoutput, "specifications"
)

xlsoutput.close()
