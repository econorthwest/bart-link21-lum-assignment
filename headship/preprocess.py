import glob
import os

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm


def get_preprocess_data():
    # check if files exist else run preprocess
    headship = "data/intermediate/HeadshipModel/headship_rates.csv"
    headship_e = "data/intermediate/HeadshipModel/headship_rates_eased.csv"

    if headship and headship_e in glob.glob("data/intermediate/HeadshipModel/*.csv"):
        return pd.read_csv(
            headship, index_col=["vintage", "region", "rac_ethn", "age_grp_5"]
        ).value, pd.read_csv(headship_e, index_col=["Year", "rac_ethn", "age_grp_5"])

    else:
        # This section comes from Preprocess A notebook
        print("Loading County-PUMA crosswalk (2010)")
        bayareafips_full = {
            "06001": "Alameda",
            "06013": "Contra Costa",
            "06041": "Marin",
            "06055": "Napa",
            "06075": "San Francisco",
            "06081": "San Mateo",
            "06085": "Santa Clara",
            "06097": "Sonoma",
            "06095": "Solano",
        }

        pumarel = pd.read_csv("data/inputs/HeadshipModel/2010_County_to_2010_PUMA.csv", dtype=str)
        pumas = (
            pumarel[pumarel["cnt_id"].isin(bayareafips_full.keys())]["puma_id"].unique().tolist()
        )

        print("Loading PUMS data, pulled from tidycensus using CensusPull.R")

        acs = ["ACS 2012", "ACS 12-16", "ACS 2013", "ACS 2014", "ACS 14-18", "ACS 2018"]
        allpersons = {}

        for i in acs:
            print(f"- {i}")
            persons_temp = pd.read_csv(
                "data/inputs/HeadshipModel/pums_{fill}.csv".format(fill=i[4:]),
                usecols=[
                    "PWGTP",
                    "SERIALNO",
                    "SPORDER",
                    "AGEP",
                    "RAC1P",
                    "HISP",
                    "ST",
                    "PUMA",
                    "SEX",
                    "RELP",
                ],
                low_memory=False,
            )
            persons_temp["YEAR"] = "20" + i[-2:]
            allpersons[(i)] = persons_temp

        persons = pd.concat(allpersons, names=["VINTAGE", "OID"])
        persons.groupby(["VINTAGE"]).PWGTP.sum()

        print("Mapping persons to demographic variables")
        # Creating mapping dictionaries and functions
        diffbreaks_5 = list(range(0, 86, 5)) + [np.inf]

        def agebreaker2(breaks):
            labels = []
            for f in range(len(breaks) - 1):
                labels.append("Ages {fr:.0f}-{to:.0f}".format(fr=breaks[f], to=breaks[f + 1] - 1))
            labels[-1] = "Ages {dt:,.0f}+".format(dt=breaks[-2])
            return labels

        remirace_value_map = {
            1: "White",
            2: "Black",
            3: "Other",
            4: "Other",
            5: "Other",
            6: "Other",
            7: "Other",
            8: "Other",
            9: "Other",
        }

        # Mapping
        persons["STPUMA"] = persons.ST.apply(lambda x: "{:0>2}".format(x)) + persons.PUMA.apply(
            lambda x: "{:0>5}".format(x)
        )  # Locating PUMS observations
        persons = persons[persons["STPUMA"].isin(pumas)]  # Selecting people in Bay Area
        persons["region"] = "Bay Area"

        # Age
        persons["age_grp_5"] = pd.cut(
            persons.AGEP, right=False, bins=diffbreaks_5, labels=agebreaker2(diffbreaks_5)
        )
        persons["sex"] = persons.SEX.map({1: "male", 2: "female"})  # Sex

        persons["race_remi"] = persons.RAC1P.map(remirace_value_map)  # Race

        # Ethnicity
        persons["rac_ethn"] = persons.groupby(["HISP"])["race_remi"].transform(
            lambda x: "Hispanic" if x.name > 1 else x + "-NonHispanic"
        )

        persons = persons.drop(labels=["SEX"], axis=1)

        print("Calculating headship rates")
        hhgrp = ["VINTAGE", "region", "rac_ethn", "age_grp_5"]
        numerator = persons.loc[persons.RELP.isin([0])].groupby(hhgrp).PWGTP.sum()
        denominator = persons.loc[(~persons.RELP.isin([16, 17]))].groupby(hhgrp).PWGTP.sum()
        headship = numerator.div(denominator, axis=0)

        print(
            headship.unstack("VINTAGE")
            .loc["Bay Area", "White-NonHispanic", agebreaker2(diffbreaks_5)]
            .dropna()
            .reset_index([0, 1], drop=True)
        )

        print(">>> Preprocess A completed <<<")

        # This section comes from Preprocess B notebook
        print("Loading County-PUMA crosswalk (2000)")
        puma2000 = pd.read_csv("data/inputs/HeadshipModel/2000_County_to_2000_PUMA.csv")
        puma2000["cnt_id"] = puma2000.cnt_id.apply(lambda x: "{:0>5}".format(x))
        puma2000["puma_id"] = puma2000.puma_id.apply(lambda x: "{:0>7}".format(x))
        puma2000 = puma2000.groupby(["puma_id"]).cnt_id.first()

        print("Loading 2000 PUMS data, pulled from census website")
        # start by pulling lay out file
        layout = pd.ExcelFile("data/inputs/HeadshipModel/5%_PUMS_record_layout.xls")

        def get_fwf_tuples(excel, sheet, skiprw=1):
            temp = excel.parse(sheet, skiprw)
            temp = temp.loc[(temp.RT.notnull()) & (temp.BEG.notnull())]
            striplist = ["RT", "BEG", "END", "VARIABLE", "DESCRIPTION"]
            for i in striplist:
                temp[i] = temp[i].apply(
                    lambda x: np.nan
                    if str(x).strip() == "" or str(x).strip() == "nan"
                    else str(x).strip()
                )
            temp = temp[temp.BEG.notnull()]
            for i in ["BEG", "END"]:
                temp[i] = temp[i].astype(int)
            temp = (
                temp.groupby(["RT", "BEG", "END", "VARIABLE", "DESCRIPTION"])
                .size()
                .reset_index(name="value")
            )
            temp = temp.loc[~temp.DESCRIPTION.str.contains("1% file")]
            names = temp.VARIABLE.to_dict()
            coldefs = temp.apply(lambda x: (int(x.BEG) - 1, int(x.END)), axis=1).tolist()
            return names, coldefs

        persons_names, persons_coldefs = get_fwf_tuples(layout, "Person Record")
        housing_names, housing_coldefs = get_fwf_tuples(layout, "Housing Unit Record")

        # now pull fwf file and assign housing and persons observations
        if "data/intermediate/HeadshipModel/PUMS2000_Housing.csv" in glob.glob(
            "data/intermediate/HeadshipModel/*.csv"
        ):
            census2000hca = pd.read_csv("data/intermediate/HeadshipModel/PUMS2000_Housing.csv")
        else:
            print(">> Processing Household Census PUMS in chunks")
            chunk_size = 10_000
            total_chunks = int(2_342_340 / chunk_size) + 1
            h_chunks = []
            for chunk in tqdm(
                pd.read_fwf(
                    "data/inputs/HeadshipModel/REVISEDPUMS5_06.txt",
                    colspecs=housing_coldefs,
                    header=None,
                    chunksize=chunk_size,
                    # nrows=100000,
                ),
                total=total_chunks,
            ):
                h_chunk = chunk.rename(columns=housing_names)
                h_chunk = h_chunk.loc[h_chunk["RECTYPE"] == "H"]
                h_chunks.append(h_chunk[["RECTYPE", "SERIALNO", "PUMA5"]])

            census2000hca = pd.concat(h_chunks, axis=0)

            census2000hca.to_csv(
                os.path.join("data/intermediate/HeadshipModel/PUMS2000_Housing.csv")
            )

        if "data/intermediate/HeadshipModel/PUMS2000_Persons.csv" in glob.glob(
            "data/intermediate/HeadshipModel/*.csv"
        ):
            census2000pca = pd.read_csv("data/intermediate/HeadshipModel/PUMS2000_Persons.csv")
        else:
            print(">> Processing Person Census PUMS in chunks")
            p_chunks = []
            for chunk in tqdm(
                pd.read_fwf(
                    "data/inputs/HeadshipModel/REVISEDPUMS5_06.txt",
                    colspecs=persons_coldefs,
                    header=None,
                    chunksize=chunk_size,
                    # nrows=100000,
                ),
                total=total_chunks,
            ):
                p_chunk = chunk.rename(columns=persons_names)
                p_chunk = p_chunk.loc[p_chunk["RECTYPE"] == "P"]
                p_chunks.append(
                    p_chunk[
                        [
                            "RECTYPE",
                            "SERIALNO",
                            "PWEIGHT",
                            "RELATE",
                            "RACE1",
                            "AGE",
                            "HISPAN",
                        ]
                    ]
                )
            census2000pca = pd.concat(p_chunks, axis=0)

            census2000pca.to_csv(
                os.path.join("data/intermediate/HeadshipModel/PUMS2000_Persons.csv")
            )

        # Mapping to demographic
        print("Census Data Wrangling")

        census2000hca["STPUMA"] = census2000hca["PUMA5"].apply(lambda x: "06{:05d}".format(x))
        census2000hca["STCOUNTY"] = census2000hca.STPUMA.map(puma2000)
        census2000pca["STCOUNTY"] = census2000pca.SERIALNO.map(
            census2000hca.groupby("SERIALNO").STCOUNTY.first()
        )

        census2000pca["race_remi"] = census2000pca.RACE1.map(remirace_value_map)
        census2000pca["age_grp_5"] = pd.cut(
            census2000pca.AGE, right=False, bins=diffbreaks_5, labels=agebreaker2(diffbreaks_5)
        ).astype(str)
        census2000pca["rac_ethn"] = census2000pca.groupby(["HISPAN"])["race_remi"].transform(
            lambda x: "Hispanic" if x.name > 1 else x + "-NonHispanic"
        )
        census2000pca = census2000pca[census2000pca.STCOUNTY.isin(bayareafips_full.keys())]
        census2000pca["region"] = "Bay Area"

        hhgrp = ["region", "rac_ethn", "age_grp_5"]
        numerator = census2000pca.loc[(census2000pca.RELATE == 1)].groupby(hhgrp).PWEIGHT.sum()
        denominator = (
            census2000pca.loc[(~census2000pca.RELATE.isin([22, 23]))].groupby(hhgrp).PWEIGHT.sum()
        )
        headship2k = numerator / denominator
        headship2k = headship2k.fillna(0)

        # Writing to single headship rate file
        print("Writing headship rate csv")

        headship_combo_df = pd.concat(
            [pd.concat([headship2k], keys=["Census 2000"], names=["VINTAGE"]), headship]
        ).reset_index(name="value")

        headship_combo_df.columns = headship_combo_df.columns.str.lower()
        headship_combo_df.age_grp_5 = headship_combo_df.age_grp_5.astype(
            CategoricalDtype(categories=agebreaker2(diffbreaks_5), ordered=True)
        )

        headship_combo_df.to_csv(os.path.join("data/intermediate/HeadshipModel/headship_rates.csv"))

        # EASER COPIED EXACTLY FROM MTC CODES
        print("Sinusoidal easing")
        # Takes a source series (keyed on year) and transitions to a target series over a period of time.
        headship_combo = headship_combo_df.set_index(
            ["vintage", "region", "rac_ethn", "age_grp_5"]
        ).value.sort_index()
        headship_combo = headship_combo.loc[:, :, :, agebreaker2(diffbreaks_5)[3:]]

        HR_START = "ACS 14-18"  # If we are using the easer we should decide what year what ACS we start with (see code)
        HR_END = "Census 2000"

        source_series = {}
        for yr in range(2015, 2051):
            source_series[yr] = headship_combo.loc[HR_START, "Bay Area"]
        source_series = pd.concat(source_series, names=["Year"]).unstack(
            level=["rac_ethn", "age_grp_5"]
        )

        target_series = {}
        for yr in range(2015, 2051):
            target_series[yr] = headship_combo.loc[HR_END, "Bay Area"]
        target_series = pd.concat(target_series, names=["Year"]).unstack(
            level=["rac_ethn", "age_grp_5"]
        )

        def easer(
            target_series=target_series,
            source_series=source_series,
            t_0=7,
            t_1=15,
            envelope_year_start=2015,
            envelope_year_end=2050,
        ):
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

            # To the right of the convergence window, it should be 0 (i.e. no difference to, or fully transitioned to target series)
            easing[index >= t_1] = 0
            # turn in to a pd.Series
            easing = pd.Series(easing, index=np.arange(envelope_year_start, envelope_year_end + 1))
            easing.index = easing.index.set_names("Year")
            target_series.index = target_series.index.astype(str)
            source_series.index = source_series.index.astype(str)
            easing.index = easing.index.astype(str)

            source_less_target = source_series - target_series
            output = (source_less_target.mul(easing, axis=0) + target_series).stack(level=[0, 1])
            output.name = "value"
            return output

        hr_eased = easer(
            target_series=target_series,
            source_series=source_series,
            envelope_year_start=2015,
            envelope_year_end=2050,
            t_0=9,
            t_1=23,
        )

        print("\n \n", hr_eased.loc[["2015", "2020", "2025"], :].unstack(1))

        print(
            "\n \n",
            headship_combo.loc[
                ["Census 2000", "ACS 12-16", "ACS 14-18", "ACS 18"],
                ["Bay Area"],
                ["Black-NonHispanic", "Hispanic"],
            ]
            .unstack([1, 0])
            .sort_index(axis=1)
            .fillna(0),
        )

        hr_eased.to_csv(os.path.join("data/intermediate/HeadshipModel/headship_rates_eased.csv"))

        print(">>> Preprocess B completed <<<")

        return headship_combo, hr_eased


if __name__ == "__main__":
    headship_combo, hr_eased = get_preprocess_data()
