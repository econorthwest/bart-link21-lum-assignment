import itertools
import json
import multiprocessing
import time
from pathlib import Path

import choicemodels
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from tqdm import tqdm
from utils import (
    HowLong,
    Seeded,
    check_create_dir,
    read_predicted_model,
    run_location_choice_mnl,
    summarize_results,
)

data_dir = Path("../data")

dirs = {
    "lc": {
        "input": data_dir.joinpath("inputs/LocationChoice"),
        "intermediate": data_dir.joinpath("intermediate/LocationChoice"),
    },
    "scenarios": {
        "input": data_dir.joinpath("inputs/scenarios"),
        "intermediate": data_dir.joinpath("intermediate/scenarios"),
    },
}


# spaces_map = {
#     "FFRE": "industrial",
#     "CONS": "office",
#     "UTIL": "industrial",
#     "MFRG": "industrial",
#     "WTWT": "industrial",
#     "RETL": "retail",
#     "RETR": "retail",
#     "FIRE": "office",
#     "K12E": "heur",
#     "HIED": "heur",
#     "HMED": "heur",
#     "SOCS": "office",
#     "RECS": "retail",
#     "PERS": "retail",
#     "GOVT": "office",
# }


spaces_map = {
    "AGREMPN": "industrial",
    "FPSEMPN": "office",
    "MWTEMPN": "industrial",
    "OTHEMPN": "office",
    "RETEMPN": "retail",
    "HEREMPN": "retail",
}


# SECTION 1: DATA AXLE PREPROCESS


def import_naics_mapping():
    """Imports mapping csv for NAICS to TM1.5 and LINK 21 codes from repo"""
    tm15_code = pd.read_csv(dirs["lc"]["input"].joinpath("naics_tm15_crosswalk.csv"))
    tm15_code = tm15_code.filter(items=["naics_code", "tm15_code", "space_type"])
    tm15_code = tm15_code[-tm15_code.duplicated()]
    tm15_code["naics_code"] = tm15_code["naics_code"].astype(str)
    tm15_code = tm15_code.set_index("naics_code")
    return tm15_code


def remove_county_miscoded(axle):
    """Takes data Axle and removes 189 businesses where the county identifier in data axle does not agree with actual county location"""
    # Set up data axle and read county shape file
    print(">>> Removing miscoded spatial outliers")
    axle["FIPS CODE"] = axle["FIPS CODE"].astype(str).apply(lambda x: "0" + x)
    geo_axle = gpd.GeoDataFrame(
        axle, geometry=gpd.points_from_xy(axle.LONGITUDE, axle.LATITUDE, crs="epsg:4269")
    )
    countyshp = gpd.read_file(dirs["lc"]["input"].joinpath("bay_area_counties.shp"))
    # Conduct spatial join
    print(">>> Locating businesses within counties")
    joined_spatial = gpd.tools.sjoin(geo_axle, countyshp, how="left")
    # Filter out business where spatially designated county code matched axle county code
    joined_spatial = joined_spatial[joined_spatial["FIPS CODE"] == joined_spatial["GEOID"]]
    print(">>> Outlier removed")
    return joined_spatial


def process_data_axle(data_axle_df):
    """Function takes Axle data and outputs DataFrame of jobs with TAZ location, occupation codes"""
    # Filter variables of interest
    new_df = data_axle_df.filter(
        items=[
            "EMPLOYEE SIZE (5) - LOCATION",
            "PRIMARY NAICS CODE",
            "LATITUDE",
            "LONGITUDE",
            "GEOID",
        ]
    )
    new_df["employees"] = new_df["EMPLOYEE SIZE (5) - LOCATION"].replace(np.nan, 0)
    # Creating appropriate NAICS length for mapping codes

    def naics_mapping_prep(naics_prime):
        i = naics_prime[0:2]

        if i == "61" or i == "62":
            i = naics_prime[0:3]
        return i

    # Apply mapping function
    print(">>> Mapping NAICS to Link21 Codes")
    new_df["naics_lead"] = new_df["PRIMARY NAICS CODE"].astype(str).apply(naics_mapping_prep)
    new_df["naics_lead"] = new_df["naics_lead"].astype(str)
    tm15_code = import_naics_mapping()
    new_df = new_df.merge(tm15_code, how="left", left_on="naics_lead", right_index=True)

    # Remove unclassified businesses
    new_df = new_df[new_df["tm15_code"].notnull()]
    # Convert to GeoDataFrame
    print(">>> Locating businesses")
    geo_df = gpd.GeoDataFrame(
        new_df, geometry=gpd.points_from_xy(new_df.LONGITUDE, new_df.LATITUDE, crs="epsg:4269")
    )
    # Import TAZ boundries and locate businesses
    print(">>> Assigning to TAZ")
    bay_taz = gpd.read_file(dirs["lc"]["input"].joinpath("Link21_TAZ_FINAL.shp"))
    sjoined_df = gpd.tools.sjoin(geo_df, bay_taz, how="left")
    # Remove one boundry case assigned to two TAZs
    sjoined_df = sjoined_df.reset_index()
    sjoined_df.drop_duplicates(subset="index", inplace=True, ignore_index=True)
    sjoined_df = sjoined_df.drop(columns="index")
    # Correct for points just outside TAZ boundry

    def correct_points(datataz):
        if datataz["LINK21_TAZ"].isnull().sum() == 0:
            return datataz
        else:
            print(">>> Correcting for points outside TAZ boundries\n")
            preassigned = datataz[datataz["LINK21_TAZ"].notnull()]
            temp_geo = bay_taz.to_crs("EPSG:32610")
            temp_axle = datataz[datataz["LINK21_TAZ"].isnull()].iloc[:, 0:10].to_crs("EPSG:32610")
            nowassign = temp_axle.sjoin_nearest(temp_geo, how="left").to_crs("EPSG:4269")
            datataz = pd.concat([preassigned, nowassign])
            return datataz

    geo_emp = correct_points(sjoined_df).filter(
        items=["employees", "tm15_code", "space_type", "LINK21_TAZ", "GEOID"]
    )
    # Expand each business based on number of workers
    output_df = geo_emp.loc[geo_emp.index.repeat(geo_emp.employees)]
    output_df = (
        output_df.drop(columns="employees")
        .reset_index()
        .rename(columns={"index": "employer_id", "LINK21_TAZ": "chosenTAZ"})
    )

    # Correct for one miscoded TAZ that appears in two counties
    output_df.set_index("chosenTAZ", inplace=True)
    if len(output_df.loc[3081]["GEOID"].unique()) > 1:
        output_df.at[3081, "GEOID"] = "06055"
    output_df.reset_index(inplace=True)

    return output_df


def write_jobs_per_county(jobsdata):
    """Function classifies processed job-level data by county and employment category and writes output files"""
    codes = ["AGREMPN", "OTHEMPN", "MWTEMPN", "RETEMPN", "FPSEMPN", "HEREMPN"]
    counties = ["06001", "06013", "06041", "06055", "06075", "06081", "06085", "06095", "06097"]
    county_jobs = {}
    # Subset jobs for each county and write out as a Arrow Parquet file
    for county in counties:
        cntydata = jobsdata[jobsdata["GEOID"] == county]
        empsectordata = {}
        for empcode in codes:
            empsectordata[empcode] = cntydata[cntydata["tm15_code"] == empcode]
            check_create_dir(dirs["lc"]["intermediate"].joinpath("jobs/observed", county))
            filename = dirs["lc"]["intermediate"].joinpath(
                "jobs/observed", county, f"{empcode}.parquet"
            )
            pq.write_table(
                pa.Table.from_pandas(cntydata[cntydata["tm15_code"] == empcode]), filename
            )
            # csv_filename = dirs["lc"]["intermediate"].joinpath(
            #     "jobs/observed", county, f"{empcode}.csv"
            # )
            # cntydata[cntydata["tm15_code"] == empcode].to_csv(csv_filename, index=False)
        county_jobs[county] = empsectordata
    return county_jobs


def prepare_data_axle():
    if not dirs["lc"]["intermediate"].joinpath("jobs/observed/06081/RETEMPN.parquet").exists():
        print("Preparing Data Axle Data")
        # clean data create jobs observations
        raw_df = pd.read_csv(data_dir.joinpath("inputs/data-axle/Order_1432579.csv"))
        cleaned_data_df = remove_county_miscoded(raw_df)
        alljobs = process_data_axle(cleaned_data_df)
        # divide by counties and write to parquet files
        write_jobs_per_county(alljobs)
    else:
        print("Data Axle Data already prepared... doing nothing.")


def preprocess_heuristic_shares(county, category):
    """Function creates CSV with share of education jobs for each county which will be used for heuristic assignment of REMI jobs"""
    obs_emp = pq.read_pandas(
        dirs["lc"]["intermediate"].joinpath("jobs/observed", county, "HEREMPN.parquet")
    ).to_pandas()
    # Given 3 heursitic categories, subset category of interest
    if category == "HMED":
        obs_emp = obs_emp[obs_emp["space_type"] == "heur_med"]
    elif category == "HIED" or category == "K12E":
        obs_emp = obs_emp[obs_emp["space_type"] == "heur_edu"]
    # Calculate shares.
    obs_emp["count"] = 1
    obs_emp = obs_emp.groupby(["GEOID", "chosenTAZ"]).sum(numeric_only=True)["count"]
    obs_emp = obs_emp.div(obs_emp.groupby("GEOID").sum(numeric_only=True))
    return obs_emp


# SECTION 2: TAZ FEATURES PREPROCESS


def process_scenario_taz_data(scenario_name):
    """Combines TAZ feature inputs into single df. Inputs combined: County crosswalk, accessibility data, supply model data, calculating geographic variables"""
    scenario_input_dir = dirs["scenarios"]["input"].joinpath(scenario_name)
    scenario_intermediate_dir = dirs["scenarios"]["intermediate"].joinpath(scenario_name)
    crosswalk = pd.read_csv(dirs["lc"]["input"].joinpath("CountyTAZcrosswalk.csv"))
    # Prepare crosswalk dictionary to assign each taz to a county
    print(">>> Assigning TAZs to county")
    crosswalk["GEOID"] = crosswalk["GEOID"].astype(str).apply(lambda x: "0" + x)
    crosswalk = crosswalk.set_index("LINK21_TAZ").to_dict()["GEOID"]
    # Create miles to points of interest
    taz_shp = gpd.read_file(dirs["lc"]["input"].joinpath("Link21_TAZ_FINAL.shp")).set_index(
        "LINK21_TAZ"
    )
    print(">>> Calculating miles to centers")
    centers_dict = {
        "center": [
            "downtown_Oakland",
            "Angel_Island",
            "Presidio",
            "Stanford",
        ],
        "geometry": [
            Point(-122.2711, 37.8033),  # downtown Oakland
            Point(-122.4326, 37.8609),  # Angel Island
            Point(-122.4662, 37.7989),  # Presidio
            Point(-122.1697, 37.4275),  # Stanford
        ],
    }
    centers_gdf = (
        gpd.GeoDataFrame(centers_dict, crs="EPSG:4269").to_crs("EPSG:32610").set_index("center")
    )
    temp_gdf = taz_shp.to_crs("EPSG:32610")
    meters_in_a_mile = 1609.34
    col_by_center = {}
    for centerName, center in centers_gdf.iterrows():
        col_by_center["to_" + centerName] = temp_gdf.geometry.apply(
            lambda point: point.distance(center.iloc[0]) / meters_in_a_mile
        )
    spatial_var_gdf = pd.DataFrame(col_by_center).reset_index()
    taz_shp = taz_shp.merge(spatial_var_gdf, left_on="LINK21_TAZ", right_on="LINK21_TAZ")
    taz_shp = taz_shp.filter(
        items=["LINK21_TAZ", "to_downtown_Oakland", "to_Angel_Island", "to_Presidio", "to_Stanford"]
    )

    # Remove geometry and join with TAZ features data, assign county
    print(">>> Joining supply model generated TAZ features")
    supply_model_features = pd.read_csv(scenario_input_dir.joinpath("MapCraft_TAZfeatures.csv"))
    if scenario_name != "base_year":
        supply_model_features = supply_model_features[supply_model_features["link21_taz"] > 0]
        supply_model_features = supply_model_features.filter(
            items=[
                "link21_taz" "vliRentalUnitsSum",
                "liRentalUnitsSum",
                "miRentalUnitsSum",
                "marketRateRentalUnitsSum",
                "vliOwnUnitsSum",
                "liOwnUnitsSum",
                "miOwnUnitsSum",
                "marketRateOwnUnitsSum",
                "officeJobSpacesSum",
                "retailJobSpacesSum",
                "industrialJobSpacesSum",
            ]
        )
        supply_model_features = supply_model_features.rename(
            columns={
                "link21_taz": "LINK21_TAZ",
                "vliRentalUnitsSum": "vli_units_rent",
                "liRentalUnitsSum": "li_units_rent",
                "miRentalUnitsSum": "mi_units_rent",
                "marketRateRentalUnitsSum": "residential_units_rent",
                "vliOwnUnitsSum": "vli_units_own",
                "liOwnUnitsSum": "li_units_own",
                "miOwnUnitsSum": "mi_units_own",
                "marketRateOwnUnitsSum": "residential_units_own",
                "officeJobSpacesSum": "office_job_spaces",
                "retailJobSpacesSum": "retail_job_spaces",
                "industrialJobSpacesSum": "industrial_job_spaces",
            }
        )
    else:
        supply_model_features = supply_model_features.assign(
            vli_units_rent=0,
            li_units_rent=0,
            mi_units_rent=0,
            vli_units_own=0,
            li_units_own=0,
            mi_units_own=0,
        )
        scaled_capacties = (
            pd.read_csv(scenario_input_dir.joinpath("MapCraft_ScaledTAZCapacities.csv"))
            .set_index("link21_taz")
            .drop(columns="scaled_non_residential_sqft")
        )
        supply_model_features = supply_model_features.merge(
            scaled_capacties, how="left", left_on="LINK21_TAZ", right_index=True
        )
        supply_model_features = supply_model_features.drop(
            columns=["office_job_spaces", "retail_job_spaces", "industrial_job_spaces"]
        ).rename(
            columns={
                "scaled_office_job_spaces": "office_job_spaces",
                "scaled_retail_job_spaces": "retail_job_spaces",
                "scaled_industrial_job_spaces": "industrial_job_spaces",
            }
        )

    supply_model_features["county"] = supply_model_features["LINK21_TAZ"].map(crosswalk)
    supply_model_features = supply_model_features.merge(taz_shp, on="LINK21_TAZ")
    supply_model_features = supply_model_features.rename(columns={"LINK21_TAZ": "altTAZ"})
    supply_model_features = supply_model_features.replace(np.nan, 0)

    print(">>> Joining composite accessibility variables")
    supply_model_features = supply_model_features.merge(
        prepare_accessibility(scenario_input_dir), left_on="altTAZ", right_on="zone_ID"
    )

    print(">>> Joining price surface")
    price_surface = gpd.read_file(scenario_input_dir.joinpath("price_surface.geojson"))

    price_surface["sfr_prototype"] = (
        price_surface["sfr_spatial_const"]
        + (price_surface["sfr_sq_ft_coef"] * 1730)
        + (price_surface["sfr_beds_coef"] * 3)
        + (price_surface["sfr_baths_coef"] * 3.5)
        + (price_surface["sfr_stories_coef"] * 3)
        + (price_surface["sfr_is_townhome_coef"] * 1)
        + (price_surface["sfr_is_gt_2800_sf_coef"] * 0)
        + (price_surface["sfr_yr_built_gt_2009_coef"] * 1)
        + (price_surface["sfr_transit_0_40_coef"] * price_surface["sfr_transit_0_40"])
        + (price_surface["sfr_transit_40_80_coef"] * price_surface["sfr_transit_40_80"])
        + (price_surface["sfr_highway_0_40_coef"] * price_surface["sfr_highway_0_40"])
        + (price_surface["sfr_highway_40_80_coef"] * price_surface["sfr_highway_40_80"])
        + (price_surface["sfr_totemp_coef"] * price_surface["TOTEMP"])
    )

    price_surface["1bd_rent"] = (
        price_surface["mfr_spatial_const"]
        + (price_surface["mfr_star_rating_coef"] * 5)
        + (price_surface["mfr_yr_built_coef"] * 2019)
        + (price_surface["mfr_n_amenities_coef"] * 15)
        + (price_surface["mfr_transit_0_40_coef"] * price_surface["mfr_transit_0_40"])
        + (price_surface["mfr_highway_0_30_coef"] * price_surface["mfr_highway_0_30"])
        + (price_surface["mfr_highway_30_60_coef"] * price_surface["mfr_highway_30_60"])
    )
    price_surface["mfr_prototype"] = 0.89775 * (price_surface["1bd_rent"] * 0.75) + (
        price_surface["mfr_number_of_stories_coef"] * 3
    )
    price_surface["off_prototype"] = (1 + 0.05) * (
        price_surface["off_spatial_const"]
        + (price_surface["off_number_of_stories_coef"] * 4)
        + (price_surface["off_parking_ratio_coef"] * 1)
        + (price_surface["off_star_rating_coef"] * 5)
        + (price_surface["off_yr_built_gt_2009_coef"] * 1)
        + (price_surface["off_transit_0_40_coef"] * price_surface["off_transit_0_40"])
        + (price_surface["off_transit_40_80_coef"] * price_surface["off_transit_40_80"])
        + (price_surface["off_totemp_coef"] * price_surface["TOTEMP"])
    )

    price_surface = price_surface.filter(
        items=["LINK21_TAZ", "sfr_prototype", "mfr_prototype", "off_prototype"]
    )

    supply_model_features = supply_model_features.merge(
        price_surface,
        left_on="altTAZ",
        right_on="LINK21_TAZ",
    ).drop(columns="LINK21_TAZ")

    print(">>> Joining avg household incomes\n")

    syn_pop = pd.read_csv(dirs["lc"]["input"].joinpath("2015_CS_Pop_1215.csv"))
    syn_pop = syn_pop.filter(items=["TAZ", "HINC"]).groupby("TAZ").mean()

    supply_model_features = supply_model_features.merge(
        syn_pop, left_on="altTAZ", right_on="TAZ", how="left"
    )
    supply_model_features["HINC"] = supply_model_features["HINC"].replace(np.nan, 0)

    check_create_dir(scenario_intermediate_dir)
    pq.write_table(
        pa.Table.from_pandas(supply_model_features),
        scenario_intermediate_dir.joinpath("taz_features.parquet"),
    )

    return supply_model_features


def prepare_accessibility(scenario_input_dir):
    """Prepare composite accessibility variables"""
    # Using simple counts for non-motorized, percived for others
    accessibilities_filepath = scenario_input_dir.joinpath("accessibility.xlsx")
    acc_met = pd.read_excel(
        accessibilities_filepath,
        sheet_name="perceived_TT",
    )

    acc_met.set_index("zone_ID", inplace=True)

    def make_composite_acc_metric(populaiton, mode, lowerbound, upperbound):
        output = []
        for i in range(lowerbound, upperbound, 10):
            output.append("%s_AM_%s_%s_%s" % (populaiton, mode, i, i + 10))
        return acc_met[output].sum(axis=1)

    acc_met = acc_met.assign(
        TOTEMP_highway_0_30=make_composite_acc_metric("TOTEMP", "highway", 0, 30),
        TOTEMP_highway_30_60=make_composite_acc_metric("TOTEMP", "highway", 30, 60),
        TOTEMP_highway_0_40=make_composite_acc_metric("TOTEMP", "highway", 0, 40),
        TOTEMP_highway_40_80=make_composite_acc_metric("TOTEMP", "highway", 40, 80),
        EMPRES_highway_0_40=make_composite_acc_metric("EMPRES", "highway", 0, 40),
        EMPRES_highway_40_80=make_composite_acc_metric("EMPRES", "highway", 40, 80),
        TOTEMP_wtransit_40_80=make_composite_acc_metric("TOTEMP", "wtransit", 40, 80),
        EMPRES_wtransit_40_80=make_composite_acc_metric("EMPRES", "wtransit", 40, 80),
    )

    acc_met = acc_met.rename(
        columns={
            "TOTEMP_AM_wtransit_0_40": "TOTEMP_wtransit_0_40",
            "EMPRES_AM_wtransit_0_40": "EMPRES_wtransit_0_40",
        }
    ).filter(
        items=[
            "zone_ID",
            "TOTEMP_highway_0_30",
            "TOTEMP_highway_30_60",
            "TOTEMP_highway_0_40",
            "TOTEMP_highway_40_80",
            "EMPRES_highway_0_40",
            "EMPRES_highway_40_80",
            "TOTEMP_wtransit_0_40",
            "TOTEMP_wtransit_40_80",
            "EMPRES_wtransit_0_40",
            "EMPRES_wtransit_40_80",
        ]
    )
    return acc_met


def prepare_taz_data(scenario_name):
    """TEMP FUNCTION: Final version should only look in scenerio file"""
    try:
        taz_alts = pq.read_pandas(
            dirs["scenarios"]["intermediate"].joinpath(f"{scenario_name}/taz_features.parquet")
        ).to_pandas()
    except FileNotFoundError:
        print("Preparing TAZ Alternatives")
        taz_alts = process_scenario_taz_data(scenario_name)
    return taz_alts


# SECTION 3: MNL MODELING


def get_taz_alts(county):
    taz_alts = prepare_taz_data(scenario_name="base_year")
    return taz_alts[taz_alts["county"] == county]


def estimate_single_employment_model(row):
    county, sector, model_exp, obs_sample_size, taz_sample_size, filename = row
    alts_emp = get_taz_alts(county)
    alts_emp.set_index("altTAZ", inplace=True)
    outputfile = {"exp": model_exp}
    obs_emp = pq.read_pandas(
        dirs["lc"]["intermediate"].joinpath("jobs/observed", county, f"{sector}.parquet")
    ).to_pandas()

    # Remove heurictically assigned jobs
    if sector == "HEREMPN":
        obs_emp = obs_emp[obs_emp["space_type"] == "retail"]

    # Some emp files still have 1-2 obs from outside county, drop emps if this has happened
    outliers_list = list(set(obs_emp["chosenTAZ"]) - set(alts_emp.reset_index()["altTAZ"]))
    if len(outliers_list) > 0:
        for i in outliers_list:
            obs_emp = obs_emp[obs_emp["chosenTAZ"] != i]

    if obs_sample_size is not None:
        if obs_sample_size <= 1:
            raise NotImplementedError
        elif obs_sample_size > 1:
            if len(obs_emp) > obs_sample_size:
                seed = Seeded(4925)
                with seed:
                    obs_emp = obs_emp.sample(obs_sample_size)
    alt_count = len(alts_emp)
    if taz_sample_size and taz_sample_size < alt_count:
        alt_count = taz_sample_size

    tqdm.write(
        f"{county} {sector}: {len(obs_emp):,} choosers * {alt_count:,} alternatives"
        f" = {len(obs_emp) * alt_count:,} combinations"
    )
    est_seed = Seeded(1298)
    # Run MultiNomialLogit model
    with est_seed:
        with HowLong(message=f"{county} {sector} {len(obs_emp) * alt_count:,} combinations"):
            fitted_model = run_location_choice_mnl(
                obs_emp, alts_emp, model_exp, alt_count
            ).get_raw_results()

    # Converting pd.DataFrame within lc raw results to json
    fitted_model["fit_parameters"] = fitted_model["fit_parameters"].to_json(orient="split")
    outputfile["%s" % sector] = fitted_model

    # Export raw results for whole county as a json
    check_create_dir(dirs["lc"]["intermediate"].joinpath(f"jobs/results/{filename}"))
    with open(
        dirs["lc"]["intermediate"].joinpath(f"jobs/results/{filename}", f"{county}-{sector}.json"),
        "w",
    ) as outfile:
        outfile.write(json.dumps(outputfile, indent=4))


emp_spec1 = "np.log1p(HINC) + np.log1p(to_downtown_Oakland) + np.log1p(to_Stanford) + np.log1p(TOTEMP_highway_0_40) + np.log1p(TOTEMP_highway_40_80) + np.log1p(EMPRES_highway_0_40) + np.log1p(TOTEMP_wtransit_0_40) + np.log1p(TOTEMP_wtransit_40_80) + np.log1p(EMPRES_wtransit_0_40) + np.log1p(office_job_spaces) + np.log1p(retail_job_spaces) + np.log1p(industrial_job_spaces) + np.log1p(off_spatial_const)"
emp_spec2 = "np.log1p(HINC) + np.log1p(to_downtown_Oakland) + np.log1p(to_Stanford) + np.log1p(TOTEMP_highway_0_40) + np.log1p(TOTEMP_highway_40_80) + np.log1p(EMPRES_highway_0_40) + np.log1p(TOTEMP_wtransit_0_40) + np.log1p(TOTEMP_wtransit_40_80) + np.log1p(EMPRES_wtransit_0_40) + np.log1p(off_prototype)"
emp_spec3 = "np.log1p(HINC) + np.log1p(to_downtown_Oakland) + np.log1p(to_Stanford) + np.log1p(TOTEMP_highway_0_40) + np.log1p(TOTEMP_wtransit_0_40) + np.log1p(EMPRES_highway_0_40) + np.log1p(EMPRES_highway_40_80) + np.log1p(EMPRES_wtransit_0_40) + np.log1p(EMPRES_wtransit_40_80) + np.log1p(off_prototype)"


def estimate_employment_lc(model_exp, obs_sample_size=None, taz_sample_size=None, num_workers=1):
    """Function runs multinomial logit for employment and exports choicemodels.MultinomialLogitResults object as .json"""

    counties = ["06081", "06085", "06075", "06001", "06055", "06095", "06013", "06097", "06041"]
    # counties = ["06055"]
    sectors = ["AGREMPN", "OTHEMPN", "MWTEMPN", "RETEMPN", "FPSEMPN", "HEREMPN"]
    exp = [model_exp]
    filename = ["ELCM_Est_" + time.strftime("%m%d_%I-%M%p")]
    models = list(
        itertools.product(counties, sectors, exp, [obs_sample_size], [taz_sample_size], filename)
    )

    if num_workers == 1:
        for model_args in tqdm(models, total=len(models)):
            estimate_single_employment_model(model_args)
    else:
        pool = multiprocessing.Pool(processes=num_workers)
        try:
            with tqdm(total=len(models)) as pbar:
                for _ in pool.imap_unordered(estimate_single_employment_model, models):
                    pbar.update()
        finally:
            pool.close()
            pool.join()
    return


# SECTION 4: SIMULATION

INPUT_REMI_PATH = Path(
    "~/Dropbox (ECONW)/24182 BART Transbay Rail Crossing/Data/LandEcon Group REMI"
)


def create_remi_job_counts(scenario_name):
    """Function reads in required REMI output files to create job counts by county"""
    # Import Employment by Industry REMI output table
    # TODO: Make this glob glob so it can read based on string
    indust = pd.read_excel(
        INPUT_REMI_PATH.joinpath(
            scenario_name,
            "REMI 3.0.0 Tables/5v2_REMI3.0.0 Employment- Employment by Occupation - Employment by Industry and Occupation.xlsx",
        ),
        sheet_name="All",
        skiprows=5,
    )
    # Subset to data county-industry specific data
    indust = (
        indust[indust["Region"] != "All Regions"]
        .rename(columns={"Region": "county"})
        .set_index("county")
    )
    indust = indust[indust["Industry"] != "All Industries"]
    indust = indust.filter(items=["Industry", 2050]).set_index("Industry", append=True)
    # Import and join TM1.5 and Link21 occupation code crosswalks
    remi_occ_maps = pd.read_csv(dirs["lc"]["input"].joinpath("naics_tm15_crosswalk.csv")).drop(
        columns=["naics_class"]
    )
    # Import MTC employment tranlation factors and multiply with raw REMI counts as per MTC methodology
    remi_emp_translation_sectors = pd.read_csv("../data/inputs/LocationChoice/sectormap.csv")
    remi_emp_translation_factors = pd.read_csv(
        "../data/inputs/LocationChoice/emp_translation_factors.csv"
    )
    remi_emp_translation_factors = remi_emp_translation_sectors.merge(
        remi_emp_translation_factors
    ).drop(columns="ind")

    output = (
        indust.reset_index(0)
        .merge(remi_occ_maps, left_on="Industry", right_on="naics_classification")
        .merge(remi_emp_translation_factors, right_on="REMI_NAICS", left_on="naics_code")
        .set_index("county")
        .filter(items=[2050, "tm15_code", "space_type", "adjustment"])
    )

    output[2050] = output[2050].mul(output["adjustment"]).mul(1000)
    output = output.reset_index()

    # Convert county names to fips
    names_to_fips = {
        "Alameda County": "06001",
        "Contra Costa County": "06013",
        "Marin County": "06041",
        "Napa County": "06055",
        "San Francisco County": "06075",
        "San Mateo County": "06081",
        "Santa Clara County": "06085",
        "Solano County": "06095",
        "Sonoma County": "06097",
    }

    output["county"] = output["county"].map(names_to_fips)
    output = (
        output.drop(columns="adjustment")
        .groupby(["county", "tm15_code", "space_type"])
        .sum(numeric_only=True)
        .round()
        .astype(int)
        .reset_index()
    )
    return output


def explode_remi_job_counts(df_count):
    """Utility function for ELCM simulation that ennumarates job counts into a DataFrame where each row is a job"""
    df_count = df_count[df_count[2050].notnull()]
    repeat_idx = df_count.index.repeat(df_count[2050].astype(int))
    exploded = df_count.reindex(repeat_idx)
    exploded = exploded.reset_index().drop(columns=[2050, "index"])
    return exploded


def run_elcm_simulation(remi_jobs, taz_alts_scenario, relocation_rates=None):
    """Creates a relocation pool (base year relocating + new jobs) from REMI jobs and outputs final horizon TAZ assignment"""
    sectors = ["AGREMPN", "OTHEMPN", "MWTEMPN", "RETEMPN", "FPSEMPN", "HEREMPN"]
    # Assign relocation rates for each sector
    if relocation_rates is None:
        relocation_rates = {
            "AGREMPN": 0.2,
            "OTHEMPN": 0.2,
            "MWTEMPN": 0.2,
            "RETEMPN": 0.2,
            "FPSEMPN": 0.2,
            "HEREMPN": 0.2,
        }
    # remi_jobs["space_type"] = remi_jobs["link21_occmap"].map(spaces_map)
    spaces = ["retail", "industrial", "office"]
    # Iterate through counties and assign jobs
    counties = ["06081", "06085", "06075", "06001", "06055", "06095", "06013", "06097", "06041"]
    # counties = ["06075", "06001", "06055", "06095", "06013", "06097", "06041"]
    final_output = {}
    df_sampling_seed = Seeded(4925)
    for county in tqdm(counties):
        tqdm.write(f"Assigning for {county}")
        # Step 1: Prepare preassigned jobs (by base year) and assignment pool
        jobs_assigned = {}
        jobs_for_assignment = []
        capacity_adjustments = []
        remi_jobs_county = remi_jobs[remi_jobs["county"] == county]
        taz_alts_county = taz_alts_scenario[taz_alts_scenario["county"] == county]
        taz_alts_county.set_index("altTAZ", inplace=True)

        # Introduce capacity multipliers to adjust for Data Axle job demand mismatch
        capacity_needed = []
        for sector in sectors:
            base_jobs = pq.read_pandas(
                dirs["lc"]["intermediate"].joinpath(f"jobs/observed/{county}/{sector}.parquet")
            ).to_pandas()
            capacity_needed.append(base_jobs)

        capacity_needed = (
            pd.concat(capacity_needed, axis=0)
            .groupby(["space_type"])
            .count()
            .rename(columns={"employer_id": "demand"})["demand"]
        )
        capacity_available = (
            taz_alts_county.filter(
                items=["office_job_spaces", "retail_job_spaces", "industrial_job_spaces"]
            )
            .rename(
                columns={
                    "office_job_spaces": "office",
                    "retail_job_spaces": "retail",
                    "industrial_job_spaces": "industrial",
                }
            )
            .sum()
        )
        capacity_multiplier = pd.concat([capacity_needed, capacity_available], axis=1).rename(
            columns={0: "available"}
        )
        capacity_multiplier["multiplier"] = (
            1.05 * capacity_multiplier["demand"] / capacity_multiplier["available"]
        )
        capacity_multiplier["multiplier"] = capacity_multiplier["multiplier"].mask(
            capacity_multiplier["multiplier"].lt(1), 1
        )
        # import pdb

        # pdb.set_trace()

        taz_alts_county["office_job_spaces"] = (
            (taz_alts_county["office_job_spaces"] * capacity_multiplier["multiplier"].loc["office"])
            .round()
            .astype(int)
        )
        taz_alts_county["retail_job_spaces"] = (
            (taz_alts_county["retail_job_spaces"] * capacity_multiplier["multiplier"].loc["retail"])
            .round()
            .astype(int)
        )
        taz_alts_county["industrial_job_spaces"] = (
            (
                taz_alts_county["industrial_job_spaces"]
                * capacity_multiplier["multiplier"].loc["industrial"]
            )
            .round()
            .astype(int)
        )

        capacity_test = (
            taz_alts_county.filter(
                items=["office_job_spaces", "retail_job_spaces", "industrial_job_spaces"]
            )
            .rename(
                columns={
                    "office_job_spaces": "office",
                    "retail_job_spaces": "retail",
                    "industrial_job_spaces": "industrial",
                }
            )
            .sum()
        )
        capacity_multiplier = pd.concat([capacity_multiplier, capacity_test], axis=1).rename(
            columns={0: "corrected"}
        )
        capacity_multiplier["test"] = (
            capacity_multiplier["demand"] / capacity_multiplier["corrected"]
        )

        assert not (capacity_multiplier["test"] > 1).any()

        # Generate assignment pool

        for sector in tqdm(sectors):

            tqdm.write(f"Calculating assignment requirements in {sector}")
            remi_jobs_temp = remi_jobs_county[remi_jobs_county["tm15_code"] == sector]
            # Import base year job counts
            base_jobs = pq.read_pandas(
                dirs["lc"]["intermediate"].joinpath(f"jobs/observed/{county}/{sector}.parquet")
            ).to_pandas()

            if sector == "HEREMPN":
                # Subset jobs for heuristic assignement
                heur_assignment = remi_jobs_temp[remi_jobs_temp["space_type"] != "retail"]
                heur_base = base_jobs[base_jobs["space_type"] != "retail"]
                # Remove heursitically assigned jobs from files that will be passed to simulation
                remi_jobs_temp = remi_jobs_temp[remi_jobs_temp["space_type"] == "retail"]
                base_jobs = base_jobs[base_jobs["space_type"] == "retail"]
                # Heuristically assign set aside jobs from K12E, HIED and HMED
                # Starting with HMED
                hmed_base = heur_base[heur_base["space_type"] == "heur_med"]
                hmed_assign = heur_assignment[heur_assignment["space_type"] == "heur_med"]
                hmed_assign_count = hmed_assign.loc[:, 2050].iloc[0] - len(hmed_base)

                # Raise warning if horizon year has fewer jobs that base year.
                if hmed_assign_count > 0:
                    hmed_base = hmed_base.pivot_table(
                        index="chosenTAZ",
                        columns="space_type",
                        aggfunc="count",
                        values="GEOID",
                    ).replace(np.nan, 0)
                    hmed_assigned = preprocess_heuristic_shares(county, "HMED") * hmed_assign_count
                    hmed_base["assigned"] = hmed_assigned.reset_index(level=0, drop=True)
                    hmed_base["total"] = hmed_base["heur_med"] + hmed_base["assigned"]
                    jobs_assigned["heur_HMED"] = hmed_base["total"].round(0).rename("heur_med")
                else:
                    with df_sampling_seed:
                        hmed_base = (
                            hmed_base.sample(n=(len(hmed_base) + hmed_assign_count))
                            .pivot_table(
                                index="chosenTAZ",
                                columns="space_type",
                                aggfunc="count",
                                values="GEOID",
                            )
                            .replace(np.nan, 0)
                        )
                    jobs_assigned["heur_HMED"] = hmed_base["heur_med"].round(0)
                    tqdm.write(
                        f"\nWARNING: Base year jobs exceed REMI prediction in {county} for Link21 code HMED"
                    )
                assert (
                    heur_assignment[heur_assignment["space_type"] == "heur_med"][2050].iloc[0]
                ) == (jobs_assigned["heur_HMED"].sum())

                # Repeat heursitic assignment process for education jobs
                educ_base = heur_base[heur_base["space_type"] == "heur_edu"]
                educ_assign = heur_assignment[heur_assignment["space_type"] == "heur_edu"]
                educ_assign_count = educ_assign.loc[:, 2050].iloc[0] - len(educ_base)

                if educ_assign_count > 0:
                    educ_base = educ_base.pivot_table(
                        index="chosenTAZ",
                        columns="space_type",
                        aggfunc="count",
                        values="GEOID",
                    ).replace(np.nan, 0)
                    educ_assigned = preprocess_heuristic_shares(county, "HIED") * educ_assign_count
                    educ_base["assigned"] = educ_assigned.reset_index(level=0, drop=True)
                    educ_base["total"] = educ_base["heur_edu"] + educ_base["assigned"]
                    jobs_assigned["heur_HIED"] = educ_base["total"].round(0).rename("heur_edu")
                else:
                    with df_sampling_seed:
                        educ_base = (
                            educ_base.sample(n=(len(educ_base) + educ_assign_count))
                            .pivot_table(
                                index="chosenTAZ",
                                columns="space_type",
                                aggfunc="count",
                                values="GEOID",
                            )
                            .replace(np.nan, 0)
                        )
                    jobs_assigned["heur_HIED"] = educ_base["heur_edu"].round(0)
                    tqdm.write(
                        f"\n WARNING: Base year jobs exceed REMI prediction in {county} for Link21 code HIED+K12E"
                    )
                assert (
                    heur_assignment[heur_assignment["space_type"] == "heur_edu"][2050].iloc[0]
                ) == (jobs_assigned["heur_HIED"].sum())

            # Subset REMI jobs that need to be assigned, append to assignment listexir
            if remi_jobs_temp[2050].sum() - len(base_jobs) < 0:
                tqdm.write(
                    f"\n WARNING: Base year jobs exceed REMI prediction in {county} for {sector}"
                )

            # To account for counties with negative growth, relocating population is min(base_year, horizon_year)*relocation_rate
            relocating_jobs = min(len(base_jobs), remi_jobs_temp[2050].sum())
            relocating_jobs = round(relocating_jobs * relocation_rates[sector])

            # Subset base year jobs that will relocate, summarize remainaining jobs to TAZ level
            with df_sampling_seed:
                if len(base_jobs) > remi_jobs_temp[2050].sum():
                    base_jobs = base_jobs.sample(n=remi_jobs_temp[2050].sum(), replace=False)
                base_jobs = base_jobs.sample(n=len(base_jobs) - relocating_jobs)

            adjust_stayers = (
                base_jobs.rename(columns={"GEOID": "county"})
                .pivot_table(
                    index=["county", "tm15_code"],
                    aggfunc="count",
                    columns="space_type",
                )["chosenTAZ"]
                .replace(np.nan, 0)
                .stack()
            )

            remi_jobs_temp = pd.concat(
                [remi_jobs_temp.set_index(["county", "tm15_code", "space_type"]), adjust_stayers],
                axis=1,
            ).rename(columns={0: "stayers"})
            remi_jobs_temp[2050] = remi_jobs_temp[2050] - remi_jobs_temp["stayers"]
            remi_jobs_temp = remi_jobs_temp.drop(columns="stayers").reset_index()

            jobs_for_assignment.append(explode_remi_job_counts(remi_jobs_temp))

            base_jobs = base_jobs.pivot_table(
                index="chosenTAZ", columns="tm15_code", aggfunc="count", values="GEOID"
            ).replace(np.nan, 0)

            jobs_assigned[f"{sector}_base"] = base_jobs

        jobs_for_assignment = pd.concat(jobs_for_assignment, axis=0)
        jobs_for_assignment = jobs_for_assignment.filter(items=["tm15_code", "space_type"])

        # Adjust TAZ capacities by subtracting base year non-relocating jobs and correcting for overcapacity TAZs
        capacity_adjustments = pd.concat(jobs_assigned, axis=1).replace(np.nan, 0)

        capacity_adjustments.columns = [
            "AGREMPN",
            "OTHEMPN",
            "MWTEMPN",
            "RETEMPN",
            "FPSEMPN",
            "heur_HMED",
            "heur_HIED",
            "HEREMPN",
        ]

        # TODO: Reevaluate this logic now that we have appropriate TAZ capacities
        # Adjust office space TAZ capacities
        capacity_adjustments_office = pd.concat(
            [capacity_adjustments[["OTHEMPN", "FPSEMPN"]], taz_alts_county["office_job_spaces"]],
            axis=1,
        )
        capacity_adjustments_office = capacity_adjustments_office.assign(
            office_total=lambda x: x.OTHEMPN + x.FPSEMPN,
            overcapacity=lambda x: x.office_job_spaces - x.office_total,
        )
        capacity_adjustments_office["overcapacity"] = capacity_adjustments_office[
            "overcapacity"
        ].apply(lambda x: abs(x) if (x < 0) else 0)
        capacity_adjustments_office = capacity_adjustments_office.assign(
            OTHEMPN_relocator=lambda x: ((x.overcapacity / x.office_total) * x.OTHEMPN).round(0),
            FPSEMPN_relocator=lambda x: ((x.overcapacity / x.office_total) * x.FPSEMPN).round(0),
            OTHEMPN=lambda x: x.OTHEMPN - x.OTHEMPN_relocator,
            FPSEMPN=lambda x: x.FPSEMPN - x.FPSEMPN_relocator,
        )

        # Correct for rounding error
        test_office_adjustments = (
            capacity_adjustments_office[["OTHEMPN", "FPSEMPN"]].sum(axis=1)
            > capacity_adjustments_office["office_job_spaces"]
        )
        while test_office_adjustments.any():
            correct = capacity_adjustments_office[-test_office_adjustments].copy()
            incorrect = capacity_adjustments_office[test_office_adjustments].copy()
            if (incorrect["OTHEMPN"] > 1).all():
                incorrect["OTHEMPN"] = incorrect["OTHEMPN"] - 1
                incorrect["OTHEMPN_relocator"] = incorrect["OTHEMPN_relocator"] + 1
            else:
                incorrect["FPSEMPN"] = incorrect["FPSEMPN"] - 1
                incorrect["FPSEMPN_relocator"] = incorrect["FPSEMPN_relocator"] + 1
            capacity_adjustments_office = pd.concat([correct, incorrect], axis=0)
            del correct
            del incorrect
            test_office_adjustments = (
                capacity_adjustments_office[["OTHEMPN", "FPSEMPN"]].sum(axis=1)
                > capacity_adjustments_office["office_job_spaces"]
            )

        assert not test_office_adjustments.any()
        del test_office_adjustments

        office_relocators_addendum = pd.DataFrame(
            {
                "tm15_code": ["OTHEMPN", "FPSEMPN"],
                "space_type": ["office", "office"],
                2050: capacity_adjustments_office[["OTHEMPN_relocator", "FPSEMPN_relocator"]]
                .sum()
                .astype(int)
                .to_list(),
            }
        )
        jobs_for_assignment = pd.concat(
            [jobs_for_assignment, explode_remi_job_counts(office_relocators_addendum)], axis=0
        )

        # Adjust retail space TAZ capacities

        capacity_adjustments_retail = pd.concat(
            [capacity_adjustments[["RETEMPN", "HEREMPN"]], taz_alts_county["retail_job_spaces"]],
            axis=1,
        )
        capacity_adjustments_retail = capacity_adjustments_retail.assign(
            retail_total=lambda x: x.RETEMPN + x.HEREMPN,
            overcapacity=lambda x: x.retail_job_spaces - x.retail_total,
        )
        capacity_adjustments_retail["overcapacity"] = capacity_adjustments_retail[
            "overcapacity"
        ].apply(lambda x: abs(x) if (x < 0) else 0)
        capacity_adjustments_retail = capacity_adjustments_retail.assign(
            RETEMPN_relocator=lambda x: ((x.overcapacity / x.retail_total) * x.RETEMPN).round(0),
            HEREMPN_relocator=lambda x: ((x.overcapacity / x.retail_total) * x.HEREMPN).round(0),
            RETEMPN=lambda x: x.RETEMPN - x.RETEMPN_relocator,
            HEREMPN=lambda x: x.HEREMPN - x.HEREMPN_relocator,
        )

        test_retail_adjustments = (
            capacity_adjustments_retail[["RETEMPN", "HEREMPN"]].sum(axis=1)
            > capacity_adjustments_retail["retail_job_spaces"]
        )
        while test_retail_adjustments.any():
            correct = capacity_adjustments_retail[-test_retail_adjustments].copy()
            incorrect = capacity_adjustments_retail[test_retail_adjustments].copy()
            if (incorrect["RETEMPN"] > 1).all():
                incorrect["RETEMPN"] = incorrect["RETEMPN"] - 1
                incorrect["RETEMPN_relocator"] = incorrect["RETEMPN_relocator"] + 1
            else:
                incorrect["HEREMPN"] = incorrect["HEREMPN"] - 1
                incorrect["HEREMPN_relocator"] = incorrect["HEREMPN_relocator"] + 1
            capacity_adjustments_retail = pd.concat([correct, incorrect], axis=0)
            del correct
            del incorrect
            test_retail_adjustments = (
                capacity_adjustments_retail[["RETEMPN", "HEREMPN"]].sum(axis=1)
                > capacity_adjustments_retail["retail_job_spaces"]
            )

        assert not test_retail_adjustments.any()
        del test_retail_adjustments

        retail_relocators_addendum = pd.DataFrame(
            {
                "tm15_code": ["RETEMPN", "HEREMPN"],
                "space_type": ["retail", "retail"],
                2050: capacity_adjustments_retail[["RETEMPN_relocator", "HEREMPN_relocator"]]
                .sum()
                .astype(int)
                .to_list(),
            }
        )
        jobs_for_assignment = pd.concat(
            [jobs_for_assignment, explode_remi_job_counts(retail_relocators_addendum)], axis=0
        )
        # Adjust industrial space TAZ capacities
        capacity_adjustments_industrial = pd.concat(
            [
                capacity_adjustments[["AGREMPN", "MWTEMPN"]],
                taz_alts_county["industrial_job_spaces"],
            ],
            axis=1,
        )
        capacity_adjustments_industrial = capacity_adjustments_industrial.assign(
            industrial_total=lambda x: x.AGREMPN + x.MWTEMPN,
            overcapacity=lambda x: x.industrial_job_spaces - x.industrial_total,
        )
        capacity_adjustments_industrial["overcapacity"] = capacity_adjustments_industrial[
            "overcapacity"
        ].apply(lambda x: abs(x) if (x < 0) else 0)
        capacity_adjustments_industrial = capacity_adjustments_industrial.assign(
            AGREMPN_relocator=lambda x: ((x.overcapacity / x.industrial_total) * x.AGREMPN).round(
                0
            ),
            MWTEMPN_relocator=lambda x: ((x.overcapacity / x.industrial_total) * x.MWTEMPN).round(
                0
            ),
            AGREMPN=lambda x: x.AGREMPN - x.AGREMPN_relocator,
            MWTEMPN=lambda x: x.MWTEMPN - x.MWTEMPN_relocator,
        )

        test_industrial_adjustments = (
            capacity_adjustments_industrial[["AGREMPN", "MWTEMPN"]].sum(axis=1)
            > capacity_adjustments_industrial["industrial_job_spaces"]
        )
        while test_industrial_adjustments.any():
            correct = capacity_adjustments_industrial[-test_industrial_adjustments].copy()
            incorrect = capacity_adjustments_industrial[test_industrial_adjustments].copy()
            if (incorrect["AGREMPN"] > 1).all():
                incorrect["AGREMPN"] = incorrect["AGREMPN"] - 1
                incorrect["AGREMPN_relocator"] = incorrect["AGREMPN_relocator"] + 1
            else:
                incorrect["MWTEMPN"] = incorrect["MWTEMPN"] - 1
                incorrect["MWTEMPN_relocator"] = incorrect["MWTEMPN_relocator"] + 1
            capacity_adjustments_industrial = pd.concat([correct, incorrect], axis=0)
            del correct
            del incorrect
            test_industrial_adjustments = (
                capacity_adjustments_industrial[["AGREMPN", "MWTEMPN"]].sum(axis=1)
                > capacity_adjustments_industrial["industrial_job_spaces"]
            )

        assert not test_industrial_adjustments.any()
        del test_industrial_adjustments

        industrial_relocators_addendum = pd.DataFrame(
            {
                "tm15_code": ["AGREMPN", "MWTEMPN"],
                "space_type": ["industrial", "industrial"],
                2050: capacity_adjustments_industrial[["AGREMPN_relocator", "MWTEMPN_relocator"]]
                .sum()
                .astype(int)
                .to_list(),
            }
        )
        jobs_for_assignment = pd.concat(
            [jobs_for_assignment, explode_remi_job_counts(industrial_relocators_addendum)], axis=0
        )

        jobs_assigned = pd.concat(
            [
                capacity_adjustments_office[["OTHEMPN", "FPSEMPN"]],
                capacity_adjustments_retail[["RETEMPN", "HEREMPN"]],
                capacity_adjustments_industrial[["AGREMPN", "MWTEMPN"]],
                jobs_assigned["heur_HMED"],
                jobs_assigned["heur_HIED"],
            ],
            axis=1,
        ).replace(np.nan, 0)
        jobs_assigned = jobs_assigned.assign(HEREMPN=lambda x: x.HEREMPN + x.heur_med + x.heur_edu)
        jobs_assigned = jobs_assigned.drop(columns=["heur_med", "heur_edu"])

        # Assert that total assignment+assigned pool is exactly equal to REMI jobs

        test_for_assignment = jobs_for_assignment.groupby("tm15_code").count()

        test_assigned = jobs_assigned.sum()
        test_remi_totals = remi_jobs_county.groupby("tm15_code")[2050].sum()
        test_diff = pd.concat(
            [test_for_assignment, test_assigned, test_remi_totals], axis=1
        ).rename(columns={2050: "REMI", 0: "assigned", "space_type": "for_assignment"})

        test_diff["sumtotal"] = (
            test_diff["REMI"] - test_diff["assigned"] - test_diff["for_assignment"]
        )
        assert not (test_diff["sumtotal"] != 0).any()

        print(
            "> County: %s" % county,
            "\nTotal horizon year jobs :",
            remi_jobs[remi_jobs["county"] == county][2050].sum(),
            "\nNew and Relocating jobs for assignment: ",
            len(jobs_for_assignment),
            "\n\n",
        )

        # Step 2: Update probs_callable used by simulation to make it county specific (this cannot be an input due to ChoiceModels mechanism)
        def mct_callable(obs, alts, intx_ops=None):
            return choicemodels.tools.MergedChoiceTable(obs, alts, sample_size=None)

        def probs_callable(mct, county="%s" % county):
            output_probs = []
            df = mct.to_frame()
            temp_inds_comp = list(mct.to_frame()["tm15_code"].unique())
            for sector in temp_inds_comp:  # only tm15_occs from line 321 above
                df_temp = df[df["tm15_code"] == sector].copy()
                mct_temp = choicemodels.tools.MergedChoiceTable.from_df(df_temp)
                mct_temp.sample_size = len(df.groupby(level=1))
                model_dir = dirs["lc"]["intermediate"].joinpath("jobs")
                fitted_model = read_predicted_model(model_dir, county, sector)
                output_probs.append(fitted_model.probabilities(mct_temp))
            output = pd.concat(output_probs, axis=0, sort=True)
            return output

        results = []

        # Step 3: For all jobs using a type of job space, simulate location choice
        for space in tqdm(spaces):
            tqdm.write(f"Assigning {space} type jobs")
            jobs_space_assignment = jobs_for_assignment[jobs_for_assignment["space_type"] == space]
            jobs_space_assignment = jobs_space_assignment.assign(
                newind=range(len(jobs_space_assignment))
            ).set_index("newind")
            jobs_space_assignment.index.name = None

            # Step3c: Filter relevant columns and set "capacity" column relevant to assignment
            taz_alts_county["capacity"] = taz_alts_county["%s_job_spaces" % space]
            simSeed = Seeded(1761)
            with simSeed:
                sim_out = choicemodels.tools.iterative_lottery_choices(
                    jobs_space_assignment,
                    taz_alts_county,
                    mct_callable,
                    probs_callable,
                    chooser_batch_size=10000,
                    alt_capacity="capacity",
                )

            sim_out_result = (
                jobs_space_assignment.reset_index()
                .merge(sim_out.reset_index(), on="index")
                .drop(columns=["index"])
            )

            results.append(sim_out_result)

        results = pd.concat(results, axis=0)

        results = results.rename(columns={"altTAZ": "TAZ"})
        assert len(results) == len(jobs_for_assignment)

        # Pivot to summary table with counts for each TAZ and TM1.5 code
        results["jobs"] = 1
        simulation_assigned = pd.pivot_table(
            results, index="TAZ", columns="tm15_code", aggfunc=sum, values="jobs"
        ).replace(np.nan, 0)
        output = pd.DataFrame(index=taz_alts_county.index)
        for sector in sectors:
            output["%s" % sector] = (
                pd.concat(
                    [simulation_assigned["%s" % sector], jobs_assigned["%s" % sector]], axis=1
                )
                .replace(np.nan, 0)
                .sum(axis=1)
            )
        output["TOTEMP"] = output.sum(axis=1)
        output = output.rename(columns={"altTAZ": "TAZ"})
        try:
            assert output["TOTEMP"].sum() == remi_jobs_county[2050].sum()
        except AssertionError:
            if abs(output["TOTEMP"].sum() - remi_jobs_county[2050].sum()) < 100:
                pass
            else:
                raise AssertionError

        tqdm.write(f"{output['TOTEMP'].sum()} of {remi_jobs_county[2050].sum()} assigned")
        final_output["%s" % county] = output

    return final_output


# HOUSEHOLD LOCATION CHOICE MODEL

# Note maintain rent surface as formatable modeltype
hh_spec1 = "np.log1p({modeltype}) + np.log1p(HINC) + np.log1p(to_downtown_Oakland) + np.log1p(to_Stanford) + np.log1p(to_Presidio) + np.log1p(TOTEMP_highway_0_40) + np.log1p(TOTEMP_highway_40_80) + np.log1p(TOTEMP_wtransit_0_40) + np.log1p(TOTEMP_wtransit_40_80)"


def estimate_single_household_model(row):
    county, submodel, model_exp, obs_sample_size, taz_sample_size, filename = row
    if "rent" in submodel:
        model_exp = model_exp.format(modeltype="mfr_spatial_const")
    else:
        model_exp = model_exp.format(modeltype="sfr_spatial_const")

    alts_hh = get_taz_alts(county)
    afford_shares = pd.read_csv(dirs["lc"]["input"].joinpath("afford_quartiles.csv")).filter(
        items=[
            "LINK21_TAZ",
            "q1_affordable_share",
            "q2_affordable_share",
            "q3_affordable_share",
            "q4_affordable_share",
        ]
    )
    alts_hh = pd.merge(alts_hh, afford_shares, left_on="altTAZ", right_on="LINK21_TAZ").drop(
        columns="LINK21_TAZ"
    )
    alts_hh.set_index("altTAZ", inplace=True)
    outputfile = {"exp": model_exp}
    # Import households
    obs_hh = pd.read_csv(dirs["lc"]["input"].joinpath("2015_CS_Pop_1215.csv")).rename(
        columns={"TAZ": "chosenTAZ"}
    )
    obs_hh = pd.merge(
        obs_hh,
        pd.read_csv(dirs["lc"]["input"].joinpath("CountyTAZcrosswalk.csv")),
        left_on="chosenTAZ",
        right_on="LINK21_TAZ",
    )
    obs_hh["county"] = obs_hh["GEOID"].astype(str).apply(lambda x: "0" + x)
    # Assign quartile based on 9 county income distribution.
    obs_hh["inc_cat"] = pd.qcut(obs_hh.HINC, 4, labels=False)
    # Filter to county of interest
    obs_hh = obs_hh[obs_hh["county"] == county].drop(columns=["LINK21_TAZ", "GEOID", "HINC"])
    # Remove group quarters
    obs_hh = obs_hh[obs_hh["TEN"].notnull()]
    obs_hh["tenure"] = obs_hh["TEN"].map({1: "own", 2: "own", 3: "rent", 4: "rent"})
    obs_hh["submodel"] = obs_hh["tenure"] + "_q" + (obs_hh["inc_cat"] + 1).astype(str)

    obs_hh = obs_hh[obs_hh["submodel"] == submodel].drop(columns=["county"])

    if obs_sample_size is not None:
        if obs_sample_size <= 1:
            raise NotImplementedError
        elif obs_sample_size > 1:
            if len(obs_hh) > obs_sample_size:
                seed = Seeded(4925)
                with seed:
                    obs_hh = obs_hh.sample(obs_sample_size)
    alt_count = len(alts_hh)
    if taz_sample_size and taz_sample_size < alt_count:
        alt_count = taz_sample_size

    tqdm.write(
        f"{county} {submodel}: {len(obs_hh):,} choosers * {alt_count:,} alternatives"
        f" = {len(obs_hh) * alt_count:,} combinations"
    )
    # Run MultiNomialLogit model
    with HowLong(message=f"{county} {submodel} {alt_count:,} combinations"):
        fitted_model = run_location_choice_mnl(
            obs_hh, alts_hh, model_exp, taz_sample_size
        ).get_raw_results()

    # Converting pd.DataFrame within lc raw results to json
    fitted_model["fit_parameters"] = fitted_model["fit_parameters"].to_json(orient="split")
    outputfile["%s" % submodel] = fitted_model

    # Export raw results for whole county as a json
    check_create_dir(dirs["lc"]["intermediate"].joinpath(f"hh/results/{filename}"))
    with open(
        dirs["lc"]["intermediate"].joinpath(f"hh/results/{filename}", f"{county}-{submodel}.json"),
        "w",
    ) as outfile:
        outfile.write(json.dumps(outputfile, indent=4))


def estimate_household_lc(
    model_exp, counties, obs_sample_size=None, taz_sample_size=None, num_workers=1
):
    """Function runs multinomial logit for households and exports choicemodels.MultinomialLogitResults object as .json"""

    counties = ["06081", "06085", "06075", "06001", "06055", "06095", "06013", "06097", "06041"]
    # counties = ["06081", "06085", "06075", "06001", "06095", "06013", "06097", "06041"]
    submodels = ["own_q1", "own_q2", "own_q3", "own_q4", "rent_q1", "rent_q2", "rent_q3", "rent_q4"]

    exp = [model_exp]
    filename = ["HLCM_Est_" + time.strftime("%m%d_%I-%M%p")]
    models = list(
        itertools.product(counties, submodels, exp, [obs_sample_size], [taz_sample_size], filename)
    )

    if num_workers == 1:
        for model_args in tqdm(models, total=len(models)):
            estimate_single_household_model(model_args)
    else:
        pool = multiprocessing.Pool(processes=num_workers)
        try:
            with tqdm(total=len(models)) as pbar:
                for _ in pool.imap_unordered(estimate_single_household_model, models):
                    pbar.update()
        finally:
            pool.close()
            pool.join()
    return


def get_gqpop_shares():
    """Utility funtion for calculating distribtuion of GQ for each county"""
    # Import base year synthetic population
    obs_hh = pd.read_csv(dirs["lc"]["input"].joinpath("2015_CS_Pop_1215.csv")).rename(
        columns={"TAZ": "chosenTAZ"}
    )
    obs_hh = pd.merge(
        obs_hh,
        pd.read_csv(dirs["lc"]["input"].joinpath("CountyTAZcrosswalk.csv")),
        left_on="chosenTAZ",
        right_on="LINK21_TAZ",
    )
    obs_hh = obs_hh.assign(county=obs_hh["GEOID"].astype(str).apply(lambda x: "0" + x)).drop(
        columns=["GEOID", "LINK21_TAZ"]
    )
    # Keep GQ population and summarize at county level
    gqpop = obs_hh[obs_hh["TEN"].isnull()]
    gqpop = gqpop.pivot_table(
        index=["chosenTAZ", "county"], values="HHID", aggfunc="count"
    ).reset_index()
    # Calculate share of GQ for each TAZ (by county) and return with shares and absolute values
    gqpop["share"] = gqpop["HHID"] / gqpop.groupby("county")["HHID"].transform(sum)
    return gqpop


def resample_population(scenario_name, write_out=True):
    """Utility function that resamples base year synthetic households to match Headship Model HH counts and quartile distbution"""
    # Import processed REMI outputs
    remi_headship_output = pd.read_csv(
        dirs["scenarios"]["input"].joinpath(scenario_name, "REMI_output.csv")
    )
    remi_headship_output = remi_headship_output.assign(
        county=remi_headship_output["county"].astype(str).apply(lambda x: "0" + x)
    ).set_index("county")

    # Import base year household year population
    obs_hh = pd.read_csv(dirs["lc"]["input"].joinpath("2015_CS_Pop_1215.csv")).rename(
        columns={"TAZ": "chosenTAZ"}
    )
    # Merge county crosswalk and subset for county of interest
    obs_hh = pd.merge(
        obs_hh,
        pd.read_csv(dirs["lc"]["input"].joinpath("CountyTAZcrosswalk.csv")),
        left_on="chosenTAZ",
        right_on="LINK21_TAZ",
        how="left",
    )
    obs_hh = obs_hh.assign(county=obs_hh["GEOID"].astype(str).apply(lambda x: "0" + x)).drop(
        columns=["GEOID", "LINK21_TAZ"]
    )
    obs_hh["quartile"] = pd.qcut(obs_hh["HINC"], 4, labels=False)
    obs_hh["allocation_code"] = "base_pop"
    obs_hh[obs_hh["TEN"].notnull()]

    for remi_county in tqdm(remi_headship_output.index.tolist()):
        tqdm.write(f"Resampling population for {remi_county}")
        remi_hhcount = remi_headship_output.loc[remi_county]["hhtot"]
        quartile_shares = remi_headship_output.loc[remi_county].drop("hhtot").to_dict()
        obs_hh_county = obs_hh[obs_hh["county"] == remi_county]
        # Remove GQ population
        obs_hh_county = obs_hh_county[obs_hh_county["TEN"].notnull()]
        assert len(obs_hh_county) <= remi_hhcount
        county_hhtot = []

        for quartile in tqdm(quartile_shares):
            # For each quartile, sample new 2050 households from 2015 households
            # this sample is representative of the households in that quartile with respect to tenure
            # Assumption: County hh growth is never greater than 100%
            tqdm.write(f"Generating population for quartile {int(quartile)+1}")
            obs_hh_quart = obs_hh_county[obs_hh_county["quartile"] == int(quartile)].copy()
            hhmargin = round(remi_hhcount * quartile_shares[quartile]) - len(obs_hh_quart)
            # Reduce basepop if quartile growth is negative
            if hhmargin < 0:
                newpop_quart = obs_hh_quart.sample(n=(len(obs_hh_quart) + hhmargin), replace=False)
                assert len(newpop_quart) == (remi_hhcount * quartile_shares[quartile])
                county_hhtot.append(newpop_quart)
            elif hhmargin == 0:
                county_hhtot.append(obs_hh_quart)
            else:
                s = Seeded(3849)
                with s:
                    addhh = obs_hh_quart.sample(n=hhmargin, replace=True)
                addhh["allocation_code"] = "resample_pop"
                newpop_quart = pd.concat([obs_hh_quart, addhh], axis=0)
                assert len(newpop_quart) == (remi_hhcount * quartile_shares[quartile])
                county_hhtot.append(newpop_quart)

        county_hhtot = pd.concat(county_hhtot, axis=0)
        county_hhtot = county_hhtot.assign(newind=range(len(county_hhtot))).set_index("newind")
        county_hhtot.index.name = None

        check_create_dir(
            dirs["scenarios"]["intermediate"].joinpath(scenario_name, "Resampled2050Pop"),
        )
        county_hhtot.to_parquet(
            (
                dirs["scenarios"]["intermediate"].joinpath(
                    scenario_name, "Resampled2050Pop", f"{remi_county}.parquet"
                )
            ),
        )
        assert len(county_hhtot) == remi_hhcount

    return print(">>> Synthetic Population resampled and exported")


def distribute_gq_obs(gqpop, remicounty):
    """Utility function that heuristically assigns horzion year GQ population to match Headship Model"""
    shares = get_gqpop_shares()
    shares = shares[shares["county"] == remicounty].set_index("chosenTAZ")
    output = shares["share"] * gqpop
    output = output.round()
    # assert output.sum() == gqpop
    return output


def run_hlcm_simulation(scenario_name, taz_alts_scenario):
    """Imports resampled horizon year househods, creates a relocation pool and outputs final TAZ assignment

    NOTE: This is a draft, incomplete function!
    """
    counties = ["06081", "06085", "06075", "06001", "06055", "06095", "06013", "06097", "06041"]
    # Set relocation rate for future year
    relocation_rate = 0.2
    final_output = {}

    afford_shares = pd.read_csv(dirs["lc"]["input"].joinpath("afford_quartiles.csv")).filter(
        items=[
            "LINK21_TAZ",
            "q1_affordable_share",
            "q2_affordable_share",
            "q3_affordable_share",
            "q4_affordable_share",
        ]
    )
    # Import TAZ alternatives in county
    taz_alts_scenario = pd.merge(
        taz_alts_scenario, afford_shares, left_on="altTAZ", right_on="LINK21_TAZ"
    ).drop(columns="LINK21_TAZ")
    taz_alts_scenario = taz_alts_scenario.set_index("altTAZ")

    for county in tqdm(counties):
        tqdm.write(f"Assigning for {county}")
        taz_alts_county = taz_alts_scenario[taz_alts_scenario["county"] == county].copy()
        # Creating separate capacity columns that can be adjsuted based on deed restricted units and individual simulation iteration
        taz_alts_county["capacity_rent"] = taz_alts_county["residential_units_rent"]
        taz_alts_county["capacity_own"] = taz_alts_county["residential_units_own"]
        pop50 = (
            pq.read_pandas(
                dirs["scenarios"]["intermediate"].joinpath(
                    f"{scenario_name}/Resampled2050Pop/{county}.parquet"
                )
            )
            .to_pandas()
            .drop(columns="HINC")
        )
        # Classify household for unique relocation model
        pop50["tenure"] = pop50["TEN"].map({1: "own", 2: "own", 3: "rent", 4: "rent"})
        pop50["result_code"] = (
            pop50["tenure"] + "_q" + ((pop50["quartile"] + 1).astype(str))
        )  # required for simulation probs_callable

        # Ensure there is appropriate capacity in the county, if not, scale up units
        total_supply_own = (
            taz_alts_county[
                ["residential_units_own", "vli_units_own", "li_units_own", "mi_units_own"]
            ]
            .sum()
            .sum()
        )
        total_supply_rent = (
            taz_alts_county[
                ["residential_units_rent", "vli_units_rent", "li_units_rent", "mi_units_rent"]
            ]
            .sum()
            .sum()
        )
        total_demand_own = len(pop50[pop50["tenure"] == "own"])
        total_demand_rent = len(pop50[pop50["tenure"] == "rent"])

        if total_supply_own < total_demand_own:
            multiplier = (total_demand_own / total_supply_own) * 1.05
            taz_alts_county = taz_alts_county.assign(
                residential_units_own=(
                    taz_alts_county["residential_units_own"] * multiplier
                ).round(),
                vli_units_own=(taz_alts_county["vli_units_own"] * multiplier).round(),
                li_units_own=(taz_alts_county["li_units_own"] * multiplier).round(),
                mi_units_own=(taz_alts_county["mi_units_own"] * multiplier).round(),
            )
            total_supply_own = (
                taz_alts_county[
                    ["residential_units_own", "vli_units_own", "li_units_own", "mi_units_own"]
                ]
                .sum()
                .sum()
            )
            assert total_supply_own >= total_demand_own

        if total_supply_rent < total_demand_rent:
            multiplier = (total_demand_rent / total_supply_rent) * 1.05
            taz_alts_county = taz_alts_county.assign(
                residential_units_rent=(
                    taz_alts_county["residential_units_rent"] * multiplier
                ).round(),
                vli_units_rent=(taz_alts_county["vli_units_rent"] * multiplier).round(),
                li_units_rent=(taz_alts_county["li_units_rent"] * multiplier).round(),
                mi_units_rent=(taz_alts_county["mi_units_rent"] * multiplier).round(),
            )
            total_supply_rent = (
                taz_alts_county[
                    ["residential_units_rent", "vli_units_rent", "li_units_rent", "mi_units_rent"]
                ]
                .sum()
                .sum()
            )
            assert total_supply_rent >= total_demand_rent

        del total_supply_own, total_supply_rent, total_demand_own, total_demand_rent

        # Create relocating and staying population from base year and adjust TAZ capacity
        sampling_seed = Seeded(90782)
        basepop = pop50[pop50["allocation_code"] == "base_pop"]
        with sampling_seed:
            stayers = basepop.sample(frac=1 - relocation_rate)
        relocators = basepop.loc[basepop.index.difference(stayers.index)]

        # Adjust capacities based on tenure for households that do not move
        staying_hh = stayers.pivot_table(
            index="chosenTAZ", columns="tenure", values="HHID", aggfunc="count"
        ).rename(columns={"HHID": "adjustment"})
        staying_hh.columns.name, staying_hh.index.name = None, None
        stayers_final = []
        relocation_adjusted = []

        # Check that all stayers have units to stay in, if not switch them into the relocation pool
        taz_alts_county = pd.concat([taz_alts_county, staying_hh], axis=1)
        taz_alts_county = taz_alts_county.assign(
            capacity_own=lambda x: x.capacity_own - x.own,
            capacity_rent=lambda x: x.capacity_rent - x.rent,
        ).drop(columns=["own", "rent"])

        # Adjust rent space TAZ capacities
        capacity_adjustments_rent = pd.concat(
            [staying_hh[["rent"]], taz_alts_county["residential_units_rent"]],
            axis=1,
        ).replace(np.nan, 0)
        capacity_adjustments_rent = capacity_adjustments_rent.assign(
            overcapacity=lambda x: x.residential_units_rent - x.rent,
        )
        capacity_adjustments_rent["overcapacity"] = capacity_adjustments_rent["overcapacity"].apply(
            lambda x: abs(x) if (x < 0) else 0
        )
        capacity_adjustments_rent = capacity_adjustments_rent.assign(
            rent=lambda x: x.rent - x.overcapacity
        )

        assert (
            capacity_adjustments_rent["rent"].sum()
            < capacity_adjustments_rent["residential_units_rent"].sum()
        )

        for taz in (
            capacity_adjustments_rent[capacity_adjustments_rent["overcapacity"] > 0]
            .reset_index()["index"]
            .to_list()
        ):
            # Operate on the TAZ with issue and resolve capacity conflits
            taz_stayers = stayers[stayers["chosenTAZ"] == taz].copy()
            taz_stayers = taz_stayers[taz_stayers["tenure"] == "rent"]
            # Remove these TAZ renters from county stayers df to maintain well defined "correct/corrected" pools
            stayers = stayers.loc[stayers.index.difference(taz_stayers.index)]
            with sampling_seed:
                new_stayers = taz_stayers.sample(
                    n=int(capacity_adjustments_rent.loc[taz]["rent"]), replace=False
                )
            new_relocators = taz_stayers.loc[taz_stayers.index.difference(new_stayers.index)]
            assert (len(new_relocators) + len(new_stayers)) == len(taz_stayers)
            stayers_final.append(new_stayers)
            relocation_adjusted.append(new_relocators)

        # Adjust own space TAZ capacities
        capacity_adjustments_own = pd.concat(
            [staying_hh[["own"]], taz_alts_county["residential_units_own"]],
            axis=1,
        ).replace(np.nan, 0)
        capacity_adjustments_own = capacity_adjustments_own.assign(
            overcapacity=lambda x: x.residential_units_own - x.own,
        )
        capacity_adjustments_own["overcapacity"] = capacity_adjustments_own["overcapacity"].apply(
            lambda x: abs(x) if (x < 0) else 0
        )
        capacity_adjustments_own = capacity_adjustments_own.assign(
            own=lambda x: x.own - x.overcapacity
        )

        assert (
            capacity_adjustments_own["own"].sum()
            < capacity_adjustments_own["residential_units_own"].sum()
        )

        for taz in (
            capacity_adjustments_own[capacity_adjustments_own["overcapacity"] > 0]
            .reset_index()["index"]
            .to_list()
        ):
            # Operate on the TAZ with issue and resolve capacity conflits
            taz_stayers = stayers[stayers["chosenTAZ"] == taz].copy()
            taz_stayers = taz_stayers[taz_stayers["tenure"] == "own"]
            # Remove these TAZ owners from county stayers df to maintain well defined "correct/corrected" pools
            stayers = stayers.loc[stayers.index.difference(taz_stayers.index)]
            with sampling_seed:
                new_stayers = taz_stayers.sample(
                    n=int(capacity_adjustments_own.loc[taz]["own"]), replace=False
                )
            new_relocators = taz_stayers.loc[taz_stayers.index.difference(new_stayers.index)]
            assert (len(new_relocators) + len(new_stayers)) == len(taz_stayers)
            stayers_final.append(new_stayers)
            relocation_adjusted.append(new_relocators)

        # Available capacity adjusted stayers
        stayers_final = pd.concat([pd.concat(stayers_final, axis=0), stayers], axis=0)

        # for_assignment = adjusted relocators + original relocators + new population
        relocation_adjusted = pd.concat(relocation_adjusted, axis=0)
        for_assignment = pd.concat(
            [relocators, relocation_adjusted, pop50[pop50["allocation_code"] != "base_pop"]], axis=0
        )
        tqdm.write(
            f"{len(relocation_adjusted)} of {int(len(basepop)*(1-relocation_rate))} stayers moved back to relocation pool"
        )
        # Assertion 1: after relocation adjustments, jobs into stayers_final+relocation_adjusted == original_stayers
        assert len(stayers_final) + len(relocation_adjusted) == int(
            len(basepop) * (1 - relocation_rate)
        )
        # Assertion 2: total assigned+for_assignment matches REMI input/control
        assert len(stayers_final) + len(for_assignment) == len(pop50)

        # Compute actual capacity adjustments
        staying_hh = stayers_final.pivot_table(
            index="chosenTAZ", columns="tenure", values="HHID", aggfunc="count"
        ).rename(columns={"HHID": "adjustment"})
        staying_hh.columns.name, staying_hh.index.name = None, None

        taz_alts_county = taz_alts_county.assign(
            residential_units_rent=taz_alts_county["residential_units_rent"] - staying_hh["rent"],
            residential_units_own=taz_alts_county["residential_units_own"] - staying_hh["own"],
            capacity_rent=taz_alts_county["residential_units_rent"],
            capacity_own=taz_alts_county["residential_units_own"],
        )

        assert (
            not taz_alts_county["capacity_rent"].lt(0).any()
            & taz_alts_county["capacity_own"].lt(0).any()
        )

        def mct_callable(obs, alts, intx_ops=None):
            return choicemodels.tools.MergedChoiceTable(obs, alts, sample_size=None)

        def probs_callable(mct, county="%s" % county):
            df = mct.to_frame()
            mct.sample_size = len(df.groupby(level=1))
            hh_code = df["result_code"].unique().tolist()[0]
            model_dir = dirs["lc"]["intermediate"].joinpath("HH")
            fitted_model = read_predicted_model(model_dir, county, hh_code)
            return fitted_model.probabilities(mct)

        hhten = ["rent", "own"]
        results = []

        for quartile in tqdm(range(3, -1, -1)):
            assign_quartile = for_assignment[for_assignment["quartile"] == quartile].copy()

            if quartile == 2:
                # Add 80-120% AMI units
                taz_alts_county = taz_alts_county.assign(
                    capacity_rent=taz_alts_county["capacity_rent"]
                    + taz_alts_county["mi_units_rent"],
                    capacity_own=taz_alts_county["capacity_own"] + taz_alts_county["mi_units_own"],
                )

            if quartile == 1:
                # Add 50-80% AMI units
                taz_alts_county = taz_alts_county.assign(
                    capacity_rent=taz_alts_county["capacity_rent"]
                    + taz_alts_county["li_units_rent"],
                    capacity_own=taz_alts_county["capacity_own"] + taz_alts_county["li_units_own"],
                )

            if quartile == 0:
                # Add 0-50% AMI units
                taz_alts_county = taz_alts_county.assign(
                    capacity_rent=taz_alts_county["capacity_rent"]
                    + taz_alts_county["vli_units_rent"],
                    capacity_own=taz_alts_county["capacity_own"] + taz_alts_county["vli_units_own"],
                )

            for status in tqdm(hhten):
                tqdm.write(f"Assigning Q{quartile+1} {status}ers")
                assign_ten_quart = assign_quartile[assign_quartile["tenure"] == status]
                taz_alts_county["capacity"] = taz_alts_county[f"capacity_{status}"].copy()
                simSeed = Seeded(1761)

                with simSeed:
                    sim_out = choicemodels.tools.iterative_lottery_choices(
                        assign_ten_quart,
                        taz_alts_county,
                        mct_callable,
                        probs_callable,
                        chooser_batch_size=10000,
                        alt_capacity="capacity",
                    )
                results.append(sim_out)
                import pdb

                pdb.set_trace()
                # Remove assigned spaces from capacity
                space_correction = (
                    sim_out.reset_index().assign(count=1).groupby("altTAZ").sum("count")
                )
                space_correction.index.name = None
                space_correction = pd.concat(
                    [taz_alts_county[f"capacity_{status}"], space_correction], axis=1
                ).replace(np.nan, 0)
                taz_alts_county[f"capacity_{status}"] = (
                    space_correction[f"capacity_{status}"] - space_correction["count"]
                )

        results = pd.concat(results, axis=0)
        results = (
            pop50.merge(results, left_index=True, right_index=True)
            # pop50.reset_index().merge(results.reset_index(), on="index").drop(columns=["index"])
        )
        results = results.rename(columns={"altTAZ": "TAZ"})
        if len(results) != len(pop50):
            print(f"\n\n WARNING: {len(pop50) - len(results)} county households unassigned")

        results = pd.concat([results, stayers.rename(columns={"chosenTAZ": "TAZ"})], axis=0)

        # Pivot to summarry table with counts for each TAZ and TM1.5 code
        results["households"] = 1

        assigned = pd.pivot_table(
            results, index="TAZ", columns="result_code", aggfunc=sum, values="households"
        ).replace(np.nan, 0)
        characteristics = (
            results.filter(items=["TAZ", "PERSONS", "hworkers", "HHT", "VEHICL"])
            .groupby("TAZ")
            .sum()
        )
        # TODO: Add persons binned variable for age grps soon

        # The MTC file was originally indexed for 1454 TAZs for TM1. It has been merged with crosswalk!
        # mtc_multipliers = pd.read_csv(dirs['lc']["input"].joinpath("zone_forecast_inputs.csv"))
        # characteristics[]
        # TODO: Figure out how to import gq numbers from REMI
        # characteristics["gqpop"] = distribute_gq_obs(180900, "06081").replace(np.nan, 0)
        output = pd.concat([assigned, characteristics], axis=1)

        final_output["%s" % county] = output
    return final_output


def run_test_simulation_elcm(relocation_rate=None):
    """Function simulates Data Axle jobs using estimation results in jobs/Results and create summary file for Shiny input"""
    # Data axle jobs distributions, static file does not change"
    # TODO: Recreate base_dist before running further tests
    real_counts = pd.read_csv("../shinyapp/ELCM/base_dist.csv").rename(columns={"2050": 2050})
    real_counts["county"] = "0" + real_counts["county"].astype(str)
    taz_alts = prepare_taz_data(scenario_name="base_year")
    # Running simulaiton based on results injobs/Results
    sim_out = run_elcm_simulation(real_counts, taz_alts, relocation_rate)
    taz_output = pd.concat(sim_out, axis=0)
    taz_output = taz_output.reset_index().rename(columns={"level_0": "county"})
    # Preparing coefficient summaries
    print("Simulation complete.. preparing estimation coefficient summary file")
    taz_coefs = summarize_results("jobs").reset_index()
    check_create_dir(data_dir.joinpath("outputs/LocationChoiceEstimation/"))
    filename = "ELCM_Sim_" + time.strftime("%m%d_%I-%M%p")
    with pd.ExcelWriter(
        data_dir.joinpath(f"outputs/LocationChoiceEstimation/{filename}.xlsx")
    ) as writer:
        taz_output.to_excel(writer, sheet_name="Sheet 1", index=False)
        taz_coefs.to_excel(writer, sheet_name="Sheet 2", index=False)
    return print("File written to outputs folder")


if __name__ == "__main__":
    prepare_data_axle()
    taz_alts = prepare_taz_data(scenario_name="base_year")

    # estimate_employment_lc(emp_spec3, obs_sample_size=10000, taz_sample_size=100, num_workers=1)

    relocation_rate_list = [
        None,
        {
            "AGREMPN": 0.5,
            "OTHEMPN": 0.5,
            "MWTEMPN": 0.5,
            "RETEMPN": 0.5,
            "FPSEMPN": 0.5,
            "HEREMPN": 0.5,
        },
        {
            "AGREMPN": 0.8,
            "OTHEMPN": 0.8,
            "MWTEMPN": 0.8,
            "RETEMPN": 0.8,
            "FPSEMPN": 0.8,
            "HEREMPN": 0.8,
        },
    ]

    # for i in relocation_rate_list:
    #     run_test_simulation_elcm(i)
