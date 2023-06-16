bayarearegions = {
    "06001": "East Bay",
    "06013": "East Bay",
    "06085": "South Bay",
    "06055": "North Bay",
    "06081": "West Bay",
    "06075": "West Bay",
    "06097": "North Bay",
    "06041": "West Bay",
    "06095": "North Bay",
}

north_bay = {"06055": "North Bay", "06097": "North Bay", "06095": "North Bay"}


remiregions = {
    "Alameda": "East Bay",
    "Contra Costa": "East Bay",
    "Marin": "West Bay",
    "Napa": "North Bay",
    "San Francisco": "West Bay",
    "San Mateo": "West Bay",
    "Santa Clara": "South Bay",
    "Solano": "North Bay",
    "Sonoma": "North Bay",
}

# map from quarter to first month in quarter

qtrmap = {1: "01", 2: "04", 3: "07", 4: "10"}

yr_to_vintage = {
    1990: "C1990",
    1991: "C1990",
    2000: "C2000",
    2005: "ACS2005",
    2006: "ACS2006",
    2007: "ACS2007",
    2008: "ACS2008",
    2009: "ACS2009",
    2010: "ACS2010",
    2011: "ACS2011",
    2012: "ACS2012",
    2013: "ACS2013",
    2014: "ACS2014",
    2015: "ACS2015",
    2016: "ACS2016",
    2017: "ACS2017",
    2018: "ACS2018",
}

# mapping to SOC
# occ_det_soc_map = {
#     "occ_det_mgmt": "11-0000",  # mgmt
#     "occ_det_biz": "13-0000",  # Business and Financial Operations Occupations
#     "occ_det_comp": "15-0000",  # Computer and Mathematical Occupations
#     "occ_det_community": "21-0000",  # Community and Social Service Occupations
#     "occ_det_lgl": "23-0000",  # legal
#     "occ_det_hlthsup": "31-0000",  # Healthcare Support Occupations
#     "occ_det_foodprep": "35-0000",  # Food Preparation and Serving Related Occupations
#     "occ_det_officeadmin": "43-0000",  # Office and Administrative Support Occupations
# }

occ_det_soc_map = {
    "11-0000": "occ_det_mgmt",
    "13-0000": "occ_det_biz",
    "15-0000": "occ_det_comp",
    "21-0000": "occ_det_community",
    "23-0000": "occ_det_lgl",
    "31-0000": "occ_det_hlthsup",
    "35-0000": "occ_det_foodprep",
    "43-0000": "occ_det_officeadmin",
}

# relate naics categories to abag classification

naics_map = {
    "Forestry, Fishing, and Related Activities": "Agriculture & Natural Resources",
    "Mining": "Agriculture & Natural Resources",
    "Utilities": "Transportation & Utilities",
    "Construction": "Construction",
    "Manufacturing": "Manufacturing & Wholesale",
    "Wholesale Trade": "Manufacturing & Wholesale",
    "Retail Trade": "Retail Trade",
    "Transportation and Warehousing": "Transportation & Utilities",
    "Information": "Information",
    "Finance and Insurance": "Financial & Leasing",
    "Real Estate and Rental and Leasing": "Professional & Managerial Services",
    "Professional, Scientific, and Technical Services": "Professional & Managerial Services",
    "Management of Companies and Enterprises": "Professional & Managerial Services",
    "Administrative and Waste Management Services": "Professional & Managerial Services",
    "Educational Services; private": "Health & Educational Services",
    "Health Care and Social Assistance": "Health & Educational Services",
    "Arts, Entertainment, and Recreation": "Arts, Recreation & Other Services",
    "Accommodation and Food Services": "Arts, Recreation & Other Services",
    "Other Services, except Public Administration": "Arts, Recreation & Other Services",
    "State and Local Government Employment": "Government",
    "Federal Civilian Employment": "Government",
    "Federal Military Employment": "Government",
}

# these map to descriptions in the SOC file so once we match to that, we can summarize by these categories
occ_to_census_upd = {
    "management occupations": "occ_mgmt",
    "business and financial operations occupations": "occ_mgmt",
    "computer and mathematical occupations": "occ_mgmt",
    "architecture and engineering occupations": "occ_mgmt",
    "life, physical, and social science occupations": "occ_mgmt",
    "community and social service occupations": "occ_svcs",
    "legal occupations": "occ_mgmt",
    "education, training, and library occupations": "occ_svcs",
    "arts, design, entertainment, sports, and media occupations": "occ_svcs",
    "healthcare practitioners and technical occupations": "occ_svcs",
    "healthcare support occupations": "occ_svcs",
    "protective service occupations": "occ_prod",
    "food preparation and serving related occupations": "occ_sls",
    "building and grounds cleaning and maintenance occupations": "occ_svcs",
    "personal care and service occupations": "occ_svcs",
    "sales and related occupations": "occ_sls",
    "office and administrative support occupations": "occ_sls",
    "farming, fishing, and forestry occupations": "occ_nat",
    "construction and extraction occupations": "occ_nat",
    "installation, maintenance, and repair occupations": "occ_prod",
    "production occupations": "occ_prod",
    "transportation and material moving occupations": "occ_prod",
    "military specific occupations": "occ_other",
}


# this table is custom generated in REMI--mxing in govt as industry employment

indus_to_census = {
    "State and Local Government": "ind_public",
    "Federal Civilian": "ind_public",
    "Federal Military": "ind_public",
    "Forestry, Fishing, and Related Activities": "ind_other",
    "Mining": "ind_other",
    "Utilities": "ind_other",
    "Construction": "ind_other",
    "Manufacturing": "ind_other",
    "Wholesale Trade": "ind_other",
    "Retail Trade": "ind_retail",
    "Transportation and Warehousing": "ind_other",
    "Information": "ind_prof",
    "Finance and Insurance": "ind_prof",
    "Real Estate and Rental and Leasing": "ind_prof",
    "Professional, Scientific, and Technical Services": "ind_prof",
    "Management of Companies and Enterprises": "ind_prof",
    "Administrative and Waste Management Services": "ind_other",
    "Administrative, Support, Waste Management, And Remediation Services": "ind_other",
    "Educational Services": "ind_educ",
    "Educational Services; private": "ind_educ",
    "Health Care and Social Assistance": "ind_health",
    "Arts, Entertainment, and Recreation": "ind_other",
    "Accommodation and Food Services": "ind_accom_food_svcs",
    "Other Services, except Public Administration": "ind_other",
    "Other Services": "ind_other",
}


ind_prof_map = {
    "Motion picture and sound recording industries": "ind_prof",
    "Publishing industries, except Internet": "ind_prof",
    "Administrative and support services": "ind_prof",
    "Data processing, hosting, and related services; Other information services": "ind_prof",
    "Broadcasting, except Internet": "ind_prof",
    "Telecommunications": "ind_prof",
    "Monetary authorities - central bank; Credit intermediation and related activities": "ind_prof",
    "Securities, commodity contracts, other investments; Funds, trusts, other financial vehicles": "ind_prof",
    "Insurance carriers and related activities": "ind_prof",
    "Real estate": "ind_prof",
    "Rental and leasing services; Lessors of nonfinancial intangible assets": "ind_prof",
    "Professional, scientific, and technical services": "ind_prof",
    "Management of companies and enterprises": "ind_prof",
}


# mapping that collapses finer to coarser age bins in the remi output

agemapcollapser = {
    "Ages 0-15": "Ages 0-15",
    "Ages 16-19": "Ages 16-19",
    "Ages 20-21": "Ages 20-24",
    "Ages 22-24": "Ages 20-24",
    "Ages 25-29": "Ages 25-34",
    "Ages 30-34": "Ages 25-34",
    "Ages 35-44": "Ages 35-44",
    "Ages 45-54": "Ages 45-54",
    "Ages 55-59": "Ages 55-64",
    "Ages 60-61": "Ages 55-64",
    "Ages 62-64": "Ages 55-64",
    "Ages 65-69": "Ages 65+",
    "Ages 70-74": "Ages 65+",
    "Ages 75+": "Ages 65+",
}
