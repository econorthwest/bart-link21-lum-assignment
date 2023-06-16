install.packages("pacman")
library(pacman)
p_load(tidyverse, tidycensus, tigris, sf, mapview, rio)
rm(list = ls())
gc()


# tidycensus::census_api_key("your_key",install = T)
## If you do not have one, request an api key here:  https://api.census.gov/data/key_signup.html

baycounties <- c(
  "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
  "San Mateo", "Santa Clara", "Solano", "Sonoma"
)

bayfips <- c("06001", "06013", "06041", "06055", "06075", "06081", "06085", "06097", "06095")



##################################################
##  PreprocessA: Headship  ##
##################################################
# File 1: PUMA to Counties relations (for 2010 geographies)

## I start with tracts as they are necessarily nested within PUMAs which enable a centeriod based crosswalk (rather than intersection based)

pt <- pumas(state = "CA", year = 2013, cb = T) %>%
  st_make_valid()
tt <- tracts(state = "CA", year = "2016")

st_centroid(tt) %>%
  st_join(pt) %>%
  mutate(cnt_id = str_sub(GEOID, 1, 5)) %>%
  distinct(cnt_id, puma_id = GEOID10) %>%
  filter(!is.na(puma_id)) %>%
  export("~/projects/24182-bart-link21-modeling/data/inputs/HeadshipModel/2010_County_to_2010_PUMA.csv")


## checking that Bay Area PUMAs and Counties are consistent
left_join(pt, crosswalk, by = c("GEOID10" = "puma_id")) %>%
  filter(cnt_id %in% bayfips) %>%
  mapView(zcol = "cnt_id") + mapView(cnt) %>%
  # Files 2-8: PUMS data

  years() <- c(2012, 2013, 2014, 2018)

## 1 year ACS 2012-18
for (t in years) {
  get_pums(
    variables = c("PUMA", "AGEP", "RAC1P", "HISP", "ST", "SEX", "RELP"),
    state = "CA", survey = "acs1", year = t
  ) %>%
    export(paste0("pums_", t, ".csv"))
}

## 5 year ACS 12-16, 14-18
get_pums(
  variables = c("PUMA", "AGEP", "RAC1P", "HISP", "ST", "SEX", "RELP"),
  state = "CA", survey = "acs5", year = 2016
) %>%
  export(paste0("pums_12-16.csv"))

get_pums(
  variables = c("PUMA", "AGEP", "RAC1P", "HISP", "ST", "SEX", "RELP"),
  state = "CA", survey = "acs5", year = 2018
) %>%
  export(paste0("pums_14-18.csv"))



##################################################
##  PreprocessB: 2000 Census Headship  ##
##################################################




# 2000s data not avaialable on tidy census. Pulled from: https://www2.census.gov/census_2000/datasets/PUMS/FivePercent/

# Cross walk
puma <- st_read("~/Downloads/fe_2007_06_puma500")
county <- st_read("~/Downloads/fe_2007_06_county00")


puma %>%
  st_intersection(county %>% select(cnt_id = CNTYIDFP00)) %>%
  st_make_valid() %>%
  mutate(intarea = st_area(.)) %>%
  group_by(cnt_id) %>%
  mutate(cntarea = sum(intarea, na.rm = T)) %>%
  ungroup() %>%
  mutate(ratio = as.numeric(intarea) / as.numeric(cntarea)) %>%
  filter(ratio < .001) %>%
  mapview()

# There are no intersection sliver errors so I am able to directly export the intersection
puma %>%
  st_intersection(county) %>%
  st_make_valid() %>%
  st_set_geometry(NULL) %>%
  select(cnt_id = CNTYIDFP00, puma_id = PUMA5ID00) %>%
  export("~/projects/24182-bart-link21-modeling/data/inputs/HeadshipModel/2000_County_to_2000_PUMA.csv")


### 2010 PUMA to county crosswalk to create GQ share file

# Counties of interest in Bay Area
county_vector <- c("001", "013", "041", "055", "075", "081", "085", "095", "097")

# Import county geographies
bay_cnty <- counties(state = "CA", year = 2011) %>%
  filter(COUNTYFP %in% county_vector)

mapview(bay_cnty)

# county-region crosswalk
region_crosswalk <- bay_cnty %>%
  st_drop_geometry() %>%
  select(GEOID) %>%
  mutate(region = case_when(
    GEOID %in% c("06001", "06013") ~ "East Bay",
    GEOID %in% c("06041", "06075", "06081") ~ "West Bay",
    GEOID %in% c("06055", "06095", "06097") ~ "North Bay",
    GEOID == "06085" ~ "South Bay"
  ))


# Import PUMA geographies

bay_pumas <- pumas(state = "CA", year = 2012) %>%
  st_centroid()


puma_crosswalk <- st_intersection(bay_cnty, bay_pumas) %>%
  st_drop_geometry() %>%
  filter(COUNTYFP %in% county_vector) %>%
  select("county" = GEOID, "puma" = GEOID10)


##################################################
##  HHPROJ-B: Income Category shares.  ##
##################################################


county <- counties(state = "CA", year = 2021)
pumas <- pumas(state = "CA", year = 2021)

crosswalk21 <- county %>%
  filter(GEOID %in% bayfips) %>%
  st_intersection(pumas %>% st_centroid()) %>%
  st_drop_geometry() %>%
  select(GEOID, PUMACE10)


pums2021 <- get_pums(
  variables = c("PUMA", "HINCP", "ADJINC", "TEN"),
  state = "CA", survey = "acs5", year = 2021
)



pums2021 %>%
  right_join(crosswalk21, by = c("PUMA" = "PUMACE10")) %>%
  filter(
    PUMA %in% (crosswalk21$PUMACE10 %>% unique()),
    !is.na(TEN)
  ) %>%
  mutate(
    hhinc_adj = HINCP * as.numeric(ADJINC),
    quartile = ntile(hhinc_adj, 4)
  ) %>%
  group_by(GEOID) %>%
  mutate(totpop = sum(WGTP)) %>%
  group_by(GEOID, quartile) %>%
  summarize(
    totpop = mean(totpop),
    qtpop = sum(WGTP)
  ) %>%
  mutate(share = qtpop / totpop)
