import numpy as np


def agebreaker(breaks):
    labels = []
    for f in range(len(breaks) - 1):
        labels.append("Ages {fr}-{to}".format(fr=breaks[f], to=breaks[f + 1] - 1))
    labels[-1] = "Ages {dt:,.0f}+".format(dt=breaks[-2])
    return labels


def agebreaker2(breaks):
    labels = []
    for f in range(len(breaks) - 1):
        labels.append("Ages {fr:.0f}-{to:.0f}".format(fr=breaks[f], to=breaks[f + 1] - 1))
    labels[-1] = "Ages {dt:,.0f}+".format(dt=breaks[-2])
    return labels


def agebreaker_mtc(breaks):
    labels = []
    for f in range(len(breaks) - 1):
        labels.append("age{fr:02}{to:02}".format(fr=breaks[f], to=breaks[f + 1] - 1))
    labels[-1] = "age{dt:}p".format(dt=breaks[-2])
    return labels


def rounder(s, precision=3):
    factor = 10**precision
    out = np.round(s / 10**precision, 0) * factor
    return out


def classifier(df):
    x = df.iloc[:4].tolist()
    try:
        out = next(s for s in x if not s is np.NaN)  # noqa: E714
    except:  # noqa: E722
        out = None
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
    except:  # noqa: E722
        return "none"


# helper function to adjust the wharton index at a passed rate, implying either more or less hardship in the future


def adjust_wharton_variable(rv_wharton_geo_input, regionrates=None):
    rv_wharton_geo_t = rv_wharton_geo_input.T.copy()

    for reg, rates in regionrates.items():
        # grow the last value (2014) with 1% per year to 2040
        subreg = rv_wharton_geo_t.loc[:, reg]
        subreg_last = subreg.loc[2014]
        subreg.loc[2015:2051] = subreg_last * np.multiply.accumulate(
            np.full_like(np.arange(36, dtype=float), fill_value=rates)
        )

        # move back to orig dataframe
        # rv_wharton_geo_t.loc[:,reg]=subreg
        rv_wharton_geo_t.update(subreg)
    rv_wharton_geo = rv_wharton_geo_t.T
    return rv_wharton_geo
