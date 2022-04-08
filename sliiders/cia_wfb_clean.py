"""
Contains functions to clean CIA World Factbook (WFB). There are different versions
across the years, each with its own format -- this is why there are multiple functions 
to organize different versions, some of which that can be grouped with one another due 
to sharing similar formats.
"""

import os
import re
from codecs import open as copen
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BSoup
from tqdm.auto import tqdm

from .settings import DIR_CIA_RAW, PATH_MPD_RAW, PATH_PWT_RAW

REGIONS_TO_SKIP_CIA_WFB = [
    "Southern Ocean",
    "Indian Ocean",
    "Arctic Ocean",
    "Atlantic Ocean",
    "Pacific Ocean",
    "Baker Island",
]

# manually cleaning the country codes
CCODE_MANUAL = [
    ["American Samoa", "ASM"],
    ["Andorra", "AND"],
    ["Bahamas, The", "BHS"],
    ["Bolivia", "BOL"],
    ["Bouvet Island", "BVT"],
    ["Brunei", "BRN"],
    ["British Indian Ocean Territory", "IOT"],
    ["Burma", "MMR"],
    ["Cape Verde", "CPV"],
    ["Christmas Island", "CXR"],
    ["Cocos (Keeling) Islands", "CCK"],
    ["Congo, Democratic Republic of the", "COD"],
    ["Congo, Republic of the", "COG"],
    ["Cook Islands", "COK"],
    ["Cote d'Ivoire", "CIV"],
    ["Curacao", "CUW"],
    ["Czechia", "CZE"],
    ["East Timor", "TLS"],
    ["Falkland Islands (Islas Malvinas)", "FLK"],
    ["Eritrea", "ERI"],
    ["Faroe Islands", "FRO"],
    ["Francetotal: ", "FRA"],
    ["French Guiana", "GUF"],
    ["French Polynesia", "PYF"],
    ["French Southern and Antarctic Lands", "ATF"],
    ["Gambia, The", "GMB"],
    ["Gaza Strip", "PSE"],  # Gaza Strip and West Bank will together constitute PSE
    ["West Bank", "PSE"],
    ["Jersey", "JEY"],
    ["Guernsey", "GGY"],
    ["Gibraltar", "GIB"],
    ["Greenland", "GRL"],
    ["Guadeloupe", "GLP"],
    ["Guam", "GUM"],
    ["Heard Island and McDonald Islands", "HMD"],
    ["Holy See (Vatican City)", "VAT"],
    ["Hong Kong", "HKG"],
    ["Jan Mayen", "SJM"],  # Svalbard and Jan Mayen are grouped together
    ["Svalbard", "SJM"],
    ["Kiribati", "KIR"],
    ["Korea, North", "PRK"],
    ["Korea, South", "KOR"],
    ["Kosovo", "KO-"],  # Kosovo's original code is XKX, but we use KO-
    ["Laos", "LAO"],
    ["Liechtenstein", "LIE"],
    ["Macau", "MAC"],
    ["Macedonia", "MKD"],
    ["Man, Isle of", "IMN"],
    ["Macedonia, The Former Yugoslav Republic of", "MKD"],
    ["Martinique", "MTQ"],
    ["Marshall Islands", "MHL"],
    ["Mayotte", "MYT"],
    ["Micronesia, Federated States of", "FSM"],
    ["Moldova", "MDA"],
    ["Monaco", "MCO"],
    ["Nauru", "NRU"],
    ["Netherlands Antilles", "BES+CUW+SXM"],  # CIA WFB includes SXM, BES, CUW (not ABW)
    ["Niue", "NIU"],
    ["New Caledonia", "NCL"],
    ["Norfolk Island", "NFK"],
    ["Northern Mariana Islands", "MNP"],
    ["Palau", "PLW"],
    ["Papua New Guinea", "PNG"],
    ["Pitcairn Islands", "PCN"],
    ["Reunion", "REU"],
    ["Russia", "RUS"],
    ["Saint Barthelemy", "BLM"],
    ["Saint Helena", "SHN"],  # will use the below version whenever possible
    ["Saint Helena, Ascension, and Tristan da Cunha", "SHN"],
    ["Saint Martin", "MAF"],
    ["Saint Pierre and Miquelon", "SPM"],
    ["Saint Vincent and the Grenadines", "VCT"],
    ["Samoa", "WSM"],
    ["San Marino", "SMR"],
    ["Serbia and Montenegro", "SRB+MNE"],
    ["Sint Maarten", "SXM"],
    ["Solomon Islands", "SLB"],
    ["Somalia", "SOM"],
    ["South Georgia and South Sandwich Islands", "SGS"],
    ["South Sudan", "SSD"],
    ["Syria", "SYR"],
    ["Tanzania", "TZA"],
    ["Timor-Leste", "TLS"],
    ["Tokelau", "TKL"],
    ["Tonga", "TON"],
    ["Tuvalu", "TUV"],
    ["Vanuatu", "VUT"],
    ["Venezuela", "VEN"],
    ["Virgin Islands", "VIR"],
    ["Wallis and Futuna", "WLF"],
    ["Western Sahara", "ESH"],
]
CCODE_MANUAL = pd.DataFrame(CCODE_MANUAL, columns=["country", "countrycode"])


def helper_wfb_million_cleaner(string):
    """Helper function for cleaning CIA WFB GDP values in millions of USD.

    Parameters
    ----------
    string : str
        containing information about the GDP value (e.g., '$42 million')

    Returns
    -------
    numeric : float
        containing GDP information in millions of USD

    """
    numeric = float(re.sub(r"[a-zA-Z]|\$| |\,|-", "", string))
    if "trillion" in string:
        numeric = numeric * 1000000
    elif "billion" in string:
        numeric = numeric * 1000
    elif "million" not in string:
        numeric = numeric / 1000000

    return numeric


def helper_wfb_gather_soups(
    directory, subdirectory="geos", print_ver=False, encoding=None
):
    """Helper function to go over each geographic location files (in `subdirectory`)
    and gather `bs4.BeautifulSoup` for each file.

    Parameters
    ----------
    directory : str or Path-like
        containing the overall directory containing CIA WFB information for a specific
        version
    subdirectory : str
        subdirectory (under) `directory` that contains all the geographic location files
    print_ver : bool
        if `True`, will gather `bs4.BeautifulSoup` for files with the header 'print_'
        (e.g., `print_us.html`); if `False`, will gather those for files without such
        headers
    encoding : None or str
        how the `codecs.open` function will process the .html file; default is `None`,
        and this will process it as utf-8

    Returns
    -------
    soups : list of `bs4.BeautifulSoup`
        for each of the geographic locations in the `subdirectory` under `directory`

    """

    direc = Path(directory) / subdirectory
    soups = []
    length = 7
    if print_ver:
        length = 13
    for g in os.listdir(direc):
        if not ((".html" in g) and (len(g) == length)):
            continue
        file = copen(str(direc / g), "r", encoding).read()
        soup = BSoup(file, "html.parser")
        soups.append(soup)

    return soups


def helper_fy_cleaner(list_of_years):
    """Helper function for cleaning a list of years (in string format) that may have
    financial year designations instead of YYYY format.

    Parameters
    ----------
    list_of_years : array-like of str or str
        containing years in string format

    Returns
    -------
    list of int or int
        of the year(s) cleaned in YYYY format

    """

    single = False
    if type(list_of_years) is str:
        list_of_years = [list_of_years]
        single = True

    if np.any(["FY" in x for x in list_of_years]):
        fix = []
        for yr in list_of_years:
            if "FY" in yr:
                yr = int(yr.split("/")[-1]) + 1900
                if yr < 1950:
                    yr += 100
            fix.append(str(yr))
        if single:
            return int(fix[0])
        return [int(x) for x in fix]

    if single:
        return int(list_of_years[0])
    return [int(x) for x in list_of_years]


def organize_cia_wfb_2000_2001(
    directory=(DIR_CIA_RAW / "factbook-2001"),
    no_info_names=REGIONS_TO_SKIP_CIA_WFB,
    wfb_year=2001,
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2000 and 2001 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2001 version, but can be specified to take
    care of 2000 version as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    no_info_names : array-like of str
        containing country/region names to be excluded when cleaning the information,
        largely due to their pages containing no usable population and GDP information
        (e.g., Arctic Ocean)
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_collect : pandas.DataFrame
        containing country/region-level population (in ones of people)
    gdp_collect : pandas.DataFrame
        containing country/region-level PPP GDP (in millions of USD) and PPP GDP per
        capita (in ones of USD)

    """

    msg = "Cleans only 2000 to 2001 versions of CIA WFB."
    assert wfb_year in range(2000, 2002), msg
    soups = helper_wfb_gather_soups(directory)

    # population
    pop_collect = []
    for soup in soups:
        name = soup.find("title").text.split(" -- ")[-1].strip()
        if name in no_info_names:
            continue

        popstr = soup.text[
            soup.text.find("Population:") : soup.text.find("Age structure:")
        ].replace("\n", "")
        if ("no indigenous" in popstr) or ("uninhabited" in popstr):
            continue

        popstr = [
            x
            for x in re.split(r"\(|\)", re.sub(r"(Population:)|\,|(est.)", "", popstr))
            if len(x.replace(" ", "")) > 0
        ]
        pop_val, pop_year = popstr[0], popstr[1]

        if name in ["South Africa", "Syria"]:
            pop_year = popstr[-1]

        if "note:" in pop_val:
            pop_val = pop_val.split("note:")[0]

        pop_collect.append([name, float(pop_val.strip()), int(pop_year.strip()[-4:])])
    pop_collect = pd.DataFrame(pop_collect, columns=["country", "pop", "year"])
    pop_collect["wfb_year"] = wfb_year

    # GDP and GDPpc
    gdp_collect = []
    for soup in soups:
        name = soup.find("title").text.split(" -- ")[-1].strip()
        if name in no_info_names:
            continue

        # GDP
        gdp_txt = soup.text.replace("\n", " ")
        front_txt = "GDP: purchasing power parity"
        if wfb_year == 2001:
            front_txt = "GDP:  purchasing power parity"
        if front_txt not in gdp_txt:
            continue

        gdp_txt, gdppc_txt = gdp_txt[
            gdp_txt.find(front_txt) : gdp_txt.find("GDP - composition by sector")
        ].split("GDP - real growth rate:")
        gdp_txt = [
            x.strip()
            for x in re.split(
                r"\(|\)", re.sub(r"({} - \$)|( est.)".format(front_txt), "", gdp_txt)
            )
            if len(x.strip()) > 0
        ]
        if gdp_txt[0] == "NA":
            continue
        gdp_val = helper_wfb_million_cleaner(gdp_txt[0])
        gdp_year = helper_fy_cleaner([gdp_txt[1]])[0]

        # GDPpc
        front_txt = "GDP - per capita: purchasing power parity - "
        if wfb_year == 2001:
            front_txt = "GDP - per capita:  purchasing power parity - "
        gdppc_txt = re.sub(r"\$|( est.)", "", gdppc_txt.split(front_txt)[-1]).strip()
        additional_condition = (name in ["Svalbard", "Norway"]) and (wfb_year == 2001)
        if (gdppc_txt == "NA") or additional_condition:
            continue
        gdppc_val, _ = gdppc_txt.split("(")
        gdppc_val = float(gdppc_val.replace(",", ""))
        # _ = helper_fy_cleaner([gdppc_year.replace(")", "")])[0]

        gdp_collect.append([name, gdp_val, gdppc_val, gdp_year])

    gdp_collect = pd.DataFrame(gdp_collect, columns=["country", "gdp", "gdppc", "year"])
    gdp_collect["wfb_year"] = wfb_year

    return pop_collect, gdp_collect


def organize_cia_wfb_2002_2004(
    directory=(DIR_CIA_RAW / "factbook-2002"), wfb_year=2002
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2002-2004 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2002 version, but can be specified to take
    care of 2003-2004 versions as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    s_yr, e_yr = 2002, 2004
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    # GDP and GDP per capita
    gdp_case = True
    for soup in soups[0:2]:
        gdp_lst = [
            re.sub(r"\n|\t", "", x.text)
            for x in soup.find_all("tr")
            if "power parity" in x.text
        ][1:]
        gdp_lst = [x.split("purchasing power parity - $") for x in gdp_lst]
        if not gdp_case:
            gdp_lst = [[x[0]] + [f.replace(",", "") for f in x[1:]] for x in gdp_lst]
        gdp_collect = []
        for i in gdp_lst:
            # let us manually take care of Cyprus; skip if only containing NA
            if ("Cyp" in i[0]) or (" - NA " in i[0]):
                continue

            gdp_val = i[1].strip().split(" (")
            gdp_val, gdp_year = gdp_val[0], gdp_val[1:]
            if "note" in gdp_val:
                gdp_val = gdp_val.split("note")[0].strip()
            elif "NA" in gdp_val:
                continue

            if gdp_case:
                gdp_val = helper_wfb_million_cleaner(gdp_val)
            else:
                gdp_val = float(gdp_val.strip())

            if not gdp_year:
                gdp_year = wfb_year
            else:
                gdp_year = gdp_year[0].split("est.")[0].replace(")", "").strip()
                gdp_year = helper_fy_cleaner([gdp_year])[0]

            if "World" in i[0]:
                gdp_collect.append(["World", gdp_val, gdp_year])
            else:
                gdp_collect.append([i[0], gdp_val, gdp_year])

        if gdp_case:
            gdp_df = gdp_collect.copy()
            gdp_case = False
        else:
            gdppc_df = gdp_collect.copy()

    gdppc_df = pd.DataFrame(gdppc_df, columns=["country", "gdppc", "year"])
    gdp_df = pd.DataFrame(gdp_df, columns=["country", "gdp", "year"])
    gdp_df["wfb_year"], gdppc_df["wfb_year"] = wfb_year, wfb_year

    # Population
    pop_df = []
    pop_lst = [
        re.sub(r"\n|\t", "", x.text)
        for x in soups[-1].find_all("tr")
        if "est.)" in x.text
    ][1:]
    for i in pop_lst:
        if ("no indigenous" in i) or ("uninhabited" in i):
            continue

        pop_idx = re.search(r"[0-9]", i).span()[0]
        name, pop_info = i[0:pop_idx], i[pop_idx:]
        pop_val = pop_info.split("(")
        pop_val, pop_year = pop_val[0], pop_val[-1]
        if "note" in pop_val:
            pop_val = pop_val.split("note")[0]
        if "million" in pop_val:
            pop_val = float(pop_val.strip().replace(" million", "")) * 1000000
        else:
            pop_val = float(pop_val.strip().replace(",", ""))
        pop_year = int(re.sub(r"[a-zA-Z]|\.|\)", "", pop_year))
        pop_df.append([name, pop_val, pop_year])

    pop_df = pd.DataFrame(pop_df, columns=["country", "pop", "year"])
    pop_df["wfb_year"] = wfb_year

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2005_2008(
    directory=(DIR_CIA_RAW / "factbook-2005"), wfb_year=2005
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2005 and 2007 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2005 version, but can be specified to take
    care of 2006-2007 versions as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    s_yr, e_yr = 2005, 2008
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    # GDP and GDP per capita
    for case, soup in enumerate(soups):
        collect = []
        lst = [
            re.sub(r"\n|\t", "", x.text)
            for x in soup.find_all("tr")
            if "est.)" in x.text
        ][1:]

        for i in lst:
            cnd_check = ("no indigenous" in i) or ("uninhabited" in i) or ("NA" in i)
            if ("Cyprus" in i) or cnd_check:
                continue

            searchby = r"\$"
            if case == 2:
                searchby = r"[0-9]"

            idx = re.search(searchby, i).span()[0]
            name, value = i[0:idx].replace("purchasing power parity - ", ""), i[idx:]

            if "World" in name:
                name = "World"

            value = value.split(" (")
            value, year = value[0], value[-1]
            if ("- supplemented" in value) or ("note" in value):
                value = re.split(r"note|- supplemented", value)[0]
                value = re.sub(r"\;", "", value)
            if case == 0:
                value = helper_wfb_million_cleaner(value.strip())
            else:
                value = re.sub(r"\$|\,| for Serbia", "", value).strip()
                if "million" in value:
                    value = float(value.replace("million", "").strip()) * 1000000
                value = int(value)
            year = int(
                re.sub(
                    r"[a-zA-Z]", "", year.replace(" est.", "").replace(")", "").strip()
                )
            )

            collect.append([name, value, year])
        if case == 0:
            gdp_df = collect.copy()
        elif case == 1:
            gdppc_df = collect.copy()

    # GDP and GDPpc
    gdp_df = pd.DataFrame(gdp_df, columns=["country", "gdp", "year"])
    gdppc_df = pd.DataFrame(gdppc_df, columns=["country", "gdppc", "year"])
    gdp_df["wfb_year"], gdppc_df["wfb_year"] = wfb_year, wfb_year

    # population
    pop_df = pd.DataFrame(collect, columns=["country", "pop", "year"])
    pop_df["wfb_year"] = wfb_year

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2009_2012(
    directory=(DIR_CIA_RAW / "factbook-2009"), wfb_year=2009
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2009-2012 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2009 version, but can be specified to take
    care of 2010-2012 versions as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    s_yr, e_yr = 2009, 2012
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    cat_dict = {"class": "category_data"}
    for i, soup in enumerate(soups):
        # every displayed "row" is organized as a "table"
        souptable = [
            x
            for x in soup.find_all("table")
            if x.find("td", attrs=cat_dict) is not None
        ]
        souptable = [t.find("td", attrs={"class": "fl_region"}) for t in souptable]
        names = [t.find("a").text for t in souptable]
        values = [t.find("td", attrs=cat_dict) for t in souptable]

        # for reducing redundancies, as the tables are in a nested structure
        # same country information can be searched multiple times
        already_names = ["Akrotiri", "Dhekelia"]
        cases = []
        collect_df = []
        for j, value in enumerate(values):
            name, v = names[j], value.text
            if name in already_names:
                continue
            if i != 2:
                org = [
                    x.strip()
                    for x in v.split("\n")
                    if (len(x.strip()) > 0) and ("NA" not in x)
                ]
                numbers = [
                    x.replace("note:", "").strip()
                    for x in org
                    if (("(" in x) and (")" in x) and ("MADDISON" not in x))
                    or ("est." in x)
                ]
                note = [x for x in org if (x not in numbers) and ("data are in" in x)]
                num_orgs, years = [], []
                for num in numbers:
                    n, year = num.split("(")[0], num.split("(")[-1]
                    if i == 0:
                        n = helper_wfb_million_cleaner(n)
                    else:
                        n = int(re.sub("\$|\,", "", n).strip())
                    num_orgs.append(n)
                    years.append(
                        int(re.sub(r"[a-zA-Z]|\)|\.|\;|\$", "", year).strip()[0:4])
                    )

                usd_years = years.copy()
                if note:
                    nn = note[0].split(";")[0]
                    usd_years = [int(re.sub(r"[a-zA-Z]|\:|\.", "", nn).strip())] * len(
                        years
                    )
                df = pd.DataFrame(
                    data=dict(
                        zip(
                            ["country", "year", "gdp", "usd_year"],
                            [[name] * len(years), years, num_orgs, usd_years],
                        )
                    )
                )
            else:
                if ("no indigenous" in v) or ("uninhabited" in v):
                    continue
                org = v.strip().replace("\n", "").replace(",", "").split("(")
                num = org[0].split("note")[0].strip()
                if "million" in num:
                    num = int(float(num.replace("million", "").strip()) * 1000000)
                else:
                    num = int(num.replace("total:", "").strip())

                if (name == "Curacao") and (wfb_year in [2010, 2011]):
                    year = re.sub(r"\)|\.", "", [x for x in org if "est." in x][0])
                elif (name == "South Sudan") and (wfb_year == 2011):
                    year = re.sub(r"\)", "", org[-1])
                else:
                    year = [x.split("est.)")[0] for x in org if "est.)" in x][0]
                year = int(re.sub(r"[a-zA-Z]| ", "", year))
                df = [name, num, year]

            already_names.append(name)
            collect_df.append(df)
        if i == 0:
            gdp_df = pd.concat(collect_df, axis=0).reset_index(drop=True)
        elif i == 1:
            gdppc_df = pd.concat(collect_df, axis=0).reset_index(drop=True)
            gdppc_df.rename(columns={"gdp": "gdppc"}, inplace=True)
    pop_df = pd.DataFrame(collect_df, columns=["country", "pop", "year"])
    gdp_df["wfb_year"] = wfb_year
    gdppc_df["wfb_year"], pop_df["wfb_year"] = wfb_year, wfb_year

    # drop duplicates, due to multiple entries for North Korea in 2011
    gdppc_df.drop_duplicates(inplace=True)
    gdp_df.drop_duplicates(inplace=True)

    # manual cleaning to fix or drop unreliable data
    if wfb_year in [2011, 2012]:
        gdppc_df.loc[
            (gdppc_df.country == "Gibraltar") & (gdppc_df.gdppc == 43000),
            ["year", "usd_year"],
        ] = 2008
        if wfb_year == 2012:
            gdppc_df.loc[
                (gdppc_df.country == "Kosovo") & (gdppc_df.gdppc == 7400),
                ["year", "usd_year"],
            ] = 2012

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2013_2014(
    directory=(DIR_CIA_RAW / "factbook-2013"), wfb_year=2013
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2013-2014 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2013 version, but can be specified to take
    care of 2014 version as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    s_yr, e_yr = 2013, 2014
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    for i, soup in enumerate(soups):
        if (i == 2) and (wfb_year == 2015):
            continue
        soupfind = soup.find("div", attrs={"class": "text-holder-full"}).find_all(
            "td", attrs={"class": "fl_region"}
        )
        df_agg = []
        for j, case in enumerate(soupfind):
            name = case.text.split("\n\n")[0]
            values = case.find("td").text

            cnd_skip1 = name in ["Akrotiri", "Dhekelia"]
            cnd_skip2 = (i != 2) and (("NA" in values) or (name == "Gaza Strip"))
            cnd_skip3 = ("no indigenous" in values) or ("uninhabited" in values)
            if cnd_skip1 or cnd_skip2 or cnd_skip3:
                continue
            values = [x for x in values.split("\n") if len(x.strip()) > 0]
            if (i == 2) and (wfb_year == 2014):
                note = [x for x in values if ("note" in x)]
                values = [
                    x for x in values if ("note" not in x) and ("top ten" not in x)
                ]
            else:
                note = [x for x in values if ("note: data are in" in x)]
                if np.any(["est." in x for x in values]) and (i != 2):
                    values = [
                        x
                        for x in values
                        if ("est." in x) and ("note" not in x) and ("top ten" not in x)
                    ]
                else:
                    values = [
                        x for x in values if ("note" not in x) and ("top ten" not in x)
                    ]

            nums, years = [], []
            for val in values:
                if (name == "Bahrain") and (i == 2) and (wfb_year == 2013):
                    num, year = val.split("July")
                elif (i == 2) and (wfb_year == 2014) and (len(note) > 0):
                    num = val.strip()
                    if "(" in num:
                        num, year = num.split("(")
                    if "est." in note[0]:
                        year = note[0].split("(")[-1]
                elif "(" in val:
                    num, year = val.split("(")
                else:
                    num, year = val.strip(), str(wfb_year)
                year = re.sub(r"\(|\)|est.", "", year).strip()
                if "FY" in year:
                    year = helper_fy_cleaner(year)
                else:
                    year = int(re.sub(r"[a-zA-Z]", "", year).strip())
                if i == 0:
                    num = helper_wfb_million_cleaner(num.strip())
                else:
                    num = int(re.sub(r"\$|\,", "", num))
                years.append(year)
                nums.append(num)

            if len(nums) == 0:
                continue

            if i != 2:
                usd_years = years.copy()
                if len(note) > 0:
                    note = note[0].split("note: data are in")[-1].split("US dollars")[0]
                    usd_years = [int(note.strip())] * len(years)

                columns = ["country", "year", "gdp", "usd_year"]
                datavals = [[name] * len(years), years, nums, usd_years]

            else:
                columns = ["country", "year", "pop"]
                datavals = [[name] * len(years), years, nums]

            df_agg.append(pd.DataFrame(data=dict(zip(columns, datavals))))

        if i == 0:
            gdp_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdp_df["wfb_year"] = wfb_year
        elif i == 1:
            gdppc_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdppc_df.rename(columns={"gdp": "gdppc"}, inplace=True)

    pop_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
    gdppc_df["wfb_year"], pop_df["wfb_year"] = wfb_year, wfb_year

    # manual cleaning to fix or drop unreliable data
    if wfb_year == 2013:
        gdp_df.loc[(gdp_df.country == "Macau") & (gdp_df.gdp > 47000), "year"] = 2012
        gdp_df = gdp_df.loc[
            ~((gdp_df.country == "Syria") & (gdp_df.year == 2010)), :
        ].copy()
        gdppc_df.loc[
            (gdppc_df.country == "Gibraltar") & (gdppc_df.gdppc == 43000), "year"
        ] = 2008
    else:
        gdp_df = gdp_df.loc[
            ~((gdp_df.country == "Croatia") & (gdp_df.year == 2012)), :
        ].copy()
        gdppc_df = gdppc_df.loc[
            ~((gdppc_df.country == "Kenya") & (gdppc_df.year == 2013)), :
        ].copy()
    gdppc_df = gdppc_df.loc[
        ~((gdppc_df.country == "Syria") & (gdppc_df.year == 2010)), :
    ].copy()

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2015(directory=(DIR_CIA_RAW / "factbook-2015")):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) version 2015 into `pandas.DataFrame` formats.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)
    """

    lst_directory = Path(directory) / "rankorder"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}rank.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    for i, soup in enumerate(soups):
        ranks = soup.find("table", attrs={"id": "rankOrder"})
        rows = ranks.find_all("tr")
        df = []
        for tr in rows:
            if "Date of Information" in tr.text:
                continue

            ranking, name, value, year = tr.find_all("td")
            if len(value.text.strip()) == 0:
                continue
            if len(year.text.strip()) == 0:
                year = 2014
            elif "FY" in year.text:
                front, back = year.text.split("/")
                back = re.sub(r"[a-zA-Z]|\.", "", back).strip()
                year = int(back) + 2000
                if year > 2050:
                    year -= 100
            else:
                year = int(re.sub(r"[a-zA-Z]|\.", "", year.text).strip())

            value = int(re.sub(r"\$|\,", "", value.text).strip())
            if i == 0:
                value /= 1000000
            df.append([name.text, year, value])

        if i == 0:
            gdp_df = pd.DataFrame(df, columns=["country", "year", "gdp"])
        elif i == 1:
            gdppc_df = pd.DataFrame(df, columns=["country", "year", "gdppc"])

    gdppc_df["usd_year"], gdp_df["usd_year"] = gdppc_df["year"], gdp_df["year"]
    gdppc_df["wfb_year"], gdp_df["wfb_year"] = 2015, 2015
    pop_df = pd.DataFrame(df, columns=["country", "year", "pop"])
    pop_df["wfb_year"] = 2015

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2016_2017(
    directory=(DIR_CIA_RAW / "factbook-2016"), wfb_year=2016
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2016-2017 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2016 version, but can be specified to take
    care of 2017 version as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    s_yr, e_yr = 2016, 2017
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []
    for i in [2001, 2004, 2119]:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    soup_case = []
    for i, soup in enumerate(soups):
        names = [x.text for x in soup.find_all("td", attrs={"class": "country"})]
        cases = soup.find_all("td", attrs={"class": "fieldData"})
        if i == 2:
            cases = [x.text for x in cases]
        else:
            cases = [x.text.split("\n") for x in cases]
        df_agg = []
        for j, name in enumerate(names):
            cnd_skip1 = name in ["Akrotiri", "Dhekelia"]
            cnd_skip2 = (i != 2) and np.any(["NA" in x for x in cases[j]])
            cnd_skip3 = (i == 2) and (
                ("no indigenous" in cases[j]) or ("uninhabited" in cases[j])
            )
            if cnd_skip1 or cnd_skip2 or cnd_skip3:
                continue
            if i != 2:
                values = [
                    x.strip()
                    for x in cases[j]
                    if ("est." in x)
                    and ("top ten" not in x)
                    and ("note" not in x)
                    and (len(x.strip()) > 0)
                ]
                note = [
                    x.strip()
                    for x in cases[j]
                    if (len(x.strip()) > 0) and ("note" in x)
                ]

                nums, years = [], []
                for val in values:
                    num, year = val.split("(")
                    if i == 0:
                        num = helper_wfb_million_cleaner(num)
                    else:
                        num = int(re.sub(r"\,|\$", "", num).strip())
                    year = re.sub(r"\)|est.", "", year).strip()
                    if "FY" in year:
                        year = helper_fy_cleaner(year)
                    else:
                        year = int(year.strip())
                    nums.append(num), years.append(year)
                usd_years = years.copy()
                if len(note) > 0:
                    note_fix = note[0].replace(" US", "")
                    idx = note_fix.find(" dollars")
                    if idx != -1:
                        usd_years = [int(note_fix[(idx - 4) : idx])] * len(years)

            else:
                case = cases[j].replace("\n", "").split("(")
                if np.any(["top ten" in x for x in case]):
                    num, year = case[0], case[1].split("top ten")[0]
                else:
                    num = re.split(r"note|rank by population", case[0])[0]
                    year = case[-1]
                year = year.replace(")", "").split("note")[0]
                if "FY" in year:
                    year = helper_fy_cleaner(
                        "FY" + re.sub(r"[a-zA-Z]|\.", "", year).strip()
                    )
                else:
                    year = int(re.sub(r"[a-zA-Z]|\.", "", year).strip())
                if "million" in num:
                    num = int(float(num.replace("million", "").strip()) * 1000000)
                else:
                    num = int(re.sub(r"\,|[a-zA-Z]|\:", "", num).strip())

            if i != 2:
                columns = ["country", "year", "gdp", "usd_year"]
                datavals = [[name] * len(nums), years, nums, usd_years]
                df_agg.append(pd.DataFrame(data=dict(zip(columns, datavals))))
            else:
                df_agg.append([name, year, num])

        if i == 0:
            gdp_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdp_df["wfb_year"] = wfb_year
        elif i == 1:
            gdppc_df = pd.concat(df_agg, axis=0).reset_index(drop=True)
            gdppc_df.rename(columns={"gdp": "gdppc"}, inplace=True)

    pop_df = pd.DataFrame(df_agg, columns=["country", "year", "pop"])
    gdppc_df["wfb_year"], pop_df["wfb_year"] = wfb_year, wfb_year
    gdppc_df.drop_duplicates(inplace=True)

    # manual cleaning to fix or drop unreliable data
    gdp_df = gdp_df.loc[
        ~((gdp_df.country == "American Samoa") & (gdp_df.year == 2012)), :
    ].copy()
    gdp_df = gdp_df.loc[
        ~((gdp_df.country == "Faroe Islands") & (gdp_df.year == 2013)), :
    ].copy()
    if wfb_year == 2016:
        gdppc_df = gdppc_df.loc[
            ~((gdppc_df.country == "Syria") & (gdppc_df.year == 2010)), :
        ].copy()

    return pop_df, gdp_df, gdppc_df


def organize_cia_wfb_2018_2019(
    directory=(DIR_CIA_RAW / "factbook-2018"), wfb_year=2018
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) versions 2018-2019 into `pandas.DataFrame` formats. Baseline
    function is based on organizing the 2018 version, but can be specified to take
    care of 2019 version as well.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in

    Returns
    -------
    pop_df : pandas.DataFrame
        containing country/region-level population information (in ones of people)
    gdp_df : pandas.DataFrame
        containing country/region-level PPP GDP information (in millions of USD)
    gdppc_df : pandas.DataFrame
        containing country/region-level PPP GDP per capita information (in ones of USD)

    """

    s_yr, e_yr = 2018, 2019
    msg = "Cleans only {} to {} versions of CIA WFB.".format(s_yr, e_yr)
    assert wfb_year in range(s_yr, e_yr + 1), msg

    lst_directory = Path(directory) / "fields"
    soups = []

    # html names for storing GDP, GDP per capita (both in PPP), and population
    html_lst = [208, 211, 335]
    for i in html_lst:
        file = copen(str(lst_directory / "{}.html".format(i)), "r").read()
        soups.append(BSoup(file, "html.parser"))

    find_val_fields = [
        "field-gdp-purchasing-power-parity",
        "field-gdp-per-capita-ppp",
        "field-population",
    ]
    find_category = "category_data subfield historic"
    df_cols_list = [
        ["country", "year", "gdp", "usd_year"],
        ["country", "year", "gdppc", "usd_year"],
        ["country", "year", "pop"],
    ]
    for i, soup in enumerate(soups):
        find_val_field, df_cols = find_val_fields[i], df_cols_list[i]
        souptable = soup.find("table", attrs={"id": "fieldListing"})
        countries = [
            x.text.replace("\n", "")
            for x in souptable.find_all("td", attrs={"class": "country"})
        ]
        if i == 2:
            find_category = "category_data subfield numeric"
        values = souptable.find_all("div", attrs={"id": find_val_field})
        notes = [v.find("div", attrs={"class": "category_data note"}) for v in values]
        values = [v.find_all("div", attrs={"class": find_category}) for v in values]

        df_collect = []
        for j, val in enumerate(values):
            # case when there are no information available
            if len(val) == 0:
                continue

            # getting the country name and note (note could be None)
            name, note = countries[j], notes[j]

            # multiple years and values available in versions 2017 and onwards
            numbers, years = [], []
            no_known = False
            for v in val:
                year = None
                num = v.text.replace("\n", "").split("(")
                if len(num) > 1:
                    num, year = num[0], num[-1]
                    year = re.sub(r"[a-zA-Z]|\)| |\.", "", year)
                    if ("FY" in v.text) and ("/" in year):
                        year = "FY" + year
                    year = helper_fy_cleaner(year)

                if i == 0:
                    numbers.append(helper_wfb_million_cleaner(num))
                    years.append(year)
                else:
                    cnd_check = (
                        ("no indigenous" in num)
                        or ("uninhabited" in num)
                        or ("Akrotiri" in num)
                        or ("NA" in num)
                        or (year is None)
                    )
                    if cnd_check:
                        continue
                    if ("million" in num) and (i == 2):
                        num = int(float(num.replace("million", "").strip()) * 1000000)
                        numbers.append(num)
                    else:
                        numbers.append(int(re.sub(r"\$|\,|[a-zA-Z]", "", num.strip())))
                    years.append(year)
            if len(numbers) == 0:
                continue

            name = [name] * len(years)

            # what year the GDP values are in
            if i != 2:
                usd_years = years.copy()
                if note is not None:
                    note = note.text
                    if not (("data are in" in note) and ("dollars" in note)):
                        continue
                    if (";" in note) or ("the war-driven" in note):
                        note = [
                            x
                            for x in re.split(r"\;|the war-driven", note)
                            if ("data are in" in x) and ("dollars" in x)
                        ][0]
                    noteyear = re.sub(r"[a-zA-Z]| |\n|\:", "", note)
                    usd_years = [int(noteyear)] * len(years)

                df_vals = [name, years, numbers, usd_years]
            else:
                df_vals = [name, years, numbers]
            df_collect.append(pd.DataFrame(data=dict(zip(df_cols, df_vals))))

        if i == 0:
            gdp_df = pd.concat(df_collect, axis=0).reset_index(drop=True)
        elif i == 1:
            gdppc_df = pd.concat(df_collect, axis=0).reset_index(drop=True)

    pop_df = pd.concat(df_collect, axis=0).reset_index(drop=True)
    gdp_df["wfb_year"], gdppc_df["wfb_year"] = wfb_year, wfb_year
    pop_df["wfb_year"] = wfb_year

    return pop_df, gdp_df, gdppc_df


def helper_wfb_2020(soup):
    """Simple helper function for finding and cleaning the name of a country/region,
    used for organizing CIA World Factbook versions 2018 to 2020 (in conjunction with
    the function `organize_cia_wfb_2018_2020`).

    Parameters
    ----------
    soup : bs4.BeautifulSoup
        containing country/region information

    Returns
    -------
    name : str
        of the country/region being represented in `soup`

    """
    name = soup.find("title").text
    if " :: " in name:
        name = name.split(" :: ")[1].split(" — ")[0]
    else:
        name = name.split(" - ")[0]

    return name


def organize_cia_wfb_2020(
    directory=(DIR_CIA_RAW / "factbook-2020"),
    wfb_year=2020,
    no_info_names=REGIONS_TO_SKIP_CIA_WFB,
):
    """Organizes the population, GDP, and GDP per capita information from the CIA World
    Factbook (WFB) version 2020 into `pandas.DataFrame` format.

    Parameters
    ----------
    directory : pathlib.Path or str
        containing the relevant WFB information
    wfb_year : int
        year that the WFB version was released in
    no_info_names : array-like of str
        containing country/region names to be excluded when cleaning the information,
        largely due to their pages containing no usable population and GDP information
        (e.g., Arctic Ocean)

    Returns
    -------
    pop_collect : pandas.DataFrame
        containing population information (units in ones of people)
    gdp_collect : pandas.DataFrame
        containing PPP GDP information (units in millions of USD, USD year designated
        by the column `usd_year`)
    gdppc_collect : pandas.DataFrame
        containing PPP GDP per capita information (units in ones of USD, USD year
        designated by the column `usd_year`)
    """

    msg = "Cleans only 2020 version of CIA WFB."
    assert wfb_year == 2020, msg

    # gathering soups
    soups = helper_wfb_gather_soups(directory, print_ver=True)

    # population
    pop_collect = []
    for soup in soups:
        name = helper_wfb_2020(soup)
        if name in no_info_names:
            continue

        pop_text = (
            soup.text[
                soup.text.find("People and Society ::") : soup.text.find("Nationality:")
            ]
            .split("Population:\n")[1]
            .replace("\n", " ")
        )
        if ("no indigenous" in pop_text) or ("uninhabited" in pop_text):
            pop_val, pop_year = 0, 2020
        else:
            if "note" in pop_text:
                pop_text = pop_text.split("note")[0]

            if name in ["Akrotiri", "Dhekelia"]:
                continue
            elif name == "European Union":
                pop_val = float(
                    re.sub(r" |\,", "", pop_text.split("rank by population:")[0])
                )
                pop_year = 2020
            else:
                pop_val = pop_text.split(" (")[0].replace(" ", "")
                pop_year = pop_text.split(" (")[-1]
                split_by = ")"
                if "est. est.)" in pop_year:
                    split_by = "est. est.)"
                elif "est.)" in pop_year:
                    split_by = "est.)"

                pop_year = [
                    x for x in pop_year.split(split_by)[0].split(" ") if len(x) > 0
                ][-1]
                if "million" in pop_val:
                    pop_val = float(pop_val.replace("million", "")) * 1000000
                else:
                    pop_val = float(re.sub(r"[a-zA-Z\W]", "", pop_val))

        pop_collect += [[name, pop_val, int(pop_year)]]

    pop_collect = pd.DataFrame(pop_collect, columns=["country", "pop", "year"])
    pop_collect["wfb_year"] = wfb_year

    gdp_str_first = "GDP (purchasing power parity) - real:"
    if wfb_year != 2020:
        gdp_str_first = gdp_str_first.split(" - ")[0]

    # GDP and GDP per capita
    gdp_collect, gdppc_collect = [], []
    for soup in soups:
        name = helper_wfb_2020(soup)
        if name in no_info_names + ["Gaza Strip"]:
            continue

        # GDP (not GDPpc) information
        gdp_info_all = (
            soup.text[
                soup.text.find(gdp_str_first) : soup.text.find("Gross national saving:")
            ]
            .replace("\n", " ")
            .split("GDP (official exchange rate):")
        )

        gdp_info = gdp_info_all[0].replace(gdp_str_first, "")
        if "NA" in gdp_info:
            continue

        if len(gdp_info) > 0:
            if (wfb_year != 2020) and (gdp_info[0] in [":", ";"]):
                gdp_info = gdp_info[1:]

            note = None
            if ("note: " in gdp_info) and (name != "Saint Pierre and Miquelon"):
                gdp_info = gdp_info.split("note: ")
                gdp_info, note = gdp_info[0], gdp_info[1:]

            if (wfb_year != 2020) and ("country comparison to" in gdp_info):
                gdp_info = gdp_info.split("country comparison to")[0]
            gdp_info = [
                x.strip() for x in re.split(r"\(|\)", gdp_info) if len(x.strip()) > 0
            ]

            if len(gdp_info) > 0:
                gdp_vals = gdp_info[0::2]
                if name == "Saint Pierre and Miquelon":
                    gdp_vals = gdp_vals[0:-1]
                gdp_vals = [helper_wfb_million_cleaner(x) for x in gdp_vals]
                gdp_yrs = helper_fy_cleaner(
                    [x.replace("est.", "").strip() for x in gdp_info[1::2]]
                )

                usd_year_assumed = "usd_year_assumed"
                if note is not None:
                    note = re.sub(r"[a-zA-Z]| ", "", note[0])
                    if note[0] == ";":
                        note = note[1:]
                    elif (";" in note) or ("-" in note):
                        note = re.split(r";|-", note)[0]

                    if (":" in note) and (wfb_year != 2020):
                        note = note.split(":")[0]

                    gdp_usd_yrs = [int(note.replace(".", ""))] * len(gdp_yrs)
                    usd_year_assumed = "usd_year_original"
                else:
                    gdp_usd_yrs = gdp_yrs
                append_this = []
                for l, yr in enumerate(gdp_yrs):
                    append_this.append(
                        [name, yr, gdp_usd_yrs[l], gdp_vals[l], usd_year_assumed]
                    )
                gdp_collect += append_this

        # GDPpc information
        gdppc_info = gdp_info_all[-1].split("GDP - per capita (PPP):")[-1]
        if len(gdppc_info.strip()) > 0:
            if "country comparison" not in gdppc_info:
                if "GDP - composition, by sector of origin" in gdppc_info:
                    gdppc_info = gdppc_info.split(
                        "GDP - composition, by sector of origin"
                    )[0]
            else:
                gdppc_info = gdppc_info.split("country comparison")[0]

            for string in ["Ease of Doing Business", "GDP - composition, by sector"]:
                if string in gdppc_info:
                    gdppc_info = gdppc_info.split(string)[0]

            if "NA" in gdppc_info:
                continue

            note = None
            if "note:" in gdppc_info:
                gdppc_info, note = gdppc_info.split("note:")

            gdppc_info = [
                x for x in re.split(r"\(|\)", gdppc_info) if len(x.replace(" ", "")) > 0
            ]
            gdppc_vals, gdppc_years = gdppc_info[0::2], gdppc_info[1::2]
            gdppc_vals = [float(re.sub(r"\$|,", "", x.strip())) for x in gdppc_vals]
            gdppc_years = helper_fy_cleaner(
                [x.strip().replace(" est.", "") for x in gdppc_years]
            )

            usd_year_assumed = "usd_year_assumed"
            gdppc_usd_years = gdppc_years
            if (note is not None) and (name != "West Bank"):
                gdppc_usd_years = [int(re.sub(r"[a-zA-Z]|\.", "", note).strip())] * len(
                    gdppc_years
                )
                usd_year_assumed = "usd_year_orig"

            append_this = []
            for l, yr in enumerate(gdppc_usd_years):
                append_this.append(
                    [name, gdppc_years[l], yr, gdppc_vals[l], usd_year_assumed]
                )
            gdppc_collect += append_this

    # organizing in pandas.DataFrame format
    gdp_columns = ["country", "year", "usd_year", "gdp", "usd_year_source"]
    gdp_collect = pd.DataFrame(gdp_collect, columns=gdp_columns)
    gdp_collect["wfb_year"] = wfb_year

    gdp_columns[3] = "gdppc"
    gdppc_collect = pd.DataFrame(gdppc_collect, columns=gdp_columns)
    gdppc_collect["wfb_year"] = wfb_year

    # fixing Cote d'Ivoire name
    gdp_collect.loc[
        gdp_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"
    gdppc_collect.loc[
        gdppc_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"
    pop_collect.loc[
        pop_collect.country == "Cote d&#39;Ivoire", "country"
    ] = "Cote d'Ivoire"

    # manual cleaning to fix or drop unreliable data
    gdppc_error_ctries = [
        "Togo",
        "Zimbabwe",
        "Turkmenistan",
        "Venezuela",
        "Sierra Leone",
        "Kosovo",
        "Guinea-Bissau",
        "Benin",
        "Cote d'Ivoire",
        "Kuwait",
        "Niger",
        "Taiwan",
        "Germany",
    ]
    gdppc_collect = gdppc_collect.loc[
        ~(
            (gdppc_collect.country.isin(gdppc_error_ctries))
            & (gdppc_collect.year == 2017)
        ),
        :,
    ].copy()

    gdp_error_ctries = [
        x for x in gdppc_error_ctries if x not in ["Kosovo", "Sierra Leone", "Taiwan"]
    ]
    gdp_error_ctries += ["Mozambique", "Mauritania", "Pakistan", "Jordan"]
    gdp_collect = gdp_collect.loc[
        ~((gdp_collect.country.isin(gdp_error_ctries)) & (gdp_collect.year == 2017)), :
    ].copy()

    return pop_collect, gdp_collect, gdppc_collect


def organize_gather_cia_wfb_2000_2020(years=list(range(2000, 2021))):
    """Cleaning all CIA WFB versions, from 2000 to 2020, and gathering them in list
    format (one list each for population, GDP, and GDP per capita).

    Parameters
    ----------
    years : array-like of int
        containing the version years to be cleaned; default runs from 2000 to 2020.

    Returns
    -------
    cia_pop_gather : list of pandas.DataFrame
        containing population data from the oldest version to the newest (data is in
        ones of people)
    cia_gdp_gather : list of pandas.DataFrame
        containing GDP data from the oldest version to the newest (data is in millions
        of USD)
    cia_gdppc_gather : list of pandas.DataFrame
        containing GDP per capita data from the oldest version to the newest (data is
        in ones of USD)

    """

    years = np.sort(years)
    msg = "Only cleans versions 2000 to 2020."
    assert (year.max() <= 2020) and (year.min() >= 2000), msg

    # gathering country name to country code conversion
    CCODE_PWT = pd.read_excel(PATH_PWT_RAW)[
        ["countrycode", "country"]
    ].drop_duplicates()
    CCODE_MPD = pd.read_excel(PATH_MPD_RAW)[
        ["countrycode", "country"]
    ].drop_duplicates()
    CCODE_DF = (
        pd.concat([CCODE_MPD, CCODE_PWT, CCODE_MANUAL], axis=0)
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"countrycode": "ccode"})
    )

    cia_gdp_gather = []
    cia_pop_gather = []
    cia_gdppc_gather = []
    for yr in tqdm(years):
        directory = DIR_CIA_RAW / "factbook-{}".format(yr)
        if yr in [2000, 2001]:
            pop_df, gdp_df = organize_cia_wfb_2000_2001(directory, wfb_year=yr)
            gdppc_df = gdp_df.copy()[["country", "year", "gdppc", "wfb_year"]]
            gdp_df = gdp_df[["country", "year", "gdp", "wfb_year"]]
        elif yr in [2002, 2003, 2004]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2002_2004(directory, yr)
        elif yr in range(2005, 2009):
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2005_2008(directory, yr)
        elif yr in range(2009, 2013):
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2009_2012(directory, yr)
        elif yr in [2013, 2014]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2013_2014(directory, yr)
        elif yr == 2015:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2015()
        elif yr in [2016, 2017]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2016_2017(directory, yr)
        elif yr in [2018, 2019]:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2018_2019(directory, yr)
        else:
            pop_df, gdp_df, gdppc_df = organize_cia_wfb_2020()

        if "usd_year" not in gdp_df.columns:
            gdp_df["usd_year"] = gdp_df["year"]
        if "usd_year" not in gdppc_df.columns:
            gdppc_df["usd_year"] = gdppc_df["year"]

        pop_df = pop_df.merge(CCODE_DF, on=["country"], how="left")
        gdp_df = gdp_df.merge(CCODE_DF, on=["country"], how="left")
        gdppc_df = gdppc_df.merge(CCODE_DF, on=["country"], how="left")

        # manual cleaning after Palestine (West Bank + Gaza Strip)
        if "PSE" in pop_df.ccode.values:
            pse_df = pop_df.loc[pop_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df = pse_df.groupby(["ccode", "year"]).sum()[["pop"]].reset_index()
            pse_df["country"] = "Palestine"
            pse_df["wfb_year"] = yr
            pop_df = pd.concat(
                [pse_df, pop_df.loc[pop_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)
        if "PSE" in gdp_df.ccode.values:
            pse_df = gdp_df.loc[gdp_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df = (
                pse_df.groupby(["ccode", "year", "usd_year"])
                .sum()[["gdp"]]
                .reset_index()
            )
            pse_df["country"] = "Palestine"
            pse_df["wfb_year"] = yr
            gdp_df = pd.concat(
                [pse_df, gdp_df.loc[gdp_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)
        if "PSE" in gdppc_df.ccode.values:
            # getting those that do not have more than 1 ccode-year observations
            pse_df = gdppc_df.loc[gdppc_df.ccode == "PSE", :].reset_index(drop=True)
            pse_df["counter"] = 1
            pse_counter = (
                pse_df.groupby(["ccode", "year", "usd_year"])
                .sum()[["counter"]]
                .reset_index()
            )
            pse_df.drop(["counter"], axis=1, inplace=True)
            pse_df = pse_df.merge(
                pse_counter, on=["ccode", "year", "usd_year"], how="left"
            )
            pse_df = pse_df.loc[
                pse_df.counter == 1, ["ccode", "year", "gdppc", "usd_year", "wfb_year"]
            ]
            pse_df["country"] = "Palestine"
            gdppc_df = pd.concat(
                [pse_df, gdppc_df.loc[gdppc_df.ccode != "PSE", :].copy()], axis=0
            ).reset_index(drop=True)

        cia_pop_gather.append(pop_df)
        cia_gdp_gather.append(gdp_df)
        cia_gdppc_gather.append(gdppc_df)

    return cia_pop_gather, cia_gdp_gather, cia_gdppc_gather


def wfb_merge_year_by_year(df_old, df_new, varname="gdp"):
    """Based on the version year (column `wfb_year`), updates the information of
    dataset `df_old` with that of dataset `df_new`, and also merges any information that
    is newly introduced in `df_new`.

    Parameters
    ----------
    df_old : pandas.DataFrame
        contains older data, whose information is from CIA WFB versions that are older
        than of `df_new`
    df_new : pandas.DataFrame
        contains newer data, whose information is from CIA WFB version that is newer
        than any version present in `df_new`
    varname : str
        variable name to aggregate for, can be either `gdp`, `gdppc`, or `pop`

    Returns
    -------
    pandas.DataFrame
        cleaned data containing the updated and newer data using versions of CIA WFB
        contained in both `df_old` and `df_new`.

    """

    msg = "Only able to clean 'gdp', 'gdppc' or 'pop'"
    assert varname in ["gdp", "gdppc", "pop"], msg

    msg = "`df_old` should be older than `df_new`; check the columns `wfb_year`"
    old_ver = df_old.wfb_year.max()
    new_ver = df_new.wfb_year.unique()[0]
    assert old_ver < new_ver, msg

    col_rename = {varname: varname + "_old", "wfb_year": "wfb_year_old"}
    col_select = ["ccode", "year", "wfb_year", varname]
    if varname != "pop":
        col_rename["usd_year"] = "usd_year_old"
        col_select.append("usd_year")
        if "usd_year" not in df_new.columns:
            df_new["usd_year"] = df_new["year"]

    df_old_merge = (
        df_old.loc[~pd.isnull(df_old.ccode), col_select]
        .rename(columns=col_rename)
        .set_index(["ccode", "year"])
    )
    merged_df = df_old_merge.merge(
        df_new.loc[~pd.isnull(df_new.ccode), col_select].set_index(["ccode", "year"]),
        left_index=True,
        right_index=True,
        how="outer",
    )

    merged_df.loc[pd.isnull(merged_df[varname]), "wfb_year"] = merged_df.loc[
        pd.isnull(merged_df[varname]), "wfb_year_old"
    ].values
    merged_df.loc[pd.isnull(merged_df[varname]), varname] = merged_df.loc[
        pd.isnull(merged_df[varname]), varname + "_old"
    ].values
    if varname != "pop":
        merged_df.loc[pd.isnull(merged_df["usd_year"]), "usd_year"] = merged_df.loc[
            pd.isnull(merged_df["usd_year"]), "usd_year_old"
        ].values

    return merged_df[col_select[2:]].sort_index().reset_index()
