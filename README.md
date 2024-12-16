# Weather Regimes: from Random to Multifractal

This GitHub repository hosts codes and notebooks accompanying a publication by P. Platzer, B. Chapron and G. Messori. All codes written by Paul Platzer. Running the notebooks on your machine allows to reproduce the figures of the article. It can also be used to reproduce the study on another geographical area, or to make parameter sensitivity tests.

The notebooks will call functions from the "/functions/" folder, and produce outputs in the "/output/data/" and "/output/figure/" folders. Note that you need to download ERA5 geopotential height data in order to reproduce the results. Here is a python code to do so:

```
import cdsapi
c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'area': [
            88.5, -80, 30,
            40,
        ],
        'time': ['00:00','12:00',],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022', '2023',
        ],
        'pressure_level': '500',
        'variable': 'geopotential',
        'download_format': 'unarchived'
    },
    'ERA5_2023.grib')   # put here your desired file name
```

More information [here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview) on how to download ERA5 reanalysis data realted to this work.

This article focuses on the multifractal properties of the atmospheric circulation, in particular on the local scaling exponent called "local dimension". This local dimension is a geometrical measure of the instantaneous number of degrees of freedom of the system. However, it also depends on local variations of sampling density. Therefore, this article develops a methodology to isolate the density-based variations of local dimension from the full local dimension variation, which are also multifractal-based. To do so, we generate stochastic twins of the atmosphere, which have similar density but do not share the multifractal properties of the atmospheric flow.

## North-Atlantic weather regimes

The article focuses on the following four winter-time North-Atlantic weather regimes, displayed below as the means of a Gaussian mixture model over the three leading empirical orthogonal functions of 10-days running averages of 500mb geopotential height anomaly from ERA5 reanalysis:

![Weather-regime maps](/outputs/figures/regimes_GMMattribution.png)

## Stochastic twins

The main proposition of the article is to generate stochastic twins of the atmosphere that bear similar density but are not multifractal. In this figure, one stochastic twin trajectory is compared to the full ERA5 trajectory over the period of study (1979-2023):

![Trajectories of ERA5 and stochastic twins](/outputs/figures/trajectories_EOFs.png)

## Local dimensions

Then we compute the local dimension from stochastic twins and from the original ERA5 reanalysis data. Here is a scatter plot:

![Scatter plot of ERA5-based and twins-based local dimension](/outputs/figures/d_hist_K450.png)

And here we show that the characteristic decrease of dimension around weather regime peak indices is mostly a density-based phenomenon, as the stochastic twins-based estimates of dimension anomaly (dashed for the average, dotted for the standard deviation) reproduce the behavior of the ERA-based estimates (full lines for the average +- standard deviation)

![Dimension anomaly decrease near peak WRI](/outputs/figures/d_life-cycle_twins_as_analogs.png)

We also reproduce the anticorrelation of local dimension anomaly versus peak weather regime index using the stochastic twins, showing again that this is primarily a density-based phenomenon, rather than a possible multifractal-based one.

![Dimension anomaly anti-correlated to peak WRI](/outputs/figures/d_peak_WRI.png)
