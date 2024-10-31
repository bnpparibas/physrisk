# Final version

import xarray as xr
import numpy as np
import scipy
from typing import Sequence
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator

import os


class RAINEuropeanWinterStorm:
    def __init__(self,
                scenarios: Sequence[str] = ["historical", "rcp4p5", "rcp8p5"],
                years: Sequence[int] = [2035, 2085],
                return_periods: Sequence[float] = [5, 10, 20, 50, 100, 200, 500]):
        self.return_periods = np.array(return_periods)
        self.base_periods =  np.array([5, 10, 20, 50])
        self.p = 1 /self.base_periods
        self.scenarios = scenarios
        self.years = years
        self.year_lookup = {
            -1: "1970-2000",
            2035: "2020-2050",
            2085: "2070-2100"
        }
        self.scenario_lookup = {
            "rcp4p5": "RCP45",
            "rcp8p5": "RCP85",
            "historical": "historical"
        }
                  
    def run(self):
        """
        This section runs through all processes for both scenario and historical data sets.
        It will run through 2035/2085 for scenario, but only 1970-2000 for historical.
        Outputs a data array for each combination (5 in total).

        Returns:
            da_list: list of 5 data arrays - each combination with scenario and historical. Data arrays have dimensions lat, lon, index.
        """        
        da_list = []
        for scenario in self.scenarios:
            if scenario == "historical":
                year = -1
                da = self.run_historical(scenario, year)
                da_list.append(da)
            else:
                for year in self.years:
                    da = self.run_scenario(scenario, year)
                    #da.to_zarr("filename.zarr")
                    da_list.append(da)
        return da_list
             
                    
    def run_historical(self, scenario, year):
        """        
        This section completes all processes (extrapolation, interpolation, data array production) on historical data.
        Historical data exists as absolute wind speed return levels for a given return period.
        run_historical() is different to run_scenario() as it does not require conversion from relative data.
        
        It goes through the following steps:
        - historical datasets are loaded
        - data for return period 5 to 50 are used to extrapolate up to 500 year (extrapolated), using a curve fit function and GEV_fit(). 
        - the newly extrapolated data is joined to the 5 to 50 year data to create a list of data sets (combined).
        - this data is interpolated to produce finer resolution points for each return period (interpolated).
        - this function also arranges the list of data sets into a single data array.

        Args:
            scenario (string): This defines the scenario we want to calculate for ie Historical, RCP 4.5, RCP 8.5. This will only run historical.
            year (integer): This defines the year we want to calculate for ie 1985 (1970-2000) in this case.

        Returns:
            data array: A data array with dimensions lat, lon, index. Index describes the return period for each data set within the data array. Holds windspeed data.
        """        
        datasets = self.datasets_for_returns(scenario, year)
        extrapolated = self.extrapolate(datasets)
        combined = datasets + extrapolated
        result = self.interpolate(combined)
        return result   
    
    
    def run_scenario(self, scenario, year):
        """
        This section completes all processes (conversion, extrapolation, interpolation) on scenario data.
        Scenario data exist as increase/decrease in exceedence probability relative to the hisotrical set.
        There is RCP 4.5 and RCP 8.5 for both 2035 and 2085.
        run_scenario() is different to run_historical() as it requires conversion from relative data.
        
        It goes through the following steps:
        - historical data is loaded
        - scenario data is loaded
        - scenario data is converted from relative probabilities to absolute wind speeds through comparison with historical data (historical, exceed_datasets, converted).
        - scenario data now has the same format at historical data so it is treated the same.
        - data for return period 5 to 50 are used to extrapolate up to 500 year (extrapolated), using a curve fit function and GEV_fit(). 
        - the newly extrapolated data is joined to the 5 to 50 year data to create a list of data sets (combined).
        - this data is interpolated to produce finer resolution points for each return period (interpolated).
        - this function also arranges the list of data sets into a single data array.

        Args:
            scenario (string): This defines the scenario we want to calculate for ie Historical, RCP 4.5, RCP 8.5. This will only run through RCP4.5, RCP8.5.
            year (integer): This defines the year we want to calculate for ie 2035, 2085.

        Returns:
            data array: A data array with dimensions lat, lon, index. Index describes the return period for each data set within the data array. Holds windspeed data.
        """        
        historical = self.get_historical()
        exceed_datasets = self.datasets_for_returns(scenario, year)
        converted = self.convert_datasets(exceed_datasets, historical)
        extrapolated = self.extrapolate(converted)
        combined = converted + extrapolated
        result = self.interpolate(combined)
        return result
    
    # This will retrieve initial scenario or historical data depending on the input            
    def datasets_for_returns(self, scenario, year):
        intro = ""
        if scenario == "historical":
            intro = "return_level"
        else:
            intro = "probability_change"
        
        result = [xr.open_dataset(f"wind_speed_{intro}_{r}_{self.scenario_lookup[scenario]}_{self.year_lookup[year]}.nc")[f"sfcWindmax_{r}_return_level"] for r in ["5yr", "10yr" , "20yr" , "50yr"]]
        return result    
    
    # Only needed for scenario data processing
    def get_historical(self):      
        result = [xr.open_dataset(f"wind_speed_return_level_{r}_historical_1970-2000.nc")[f"sfcWindmax_{r}_return_level"] for r in ["5yr", "10yr" , "20yr" , "50yr"]]
        return result
        
    
    def convert_datasets(self, exceed_datasets, historical):
        """
        Two types of datasets are used in this document - historical and scenario. 
        The historical data has wind speed return level data stored as real values.
        The scenario data exists as changes in probability relative to the historical values.
        Therefore, we need to convert scenario data from relative to absolute values so we can carry on further processes.
        
        In this, for every pixel (103x106 here) we produce a GEV curve (defined in GEV_fit()) fitted to historical data, then interpolate along the curve to give the new scenario point.
        This is done by using the new value of 'p', given by the change in p recorded in the original relative scenario data.

        Args:
            exceed_datasets (dataset): Datasets holding wind speed return level data as 'mean change of exceedance probability' relative to historical values.
            historical (dataset): Datasets holding wind speed return level data (ie speed of a 1 in 50 year storm, for p=1/50)

        Returns:
            converted_datasets: The scenario datasets with newly calculated absolute wind speed return values, as opposed to values relative to historical probabilities.
        """
        # Create empty datasets for newly calculated absolute return value data
        ds5_abs = xr.zeros_like(exceed_datasets[0])
        ds10_abs = xr.zeros_like(exceed_datasets[1])
        ds20_abs = xr.zeros_like(exceed_datasets[2])
        ds50_abs = xr.zeros_like(exceed_datasets[3])
        
        # With this dataset, we expect dimensions y-103 and x-106
        for i in range(exceed_datasets[0].y.size):
            for j in range(exceed_datasets[0].x.size):
                ds5_value = historical[0][0][i,j].data
                ds10_value = historical[1][0][i,j].data
                ds20_value = historical[2][0][i,j].data
                ds50_value = historical[3][0][i,j].data
                
                return_levels = [ds5_value, ds10_value, ds20_value, ds50_value]
                
                params = curve_fit(self.GEV_fit, self.p, return_levels)
                [mu, xi, sigma] = params[0]
                
                ds5_p_change_value = exceed_datasets[0][0][i,j].data
                ds10_p_change_value = exceed_datasets[1][0][i,j].data
                ds20_p_change_value = exceed_datasets[2][0][i,j].data
                ds50_p_change_value = exceed_datasets[3][0][i,j].data

                x_5=self.GEV_fit(0.2+ds5_p_change_value, mu, xi, sigma)
                x_10=self.GEV_fit(0.1+ds10_p_change_value, mu, xi, sigma)
                x_20=self.GEV_fit(0.05+ds20_p_change_value, mu, xi, sigma)
                x_50=self.GEV_fit(0.02+ds50_p_change_value, mu, xi, sigma)

                ds5_abs[0][i,j]=x_5
                ds10_abs[0][i,j]=x_10
                ds20_abs[0][i,j]=x_20
                ds50_abs[0][i,j]=x_50
                
        converted_datasets = [ds5_abs, ds10_abs, ds20_abs, ds50_abs]
        return converted_datasets
            
            
    def GEV_fit(self, p, mu, xi, sigma):
        x_p = mu - (sigma/xi)*(1-(-np.log(1-p))**(-xi))
        return x_p


    def extrapolate(self, datasets):    
        """
        The purpose of this section is to take datasets, which hold wind speed return values (absolute values, not relative), and extend return periods from 5-50 to 100-500.
        This is done by fitting the data to a GEV curve (defined in GEV_fit()) and then using this curve to calculate return values for p=1/100, 1/200, 1/500.
        This results in artificially high return period datasets.        

        Args:
            datasets (dataset): These original datasets contain wind speed return values for 5-year to 50-year return periods. They form the basis to extrapolate to 500 years.
            
        Returns:
            new_datasets: The freshly produced artificial datasets which hold wind speed return values extrapolated for 100, 200, and 500 year return periods.
        """   
        # Create empty datasets for newly calculated return value data              
        ds100=xr.zeros_like(datasets[0])
        ds200=xr.zeros_like(datasets[0])
        ds500=xr.zeros_like(datasets[0])
                
        # Produce fitted and predicted values for 100/200/500 return period datasets
        # With this dataset, we expect dimensions y-103 and x-106
        for i in range(datasets[0].y.size):
            for j in range(datasets[0].x.size):
                ds5_value = datasets[0][0][i,j].data
                ds10_value = datasets[1][0][i,j].data
                ds20_value = datasets[2][0][i,j].data
                ds50_value = datasets[3][0][i,j].data
                
                return_levels = [ds5_value, ds10_value, ds20_value, ds50_value]
                
                params = curve_fit(self.GEV_fit, self.p, return_levels)
                [mu, xi, sigma] = params[0]
                
                x_100 = self.GEV_fit(1/100, mu, xi, sigma)
                x_200 = self.GEV_fit(1/200, mu, xi, sigma)
                x_500 = self.GEV_fit(1/500, mu, xi, sigma)
                
                ds100[0][i,j]=x_100
                ds200[0][i,j]=x_200
                ds500[0][i,j]=x_500

        new_datasets = [ds100, ds200, ds500]
        return new_datasets


    def interpolate(self, combined):
        """
        This function interpolates datasets, the size of which are based on the original pixel size of raw datasets (103x106 here).
        We produce a higher resolution dataset through interpolation - 500 x 500 in this case.
        Furthermore, datasets are also transformed to purely lat, lon dependent rather than x,y.

        Args:
            combined (dataset): This makes up a full dataset per scenario with the original 5-50 year return periods with newer extrapolated 100-500 year periods.

        Returns:
            da_comb: All 'slices' of return values for different return periods are put together in one data array.
                    The data array has dimension lat, lon, and index. Index is the return period (ie 5, 10, 20 etc.) 
        """          
        return_levels = []
        interpolated_das = [0,0,0,0,0,0,0]
        for i in range(7):
            min_lat = combined[i].lat.min().data
            min_lon = combined[i].lon.min().data
            max_lat = combined[i].lat.max().data
            max_lon = combined[i].lon.max().data
            
            lon_array = combined[i].lon.to_numpy()
            np.reshape(lon_array,10918)

            lat_array = combined[i].lat.to_numpy()
            np.reshape(lat_array,10918)

            windspeed_array = combined[i][0].to_numpy()
            np.reshape(windspeed_array,10918)

            X = np.linspace(min_lon, max_lon,500) 
            Y = np.linspace(min_lat, max_lat,500) 
            X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
            interp = LinearNDInterpolator(list(zip(np.reshape(lon_array,10918), np.reshape(lat_array,10918))), np.reshape(windspeed_array,10918))
            Z = interp(X, Y)

            da = xr.DataArray(data=Z, coords = {'lat':(['lat','lon'],Y), 'lon':(['lat','lon'],X)}, dims = ['lat','lon'])
            
            interpolated_das[i] = da
            
        # Creating a combined data array with all data
        # We currently have slices but we want a cube      
        combined = np.zeros((7, 500, 500))
        returns = [5, 10, 20, 50, 100, 200, 500]
        
        for i in range(7):
            combined[i, :, :] = interpolated_das[i]
        da_comb = xr.DataArray(data=combined, coords = {"index": returns, "lat": Y[:, 0], "lon": X[0, :]}, dims = ['index','lat','lon'])
        
        return da_comb
        

