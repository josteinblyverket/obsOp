##!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import time
import h5py
import netCDF4
import scipy
import pyproj
import pyresample
import datetime
import torch
import torchsummary
import torch_geometric
import numpy as np

from Data_generator_GNN_prediction import *
from GNN_GAT import *

# In[2]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if a GPU is available
print("Using device: "  + str(device))

# # Constants

# In[3]:

# # List dates


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Get Surfex coordinates

# In[6]:


class get_surfex_coordinates():
    def __init__(self, paths):
        self.inputgrid = paths["surfex_grid"] + "PGD.nc"
    #
    def sfx2areadef(self, lat0, lon0, latori, lonori, xx, yy):
        proj2 = "+proj=lcc +lat_1=%.2f +lat_2=%.2f +lat_0=%.2f +lon_0=%.2f +units=m +ellps=WGS84 +no_defs" % (lat0,lat0,lat0,lon0)
        p2 = pyproj.Proj(proj2, preserve_units = False)
        origo = p2(lonori.data, latori.data)
        extent = origo + (origo[0] + xx[-1,-1], origo[1] + yy[-1,-1])
        area_def = pyresample.geometry.AreaDefinition("id2", "hei2", "lcc", proj2, xx.shape[1], yy.shape[0], extent)
        return(area_def)
    #
    def getSFXgrid(self):
        with netCDF4.Dataset(self.inputgrid, "r") as nc:
            areadef = self.sfx2areadef(lat0 = nc["LAT0"][0], lon0 = nc["LON0"][0], latori = nc["LATORI"][0], lonori = nc["LONORI"][0], xx = nc["XX"][:], yy = nc["YY"][:])
        return(areadef)
    #
    def __call__(self):
        areadef = self.getSFXgrid()
        lon, lat = areadef.get_lonlats()
        Surfex_coord = {}
        Surfex_coord["proj4_string"] = areadef.proj4_string
        Surfex_coord["crs"] = areadef.crs
        Surfex_coord["lon"] = lon
        Surfex_coord["lat"] = lat
        transform_to_surfex = pyproj.Transformer.from_crs(pyproj.CRS.from_proj4("+proj=latlon"), Surfex_coord["crs"], always_xy = True)
        Surfex_coord["xx"], Surfex_coord["yy"] = transform_to_surfex.transform(Surfex_coord["lon"], Surfex_coord["lat"])
        Surfex_coord["x"] = Surfex_coord["xx"][0,:]
        Surfex_coord["y"] = Surfex_coord["yy"][:,0]
        return(Surfex_coord)


# # Make model parameters



class make_model_parameters():
    def __init__(self, AMSR2_frequency, filename_normalization, predictors, activation, weight_initializer, conv_filters, batch_normalization, attention_heads):
        self.AMSR2_frequency = AMSR2_frequency
        self.filename_normalization = filename_normalization
        self.predictors = predictors
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.conv_filters = conv_filters
        self.batch_normalization = batch_normalization
        self.attention_heads = attention_heads
    #
    def load_normalization_stats(self):
        normalization_stats = {}
        with h5py.File(self.filename_normalization) as hdf:
            for var in hdf:
                normalization_stats[var] = hdf[var][()]
        return(normalization_stats)
    #
    def make_list_predictors(self):
        list_predictors = self.predictors["constants"] + self.predictors["atmosphere"] + self.predictors["ISBA"]
        for pred in self.predictors:
            if (pred != "constants") and (pred != "atmosphere") and (pred != "ISBA"):
                for lay in self.predictors[pred]:
                    var_name = pred + str(lay) + "_ga"
                    list_predictors = list_predictors + [var_name]
        return(list_predictors)
    #
    def make_list_targets(self):
        list_targets = ["AMSR2_BT" + self.AMSR2_frequency + "H", "AMSR2_BT" + self.AMSR2_frequency + "V"]
        return(list_targets)
    #
    def make_model_parameters(self, list_predictors, list_targets):
        model_params = {"list_predictors": list_predictors,
                        "list_targets": list_targets,
                        "activation": self.activation,
                        "weight_initializer": self.weight_initializer,
                        "conv_filters": self.conv_filters,
                        "batch_normalization": self.batch_normalization,
                        "heads": self.attention_heads,
                        }
        return(model_params)
    #
    def __call__(self):
        normalization_stats = self.load_normalization_stats()
        list_predictors = self.make_list_predictors()
        list_targets = self.make_list_targets()
        model_params = self.make_model_parameters(list_predictors, list_targets)
        return(normalization_stats, model_params)


# # Make loader

# In[8]:


class make_loader():
    def __init__(self, AMSR2_frequency, AMSR2_footprint_radius, list_predictors, normalization_stats, date_task, paths):
        self.AMSR2_frequency = AMSR2_frequency
        self.AMSR2_footprint_radius = AMSR2_footprint_radius 
        self.list_predictors = list_predictors
        self.normalization_stats = normalization_stats
        self.date_task = date_task
        self.paths = paths
        self.filename_data = self.paths["training"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + date_task[6:8] + "/" + "03" + "/" + "001" + "/" + "Graphs_" + self.date_task + ".h5"
    #
    def Number_of_samples_and_footprint_coordinates(self):
        Graphs_coord = {}
        Targets = {}
        #
        with h5py.File(self.filename_data, "r") as hdf:
            Number_of_graphs = len(hdf["AMSR2_xx"][()])
            #
            Graphs_coord["xx"] = np.full(Number_of_graphs, np.nan)
            Graphs_coord["yy"] = np.full(Number_of_graphs, np.nan)
            Targets["AMSR2_BT" + self.AMSR2_frequency + "H"] = np.full(Number_of_graphs, np.nan)
            Targets["AMSR2_BT" + self.AMSR2_frequency + "V"] = np.full(Number_of_graphs, np.nan)
            #
            Graphs_coord["xx"] = hdf["AMSR2_xx"][()]
            Graphs_coord["yy"] = hdf["AMSR2_yy"][()]
            Targets["AMSR2_BT" + self.AMSR2_frequency + "H"] = hdf["AMSR2_BT" + self.AMSR2_frequency + "H"][()]
            Targets["AMSR2_BT" + self.AMSR2_frequency + "V"] = hdf["AMSR2_BT" + self.AMSR2_frequency + "V"][()]
            #
            return(Number_of_graphs, Graphs_coord, Targets)
    #
    def make_data_generator_parameters(self, filename_data):
        data_generator_params = {"filename_data": filename_data,
                                 "footprint_radius": self.AMSR2_footprint_radius,
                                 "list_predictors": self.list_predictors,
                                 "normalization_stats": self.normalization_stats}
        return(data_generator_params)
    #
    def create_data_loader(self, Number_of_graphs, data_generator_params):
        dataset = Data_generator_GNN_prediction(**data_generator_params)
        return(dataset)
    #
    def __call__(self):
        Number_of_graphs, Graphs_coord, Targets = self.Number_of_samples_and_footprint_coordinates()
        params_valid = self.make_data_generator_parameters(self.filename_data)
        valid_loader = self.create_data_loader(Number_of_graphs, params_valid)
        return(Number_of_graphs, Graphs_coord, Targets, valid_loader)


# # Make predictions

# In[9]:


class make_predictions():
    def __init__(self, list_targets, model, valid_loader, paths, normalization_stats, device):
        self.list_targets = list_targets
        self.model = model
        self.valid_loader = valid_loader
        self.paths = paths
        self.normalization_stats = normalization_stats
        self.device = device
    #
    def unnormalize(self, unnormalized_predictions):
        normalized_predictions = np.full(np.shape(unnormalized_predictions), np.nan)
        for vi, var in enumerate(self.list_targets):
            normalized_predictions[:, vi] = unnormalized_predictions[:, vi] * (self.normalization_stats[var + "_max"] - self.normalization_stats[var + "_min"]) + self.normalization_stats[var + "_min"]
        return(normalized_predictions)
    #
    def predictions(self):
        self.model.eval()
        print(len(self.valid_loader))
        with torch.no_grad(), torch.amp.autocast(device_type = "cuda"):
            for batch_valid in self.valid_loader:
                data_batch = batch_valid.to(self.device)
                x, y, a, batch = data_batch.x, data_batch.y, data_batch.edge_index, data_batch.batch
                print("x,y,a", np.shape(x))
                unnormalized_predictions = self.model(x, a, batch)        
                print("unnormalized predictions")
        unnormalized_predictions = unnormalized_predictions.cpu().numpy()            
        print("to numpy")         
        return(unnormalized_predictions)
    #
    def __call__(self):
        unnormalized_predictions = self.predictions()
        normalized_predictions = self.unnormalize(unnormalized_predictions)
        return(normalized_predictions)


#  # Gridding predictions

# In[10]:


class gridding_predictions():
    def __init__(self, date_task, AMSR2_footprint_radius, Surfex_coord, list_targets, Targets, Graphs_coord, predictions, paths):
        self.date_task = date_task
        self.AMSR2_footprint_radius = AMSR2_footprint_radius 
        self.Surfex_coord = Surfex_coord
        self.list_targets = list_targets
        self.idx_nan = np.logical_or(np.isnan(Graphs_coord["xx"]) == True, np.isnan(Graphs_coord["yy"]) == True)
        self.idx_nan_extend = np.repeat(np.expand_dims(self.idx_nan, axis = 1), len(self.list_targets), axis = 1)
        self.Targets = Targets
        for var in self.Targets:
            self.Targets[var] = self.Targets[var][self.idx_nan == False]
        self.Graphs_xx = Graphs_coord["xx"][self.idx_nan == False]
        self.Graphs_yy = Graphs_coord["yy"][self.idx_nan == False]
        self.predictions = predictions[self.idx_nan_extend == False]
        self.paths = paths
    #
    def nearest_neighbor_indexes(self):
        pred_xx = np.expand_dims(self.Graphs_xx, axis = 1)
        pred_yy = np.expand_dims(self.Graphs_yy, axis = 1)
        Surfex_xx = np.expand_dims(np.ndarray.flatten(self.Surfex_coord["xx"]), axis = 1)
        Surfex_yy = np.expand_dims(np.ndarray.flatten(self.Surfex_coord["yy"]), axis = 1)
        #
        coord_input = np.concatenate((pred_xx, pred_yy), axis = 1)
        coord_output = np.concatenate((Surfex_xx, Surfex_yy), axis = 1)
        #
        tree = scipy.spatial.KDTree(coord_input)
        dist, idx = tree.query(coord_output)
        return(dist, idx)
    #
    def project_predictions_onto_Surfex_domain(self):
        dist, idx = self.nearest_neighbor_indexes()
        Gridded_distance = np.reshape(dist, (len(self.Surfex_coord["y"]), len(self.Surfex_coord["x"])), order = "C")
        #
        Gridded_targets = {}
        for vi, var in enumerate(self.list_targets):
            Target_interp = np.ndarray.flatten(self.Targets[var])[idx]
            Gridded_targets[var] = np.reshape(Target_interp, (len(self.Surfex_coord["y"]), len(self.Surfex_coord["x"])), order = "C")
            Gridded_targets[var][Gridded_distance > self.AMSR2_footprint_radius] = np.nan
        #
        Gridded_predictions = {}
        for vi, var in enumerate(self.list_targets):
            Pred_interp = np.ndarray.flatten(predictions[:, vi])[idx]
            Gridded_predictions[var] = np.reshape(Pred_interp, (len(self.Surfex_coord["y"]), len(self.Surfex_coord["x"])), order = "C")
            Gridded_predictions[var][Gridded_distance > self.AMSR2_footprint_radius] = np.nan
        #
        #Gridded_distance[Gridded_distance > self.AMSR2_footprint_radius] = np.nan
        #
        return(Gridded_predictions, Gridded_targets, Gridded_distance)
    #
    def write_netCDF(self, Gridded_predictions, Gridded_targets, Gridded_distance):
        path_output = self.paths["output"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + date_task[6:8] + "/" + "03" + "/" + "001" + "/"
        if os.path.exists(path_output) == False:
            os.system("mkdir -p " + path_output)
        output_filename = path_output + "Predictions_" + self.date_task + ".nc"
        if os.path.isfile(output_filename):
            os.system("rm " + output_filename)
        #
        with netCDF4.Dataset(str(output_filename), "w", format = "NETCDF4") as output_netcdf:
            x = output_netcdf.createDimension("x", len(self.Surfex_coord["x"]))
            y = output_netcdf.createDimension("y", len(self.Surfex_coord["y"]))
            #
            Outputs = vars()
            for var in ["x", "y"]:
                Outputs[var] = output_netcdf.createVariable(var, "d", (var))
                Outputs[var].units = "meters" 
                Outputs[var].standard_name = "projection_" + var + "_coordinates"
                Outputs[var] = np.copy(self.Surfex_coord[var])
            #
            for var in ["lat", "lon"]:
                Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
                if var == "lat":
                    Outputs[var].standard_name = "latitude"
                    Outputs[var].unit = "degrees_north"
                else:
                    Outputs[var].standard_name = "longitude"
                    Outputs[var].units = "degrees_east"
                Outputs[var][:,:] = np.copy(self.Surfex_coord[var])
            #
            for var in Gridded_targets:
                Outputs["Target_" + var] = output_netcdf.createVariable("Target_" + var, "d", ("y", "x"))
                Outputs["Target_" + var].units = "Kelvins"
                Outputs["Target_" + var].standard_name = "Brightness temperature"
                Outputs["Target_" + var][:,:] = np.copy(Gridded_targets[var])
            #
            for var in Gridded_predictions:
                Outputs["Prediction_" + var] = output_netcdf.createVariable("Prediction_" + var, "d", ("y", "x"))
                Outputs["Prediction_" + var].units = "Kelvins"
                Outputs["Prediction_" + var].standard_name = "Brightness temperature"
                Outputs["Prediction_" + var][:,:] = np.copy(Gridded_predictions[var])
            #
            Outputs["Distance_to_footprint_center"] = output_netcdf.createVariable("Distance_to_footprint_center", "d", ("y", "x"))
            Outputs["Distance_to_footprint_center"].units = "meters"
            Outputs["Distance_to_footprint_center"].standard_name = "Distance_to_footprint_center"
            Outputs["Distance_to_footprint_center"][:,:] = np.copy(Gridded_distance)
    #
    def __call__(self):
        Gridded_predictions, Gridded_targets, Gridded_distance = self.project_predictions_onto_Surfex_domain()
        self.write_netCDF(Gridded_predictions, Gridded_targets, Gridded_distance)


# # Data processing

# In[11]:


tt0 = time.time()
#
Surfex_coord = get_surfex_coordinates(paths)()
#
checkpoint = torch.load(paths["model"] + "GNN_model_" + AMSR2_frequency.split('.')[0] + "GHz.pth", weights_only = False) 
#
normalization_stats, model_params = make_model_parameters(AMSR2_frequency = AMSR2_frequency, 
                                                          filename_normalization = filename_normalization, 
                                                          predictors = predictors, 
                                                          activation = activation, 
                                                          weight_initializer = weight_initializer, 
                                                          conv_filters = conv_filers, 
                                                          batch_normalization = batch_normalization,
                                                          attention_heads = attention_heads)()
#
list_dates = make_list_dates(date_min, date_max)
for date_task in list_dates:
    #try:
        Number_of_graphs, Graphs_coord, Targets, valid_loader = make_loader(AMSR2_frequency = AMSR2_frequency,
                                                                            AMSR2_footprint_radius = AMSR2_footprint_radius, 
                                                                            list_predictors = model_params["list_predictors"], 
                                                                            normalization_stats = normalization_stats,
                                                                            date_task = date_task,
                                                                            paths = paths)()
        #
        GNN_model = GNN_GAT(**model_params).to(device)
        GNN_model.load_state_dict(checkpoint["model_state_dict"])
        #
        print("Pred starts")
        t0 = time.time()
        predictions = make_predictions(list_targets = model_params["list_targets"], 
                                    model = GNN_model, 
                                    valid_loader = valid_loader, 
                                    paths = paths, 
                                    normalization_stats = normalization_stats, 
                                    device = device)()
        #
        print("Pred OK")
        t1 = time.time()
        print(date_task, t1 - t0)
        #
        gridding_predictions(date_task = date_task, 
                            AMSR2_footprint_radius = AMSR2_footprint_radius,
                            Surfex_coord = Surfex_coord, 
                            list_targets = model_params["list_targets"], 
                            Targets = Targets, 
                            Graphs_coord = Graphs_coord, 
                            predictions = predictions, 
                            paths = paths)()
    #except:
    #   pass
#
ttf = time.time()
print("Total computing time: ", ttf - tt0)

def main():


    experiment_name = "v6"
    AMSR2_frequency = "18.7"
    #
    #function_path = "/lustre/storeB/users/josteinbl/MLP/GNN_data/v1/"
    #sys.path.insert(0, function_path)    
    #
    paths = {}
    paths["training"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/archive/"
    paths["normalization"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/Normalization/"
    paths["model"] = "/lustre/storeB/project/nwp/H2O/wp3/Deep_learning_predictions/GNN/Models_static/" + experiment_name + "/"
    paths["surfex_grid"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/climate/"
    paths["output"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/archive/" #+ "/Predictions_" + AMSR2_frequency.split('.')[0] + "GHz/"
    #
    filename_normalization = paths["normalization"] + "Stats_normalization_20200901_20220531.h5"
    #
    for var in paths:
        if os.path.isdir(paths[var]) == False:
            os.system("mkdir -p " + paths[var])
    #
    AMSR2_all_frequencies = ["6.9", "7.3", "10.7", "18.7", "23.8", "36.5"]
    AMSR2_all_footprint_radius = np.array([35 + 62, 35 + 62, 24 + 42, 14 + 22, 11 + 19, 7 + 12]) * 0.25 * 1000  # 0.5 * mean diameter (0.5 * (major + minor)), *1000 => km to meters
    AMSR2_footprint_radius = AMSR2_all_footprint_radius[AMSR2_all_frequencies.index(AMSR2_frequency)]

    # # Model parameters

    date_min = "20220704"
    date_max = "20220705"
    subsampling = "1"
    #
    def he_normal_init(weight):
        torch.nn.init.kaiming_normal_(weight, mode = "fan_out", nonlinearity = "relu")
    weight_initializer = he_normal_init
    #
    activation = torch.nn.ReLU()
    shuffle = True
    conv_filers = [32, 64, 32]
    batch_size = 512
    batch_normalization = True
    attention_heads = 4
    #
    predictors = {}
    predictors["constants"] = ["ZS", "PATCHP1", "PATCHP2", "FRAC_LAND_AND_SEA_WATER", "Distance_to_footprint_center"]
    predictors["atmosphere"] = ["lwe_thickness_of_atmosphere_mass_content_of_water_vapor"]
    #predictors["ISBA"] = ["Q2M_ISBA", "DSN_T_ISBA", "LAI_ga", "TS_ISBA", "PSN_ISBA"]
    predictors["ISBA"] = ["LAI_ga", "DSN_T_ISBA", "WSN_T_ISBA"]
    predictors["TG"] = [1, 2]
    predictors["WG"] = [1, 2]
    predictors["WGI"] = [1, 2]
    #predictors["WSN_VEG"] = [1, 6, 12]
    predictors["RSN_VEG"] = [1, 6, 12]
    predictors["HSN_VEG"] = [1, 6, 12]
    predictors["SNOWTEMP"] = [1, 6, 12]
    predictors["SNOWLIQ"] = [1, 6, 12]


if __name__ == "__main__":

    main()
