#!/usr/bin/env python
# coding: utf-8


import os
import time
import glob
import h5py
import scipy
import pyproj
import netCDF4
import datetime
import pyresample
import numpy as np
import matplotlib.pyplot as plt



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

# In[43]:


class get_surfex_coordinates():
    def __init__(self, crs, surfex_PGD_variables, paths):
        self.crs = crs
        self.surfex_PGD_variables = surfex_PGD_variables
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
    def load_PGD_variables(self):
        Surfex_PGD = {}
        with netCDF4.Dataset(self.inputgrid, "r") as nc:
            for var in self.surfex_PGD_variables:
                Surfex_PGD[var] = np.flipud(nc.variables[var][:,:])
        return(Surfex_PGD)
    #
    def __call__(self):
        areadef = self.getSFXgrid()
        #
        lon, lat = areadef.get_lonlats()
        #
        Surfex_PGD = self.load_PGD_variables()
        #
        Surfex_coord = {}
        Surfex_coord["proj4_string"] = areadef.proj4_string
        Surfex_coord["crs"] = areadef.crs
        Surfex_coord["lon"] = lon
        Surfex_coord["lat"] = lat
        #
        transform_to_surfex = pyproj.Transformer.from_crs(self.crs["latlon"], Surfex_coord["crs"], always_xy = True)
        Surfex_coord["xx"], Surfex_coord["yy"] = transform_to_surfex.transform(Surfex_coord["lon"], Surfex_coord["lat"])
        Surfex_coord["x"] = Surfex_coord["xx"][0,:]
        Surfex_coord["y"] = Surfex_coord["yy"][:,0]
        Surfex_coord["x_min"] = np.min(Surfex_coord["x"])
        Surfex_coord["x_max"] = np.max(Surfex_coord["x"])
        Surfex_coord["y_min"] = np.min(Surfex_coord["y"])
        Surfex_coord["y_max"] = np.max(Surfex_coord["y"])
        #
        return(Surfex_coord, Surfex_PGD)


# # Read SURFEX data

# In[44]:


def read_surfex_data(date_task_hours_AMSR2, paths, predictor_variables, mbr):
    Surfex_data = {}
    filename_constants = paths["surfex"] + "2022/07/03/00/000/SURFOUT.20220703_03h00.nc"
    previous_hours = "{:02d}".format(int(date_task_hours_AMSR2[9:11]) - 3)
    path_task = paths["surfex"] + date_task_hours_AMSR2[0:4] + "/" + date_task_hours_AMSR2[4:6] + "/" + date_task_hours_AMSR2[6:8] + "/" +  previous_hours + "/"
    #
    with netCDF4.Dataset(path_task + "%s"%mbr + "/" + "SURFOUT." + date_task_hours_AMSR2[0:8] + "_" + date_task_hours_AMSR2[9:11] + "h00.nc", "r") as nc:
        for var in ["PATCHP1", "PATCHP2"]:
            if var in nc.variables:
                if var == "PATCHP1":
                    var_data = np.flipud(nc.variables["PATCH"][0,:,:])
                elif var == "PATCHP2":
                    var_data = np.flipud(nc.variables["PATCH"][1,:,:])
                
                var_data[var_data.mask == True] = 0
                Surfex_data[var] = np.expand_dims(var_data, axis = 0)
            else:
                with netCDF4.Dataset(filename_constants, "r") as nc_constants:
                    if var == "PATCHP1":
                        var_data = np.flipud(nc_constants.variables["PATCH"][0,:,:])
                    elif var == "PATCHP2":
                        var_data = np.flipud(nc_constants.variables["PATCH"][1,:,:])

                    #var_data = np.flipud(nc_constants.variables[var][:,:])
                    var_data[var_data.mask == True] = 0
                    Surfex_data[var] = np.expand_dims(var_data, axis = 0)
        #
        for var in predictor_variables:
            #try:
                if (var in nc.variables) or (var.replace("_ga", "") in nc.variables):
                    if "_ga" in var:
                        var_data_P1 = np.flipud(nc.variables[var.replace("_ga", "")][0,:,:])
                        var_data_P2 = np.flipud(nc.variables[var.replace("_ga", "")][0,:,:])

                        var_data_P1 = np.expand_dims(var_data_P1, axis = 0)
                        var_data_P2 = np.expand_dims(var_data_P2, axis = 0)
                        Surfex_data[var] = np.nansum([Surfex_data["PATCHP1"] * var_data_P1, Surfex_data["PATCHP2"] * var_data_P2], axis = 0)
                        Surfex_data[var][var_data_P1.mask == True] = var_data_P2[var_data_P1.mask == True]
                        Surfex_data[var][var_data_P2.mask == True] = var_data_P1[var_data_P2.mask == True]
                        Surfex_data[var][np.logical_and(var_data_P1.mask == True, var_data_P2.mask == True)] = np.nan
                    else:
                        Surfex_data[var] = np.flipud(nc.variables[var][:,:])
                else:
                    with netCDF4.Dataset(path_task + "SURFOUT.nc", "r") as ncp:
                        if "_ga" in var:
                            var_data_P1 = np.flipud(ncp.variables[var.replace("_ga", "")][0,:,:])
                            var_data_P2 = np.flipud(ncp.variables[var.replace("_ga", "")][1,:,:])
                            var_data_P1 = np.expand_dims(var_data_P1, axis = 0)
                            var_data_P2 = np.expand_dims(var_data_P2, axis = 0)
                            Surfex_data[var] = np.nansum([Surfex_data["PATCHP1"] * var_data_P1, Surfex_data["PATCHP2"] * var_data_P2], axis = 0)
                            Surfex_data[var][var_data_P1.mask == True] = var_data_P2[var_data_P1.mask == True]
                            Surfex_data[var][var_data_P2.mask == True] = var_data_P1[var_data_P2.mask == True]
                            Surfex_data[var][np.logical_and(var_data_P1.mask == True, var_data_P2.mask == True)] = np.nan
                        else:
                            Surfex_data[var] = np.flipud(ncp.variables[var][:,:])
            #except:
            #    print("Variable not found: " + var)
            #    if (var == "SNOWTEMP9_ga") or (var == "SNOWLIQ9_ga"):
            #        pass
            #    else:
            #        sys.exit()
    #    
    return(Surfex_data)


# In[45]:

def calculate_WSN_T_ISBA(Surfex_data, n_soil_layers):
    WSN_T_ISBA = np.zeros(np.shape(Surfex_data["WSN_VEG1_ga"]))
    for layer in range(0, n_soil_layers):
        WSN_T_ISBA = WSN_T_ISBA + Surfex_data["WSN_VEG" + str(layer + 1) + "_ga"]
    WSN_T_ISBA[WSN_T_ISBA > 1e10] = np.nan
    return(WSN_T_ISBA)

# # Get MEPS variables

# In[46]:


class get_MEPS_data():
    def __init__(self, date_task, MEPS_leadtime, MEPS_dim_variables, MEPS_PL_variables, Surfex_coord, crs, paths):
        self.date_task = date_task
        self.MEPS_leadtime = MEPS_leadtime
        self.MEPS_dim_variables = MEPS_dim_variables
        self.MEPS_PL_variables = MEPS_PL_variables
        self.Surfex_coord = Surfex_coord
        self.crs = crs
        self.paths = paths
        self.filename = paths["MEPS"] + date_task[0:4] + "/" + date_task[4:6] + "/" + date_task[6:8] + "/" + "meps_det_2_5km_" + date_task + "T00Z.nc"
    #
    def nearest_neighbor_indexes(self, x_input, y_input, x_output, y_output):
        # x_input, y_input, x_output, and y_output must be vectors
        x_input = np.expand_dims(x_input, axis = 1)
        y_input = np.expand_dims(y_input, axis = 1)
        x_output = np.expand_dims(x_output, axis = 1)
        y_output = np.expand_dims(y_output, axis = 1)
        #
        coord_input = np.concatenate((x_input, y_input), axis = 1)
        coord_output = np.concatenate((x_output, y_output), axis = 1)
        #
        tree = scipy.spatial.KDTree(coord_input)
        dist, idx = tree.query(coord_output)
        #
        return(idx)
    #
    def calculate_vertically_integrated_variable(self, pressure_levels, var_3D):
        diff_pressure_extend = np.diff(pressure_levels)[:, np.newaxis, np.newaxis]
        integrated_var = np.sum(diff_pressure_extend * var_3D[1:len(pressure_levels),:,:], axis = 0)
        return(integrated_var)
    #
    def load_data(self):
        Dataset = {}
        if os.path.isfile(self.filename) == True:
            with netCDF4.Dataset(self.filename, "r") as nc:
                #
                for var in self.MEPS_dim_variables:
                    Dataset[var] = nc.variables[var][:]
                #
                for var in self.MEPS_PL_variables:
                    if "_pl" in var:
                        if nc.variables[var][:].ndim == 4:
                            output_var_name = ("vertically_integrated_" + var).replace("_pl", "")
                            var_data = nc.variables[var][self.MEPS_leadtime,:,:,:]
                            Dataset[output_var_name] = self.calculate_vertically_integrated_variable(Dataset["pressure"], var_data)
                    #
                    if var == "lwe_thickness_of_atmosphere_mass_content_of_water_vapor":
                        Dataset[var] = np.squeeze(nc.variables[var][self.MEPS_leadtime,:,:,:])
        return(Dataset)   
    #
    def projecting_MEPS_data_onto_Surfex_domain(self, Dataset):
        Dataset_on_Surfex_grid = {}
        transform_to_surfex = pyproj.Transformer.from_crs(self.crs["latlon"], self.Surfex_coord["crs"], always_xy = True)
        xx_surfex = np.ndarray.flatten(self.Surfex_coord["xx"])
        yy_surfex = np.ndarray.flatten(self.Surfex_coord["yy"])
        xx_MEPS_on_Surfex_grid, yy_MEPS_on_Surfex_grid = transform_to_surfex.transform(Dataset["longitude"], Dataset["latitude"])
        xx_MEPS_on_Surfex_grid = np.ndarray.flatten(xx_MEPS_on_Surfex_grid)
        yy_MEPS_on_Surfex_grid = np.ndarray.flatten(yy_MEPS_on_Surfex_grid)
        idx_MEPS_to_Surfex = self.nearest_neighbor_indexes(xx_MEPS_on_Surfex_grid, yy_MEPS_on_Surfex_grid, xx_surfex, yy_surfex)
        #
        for var in Dataset:
            if var not in self.MEPS_dim_variables:
                field_flat = np.ndarray.flatten(Dataset[var])
                field_interp = field_flat[idx_MEPS_to_Surfex]
                Dataset_on_Surfex_grid[var] = np.reshape(field_interp, (len(self.Surfex_coord["y"]), len(self.Surfex_coord["x"])), order = "C")
        #
        return(Dataset_on_Surfex_grid)
    #
    def __call__(self):
        Dataset = self.load_data()
        Dataset_on_Surfex_grid = self.projecting_MEPS_data_onto_Surfex_domain(Dataset)
        return(Dataset_on_Surfex_grid)


# # Read AMSR2 data

# In[47]:


class read_AMSR2_data():
    def __init__(self, filename, AMSR2_task_frequency):
        self.filename = filename
        self.AMSR2_task_frequency = AMSR2_task_frequency
    #
    def decode_CoRegistrationParameter(self, Astr):
        # from https://gitlab.met.no/met/obsklim/satellitt/microwave-sdr-to-osisaf-nc/-/blob/master/microwave_to_osisaf_nc/amsr2h5_l1b_reader.py
        ch_coreg = Astr.split(',')
        if len(ch_coreg) != 6:
            raise ValueError('Did not find expected 6 channels in co-registration parameter list')
        #
        coreg = dict()
        for ch in ch_coreg:
            try:
                ch_nam, ch_reg = ch.split('-', 1)
                ch_reg = float(ch_reg)
            except Exception as e:
                raise ValueError('Wrong format for the CoRegistration string %s (%s)' % (ch, e,))
            coreg[ch_nam] = ch_reg
        return(coreg)
    #
    def get_low_freq_latlon(self, lat89A, lon89A, A1, A2):
        # from https://gitlab.met.no/met/obsklim/satellitt/microwave-sdr-to-osisaf-nc/-/blob/master/microwave_to_osisaf_nc/amsr2h5_l1b_reader.py
        # compute lat/lon for a low-frequency channel from the 89A lat/lon
        #    (ref: AMSR2_Level1_Product_Format_EN.pdf page 4-16,4-17)
        # split the 89A positions in odd and even.
        #    WARNING: the numbering in the AMSR2 PUM is from 1, so their
        #                'even' (2,4,6,...) are (1,3,5,...) for Python
        evn_lat89A = np.deg2rad(lat89A[:, 1::2])
        evn_lon89A = np.deg2rad(lon89A[:, 1::2])
        odd_lat89A = np.deg2rad(lat89A[:, 0::2])
        odd_lon89A = np.deg2rad(lon89A[:, 0::2])
        # transform to Cartesian x,y,z coordinates
        P1 = np.dstack((np.cos(odd_lon89A) * np.cos(odd_lat89A), np.sin(odd_lon89A) * np.cos(odd_lat89A), np.sin(odd_lat89A)))
        P2 = np.dstack((np.cos(evn_lon89A) * np.cos(evn_lat89A), np.sin(evn_lon89A) * np.cos(evn_lat89A), np.sin(evn_lat89A)))
        # Get orthogonal base Ex,Ey,Ez
        Ex = P1
        Ez = np.cross(P1, P2)
        EzNorm = ((Ez ** 2).sum(axis = 2))**0.5
        for d in range(3):
            Ez[:, :, d] /= EzNorm
        Ey = np.cross(Ez, Ex)
        # Get Theta: angle between two consecutives 89A positions along the scan
        Theta = np.arccos((P1 * P2).sum(axis = 2))
        # Compute cartesian position of low-frequency channel with A1 and A2
        cA2T = np.cos(A2 * Theta)
        cA1T = np.cos(A1 * Theta)
        sA1T = np.sin(A1 * Theta)
        sA2T = np.sin(A2 * Theta)
        Pt = np.empty_like(Ex)
        for d in range(3):
            Pt[:, :, d] = cA2T * (cA1T * Ex[:, :, d] + sA1T * Ey[:, :, d]) + sA2T * Ez[:, :, d]
        # Transform back from Cartersian to Lat/Lon
        Lat = np.arcsin(Pt[:, :, 2])
        Lon = np.arctan2(Pt[:, :, 1], Pt[:, :, 0])
        return(np.rad2deg(Lat), np.rad2deg(Lon)) 
    #
    def __call__(self):
        AMSR2_dataset = {}
        with h5py.File(self.filename, "r") as hdf:
            lat89A = hdf["Latitude of Observation Point for 89A"][()]
            lon89A = hdf["Longitude of Observation Point for 89A"][()]
            coreg_str_A1 = hdf.attrs["CoRegistrationParameterA1"][0]
            coreg_str_A2 = hdf.attrs["CoRegistrationParameterA2"][0]
            coreg_A1 = self.decode_CoRegistrationParameter(coreg_str_A1)
            coreg_A2 = self.decode_CoRegistrationParameter(coreg_str_A2)
            A1 = coreg_A1[self.AMSR2_task_frequency.split(".")[0] + "G"]
            A2 = coreg_A2[self.AMSR2_task_frequency.split(".")[0] + "G"]
            lat_task_freq, lon_task_freq = self.get_low_freq_latlon(lat89A, lon89A, A1, A2)
            AMSR2_dataset["lat"] = lat_task_freq
            AMSR2_dataset["lon"] = lon_task_freq
            AMSR2_dataset["BT" + self.AMSR2_task_frequency + "H"] = hdf["Brightness Temperature (" + self.AMSR2_task_frequency + "GHz,H)"][()] * 0.01  # 0.01 = Scaling factor
            AMSR2_dataset["BT" + self.AMSR2_task_frequency + "V"] = hdf["Brightness Temperature (" + self.AMSR2_task_frequency + "GHz,V)"][()] * 0.01
            AMSR2_dataset["BT" + self.AMSR2_task_frequency + "H"][AMSR2_dataset["BT" + self.AMSR2_task_frequency + "H"] > 600] = np.nan
            AMSR2_dataset["BT" + self.AMSR2_task_frequency + "V"][AMSR2_dataset["BT" + self.AMSR2_task_frequency + "V"] > 600] = np.nan
        return(AMSR2_dataset)


# # Extract graphs

# In[48]:


class extract_graphs():
    def __init__(self, date_task, paths, AMSR2_task_frequency, static_dimension, crs, Surfex_coord, Surfex_data, MEPS_data):
        self.date_task = date_task
        self.paths = paths
        self.AMSR2_task_frequency = AMSR2_task_frequency
        self.static_dimension = static_dimension
        self.crs = crs
        self.Surfex_coord = Surfex_coord
        self.Surfex_data = Surfex_data
        self.MEPS_data = MEPS_data
        self.min_number_of_obs = 100
        self.max_hour = 6 # 6 AM
        self.transform_to_surfex = pyproj.Transformer.from_crs(self.crs["latlon"], self.Surfex_coord["crs"], always_xy = True)
        self.path_date = paths["AMSR2"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + self.date_task[6:8] + "/amsr2/jaxa/"
        self.dataset = sorted(glob.glob(self.path_date + "GW1AM2_" + self.date_task + "0*.h5"))
    #
    def AMSR2_on_surfex_domain(self, AMSR2_dataset):
        AMSR2_xx, AMSR2_yy = self.transform_to_surfex.transform(AMSR2_dataset["lon"], AMSR2_dataset["lat"])
        idx_x = np.logical_and(AMSR2_xx > self.Surfex_coord["x_min"], AMSR2_xx < self.Surfex_coord["x_max"])
        idx_y = np.logical_and(AMSR2_yy > self.Surfex_coord["y_min"], AMSR2_yy < self.Surfex_coord["y_max"])
        idx_domain = np.logical_and(idx_x == True, idx_y == True)
        N_obs = np.sum(idx_domain == True)
        #
        AMSR2_surfex_domain = {}
        if N_obs > self.min_number_of_obs:
            AMSR2_surfex_domain["N_obs"] = N_obs
            AMSR2_surfex_domain["xx"] = AMSR2_xx[idx_domain == True]
            AMSR2_surfex_domain["yy"] = AMSR2_yy[idx_domain == True]
            AMSR2_surfex_domain["lat"] = AMSR2_dataset["lat"][idx_domain == True]
            AMSR2_surfex_domain["lon"] = AMSR2_dataset["lon"][idx_domain == True]
            AMSR2_surfex_domain["BT" + self.AMSR2_task_frequency + "H"] = AMSR2_dataset["BT" + self.AMSR2_task_frequency + "H"][idx_domain == True]
            AMSR2_surfex_domain["BT" + self.AMSR2_task_frequency + "V"] = AMSR2_dataset["BT" + self.AMSR2_task_frequency + "V"][idx_domain == True]
        return(AMSR2_surfex_domain)
    #
    def calculate_distances(self, AMSR2_surfex_domain):
        AMSR2_grid_points = np.stack([AMSR2_surfex_domain["xx"], AMSR2_surfex_domain["yy"]], axis = 1)
        Surfex_grid_points = np.stack([np.ndarray.flatten(Surfex_coord["xx"]), np.ndarray.flatten(self.Surfex_coord["yy"])], axis = 1)
        Distances_to_footprint_center = scipy.spatial.distance.cdist(AMSR2_grid_points, Surfex_grid_points, metric = "euclidean")
        return(Distances_to_footprint_center)
    #
    def get_graph_data(self, Distances_to_footprint_center, AMSR2_surfex_domain):
        Graphs = {}
        for i in range(0, len(AMSR2_surfex_domain["xx"])):
            if (np.isnan(AMSR2_surfex_domain["BT" + self.AMSR2_task_frequency + "H"][i]) == False) and (np.isnan(AMSR2_surfex_domain["BT" + self.AMSR2_task_frequency + "V"][i]) == False):
                Footprint_dist = np.argsort(Distances_to_footprint_center[i,:])[0:self.static_dimension]
                xx_coord = np.ndarray.flatten(self.Surfex_coord["xx"])[Footprint_dist]
                yy_coord = np.ndarray.flatten(self.Surfex_coord["yy"])[Footprint_dist]
                Graph_coord = np.concatenate((np.expand_dims(xx_coord, axis = 1), np.expand_dims(yy_coord, axis = 1)), axis = 1)
                #
                if len(Graphs) == 0:
                    Graphs["Distance_to_footprint_center"] = np.expand_dims(Distances_to_footprint_center[i,:][Footprint_dist], axis = 0)
                    Graphs["xx"] = np.expand_dims(xx_coord, axis = 0)
                    Graphs["yy"] = np.expand_dims(yy_coord, axis = 0)
                    Graphs["Distance_matrix"] = np.expand_dims(scipy.spatial.distance.cdist(Graph_coord, Graph_coord, metric = "euclidean"), axis = 0)
                    for var in self.Surfex_data:
                        Graphs[var] = np.expand_dims(np.ndarray.flatten(self.Surfex_data[var])[Footprint_dist], axis = 0)
                        if "ISBA" in var:
                            Graphs[var][Graphs[var] > 1e10] = np.nan
                    for var in self.MEPS_data:
                        Graphs[var] = np.expand_dims(np.ndarray.flatten(self.MEPS_data[var])[Footprint_dist], axis = 0)
                    for var in AMSR2_surfex_domain:
                        if var != "N_obs":
                            Graphs["AMSR2_" + var] = np.expand_dims(AMSR2_surfex_domain[var][i], axis = 0)
                else:
                    Graphs["Distance_to_footprint_center"] = np.concatenate((Graphs["Distance_to_footprint_center"], np.expand_dims(Distances_to_footprint_center[i,:][Footprint_dist], axis = 0)), axis = 0)
                    Graphs["xx"] = np.concatenate((Graphs["xx"], np.expand_dims(xx_coord, axis = 0)), axis = 0)
                    Graphs["yy"] = np.concatenate((Graphs["yy"], np.expand_dims(yy_coord, axis = 0)), axis = 0)
                    Graphs["Distance_matrix"] = np.concatenate((Graphs["Distance_matrix"], np.expand_dims(scipy.spatial.distance.cdist(Graph_coord, Graph_coord, metric = "euclidean"), axis = 0)), axis = 0)
                    for var in self.Surfex_data:
                        Graphs[var] = np.concatenate((Graphs[var], np.expand_dims(np.ndarray.flatten(self.Surfex_data[var])[Footprint_dist], axis = 0)), axis = 0)
                        if "ISBA" in var:
                            Graphs[var][Graphs[var] > 1e10] = np.nan
                    for var in self.MEPS_data:
                        Graphs[var] = np.concatenate((Graphs[var], np.expand_dims(np.ndarray.flatten(self.MEPS_data[var])[Footprint_dist], axis = 0)), axis = 0)
                    for var in AMSR2_surfex_domain:
                        if var != "N_obs":
                            Graphs["AMSR2_" + var] = np.concatenate((Graphs["AMSR2_" + var], np.expand_dims(AMSR2_surfex_domain[var][i], axis = 0)), axis = 0)
        return(Graphs)    
    #
    def write_graphs(self, Graphs, mbr):
        path_output = self.paths["output"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + date_task[6:8] + "/" + date_task[9:11] + "/" + "03" + "/" + "%s"%mbr + "/"
        print(path_output)
        if os.path.exists(path_output) == False:
            os.system("mkdir -p " + path_output) 
        filename = path_output + "Graphs_" + self.date_task + ".h5"
        #
        with h5py.File(filename, "a") as hdf:
            for var in Graphs:
                if var in hdf:
                    hdf_var = np.array(hdf[var][:])
                    var_conc = np.concatenate((hdf_var, Graphs[var]), axis=0)
                    del hdf[var]
                    hdf.create_dataset(var, data = var_conc) 
                else:
                    hdf.create_dataset(var, data = Graphs[var])
    #
    def __call__(self):
        Graphs = {}
        for filename in self.dataset:
            hour_filename = int(filename[-29:-27])
            if hour_filename < self.max_hour:
                AMSR2_dataset = read_AMSR2_data(filename, self.AMSR2_task_frequency)()
                AMSR2_surfex_domain = self.AMSR2_on_surfex_domain(AMSR2_dataset)
                if len(AMSR2_surfex_domain) > 0:
                    print(filename)
                    Distances_to_footprint_center =  self.calculate_distances(AMSR2_surfex_domain)
                    Graphs = self.get_graph_data(Distances_to_footprint_center, AMSR2_surfex_domain)
                    self.write_graphs(Graphs, mbr)


# # Data processing

# In[49]:


def main():

    # # Constants
    
    SGE_TASK_ID = 1
    #
    date_min = "20220704"
    date_max = "20220705"
    #
    AMSR2_task_frequency = "18.7"
    AMSR2_frequencies = ["6.9", "7.3", "10.7", "18.7", "23.8", "36.5"]
    AMSR2_footprint_radius = [0.25 * (35 + 62), 0.25 * (35 + 62), 0.25 * (24 + 42), 0.25 * (14 + 22), 0.25 * (11 + 19), 0.25 * (7 + 12)]  # 0.5 * mean diameter (0.5 * (major + minor))
    #
    paths = {}
    paths["AMSR2"] = "/lustre/storeB/immutable/archive/projects/remotesensing/satellite/"
    paths["MEPS"] = "/lustre/storeB/immutable/archive/projects/metproduction/meps/"
    paths["surfex"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/archive/"
    paths["surfex_grid"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/climate/"
    paths["output"] = "/lustre/storeB/users/josteinbl/sfx_data/LDAS_NOR_LETKF/archive/" #+ AMSR2_task_frequency.split('.')[0] + "GHz_static/"
    #
    hours_AMSR2 = "H03"
    MEPS_leadtime = int(hours_AMSR2[2])
    #
    crs = {}
    crs["latlon"] = pyproj.CRS.from_proj4("+proj=latlon")
    #
    MEPS_spatial_resolution = 2500
    MEPS_dim_variables = ["time", "pressure", "x", "y", "longitude", "latitude"]
    MEPS_LWE_thickness = ["lwe_thickness_of_atmosphere_mass_content_of_water_vapor"]
    MEPS_clouds = ["mass_fraction_of_cloud_condensed_water_in_air_pl", "mass_fraction_of_cloud_ice_in_air_pl", ]
    MEPS_precip = ["mass_fraction_of_snow_in_air_pl", "mass_fraction_of_rain_in_air_pl", "mass_fraction_of_graupel_in_air_pl"]
    MEPS_specific_humidity = ["specific_humidity_pl"]
    MEPS_PL_variables = MEPS_LWE_thickness + MEPS_clouds + MEPS_precip + MEPS_specific_humidity
    #
    surfex_PGD_variables = ["ZS", "COVER004", "COVER006"]
    surfex_prognostic_variables = ["FRAC_WATER", "FRAC_NATURE", "FRAC_SEA"]
    surfex_surface_and_integrated_variables = ["Q2M_ISBA", "T2M_ISBA", "TS_ISBA", "DSN_T_ISBA", "LAI_ga", "PSN_ISBA", "PSNG_ISBA", "PSNV_ISBA"]
    predictor_variables = surfex_prognostic_variables + surfex_surface_and_integrated_variables
    #
    n_soil_layers = 2
    n_snow_layers = 12
    surfex_soil_variables = ["TG", "WSN_VEG", "WG", "WGI", "RSN_VEG", "SNOWTEMP", "SNOWLIQ", "HSN_VEG"] 
    #
    for var in surfex_soil_variables:
        if ("SNOW" in var) or ("_VEG" in var):
            for layer in range(1, n_snow_layers + 1):
                predictor_variables.append(var + str(layer) + "_ga")
        else:
            for layer in range(1, n_soil_layers + 1):
                predictor_variables.append(var + str(layer) + "_ga")

    t0 = time.time()
    #
    list_dates = make_list_dates(date_min, date_max)
    date_task = list_dates[SGE_TASK_ID - 1]
    date_task_hours_AMSR2 = date_task + hours_AMSR2
    AMSR2_task_radius = AMSR2_footprint_radius[AMSR2_frequencies.index(AMSR2_task_frequency)]
    static_dimension = int(2 * 1000 * AMSR2_task_radius / (np.sqrt(MEPS_spatial_resolution ** 2 + MEPS_spatial_resolution ** 2))) ** 2
    print(date_task)
    print(str(AMSR2_task_radius) + " km")
    #
    Surfex_coord, Surfex_PGD = get_surfex_coordinates(crs, surfex_PGD_variables, paths)()

    mbrs = ["003"]#, "001"]#,"002","003","004","005","006","007","008", "009"]

    MEPS_data = get_MEPS_data(date_task, MEPS_leadtime, MEPS_dim_variables, MEPS_PL_variables, Surfex_coord, crs, paths)()        

    for mbr in mbrs:

        Surfex_data = read_surfex_data(date_task_hours_AMSR2, paths, predictor_variables, mbr)
        Surfex_data["WSN_T_ISBA"] = calculate_WSN_T_ISBA(Surfex_data, n_soil_layers)
        Surfex_data["FRAC_LAND_AND_SEA_WATER"] = Surfex_data["FRAC_WATER"] + Surfex_data["FRAC_SEA"] 
        Surfex_data["SNOW_GRADIENT"] = (Surfex_data["SNOWTEMP12_ga"] - Surfex_data["SNOWTEMP1_ga"]) / Surfex_data["DSN_T_ISBA"]
        Surfex_data["SNOW_GRADIENT"][Surfex_data["SNOW_GRADIENT"] > 50] = 50
        Surfex_data["SNOW_GRADIENT"][Surfex_data["SNOW_GRADIENT"] < -50] = -50
        for var in Surfex_PGD:
            Surfex_data[var] = Surfex_PGD[var]
        
        extract_graphs(date_task = date_task, 
                       paths = paths, 
                       AMSR2_task_frequency = AMSR2_task_frequency, 
                       static_dimension = static_dimension,
                       crs = crs,
                       Surfex_coord = Surfex_coord, 
                       Surfex_data = Surfex_data,
                       MEPS_data = MEPS_data)()
    #
    tf = time.time()
    print("Computing time: ", tf - t0)


if __name__ == "__main__":

    main()    


# In[ ]:




