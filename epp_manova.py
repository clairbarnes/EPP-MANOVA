# load necessary modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import re

# define projection to be used in plotting
crs_osgb = ccrs.TransverseMercator(approx = False, central_longitude = -2, central_latitude = 49, scale_factor = 0.9996012717,
                                   false_easting = 400000, false_northing = -100000, globe = ccrs.Globe(datum = 'OSGB36', ellipse = 'airy'))

# Adding coastlines to the maps elicits a ShapelyDeprecationWarning due to default behaviour of Cartopy -- ignore this
import warnings
warnings.filterwarnings("ignore", message = ".+multi-part geometry.+")

#####################################################################
# SUPPORT FUNCTIONS

def reshape_to_map(M, to_map, new_labels = None):
    
    # Reshape matrix of values M into map of non-missing values
    
    # get coordinates of non-empty cells in target map
    px = np.argwhere(~np.isnan(to_map.values))
    
    # create map array to hold the reshaped matrix
    map_array = to_map.where(np.isnan(to_map), 0)
    
    if len(M.shape) == 1:     # M is a vector
    
        for i in list(range(len(px))): map_array[px[i,0], px[i,1]] = M[i]
        
    else:                     # M is a matrix
    
        # expand map array to accommodate the extra dimension; if no labels are provided, number them
        if new_labels is None: new_labels = {"n" : []}
        map_array = map_array.expand_dims(new_labels).copy()
        
        # assign values from matrix to non-empty cells in target map
        for i in list(range(len(px))): map_array[:, px[i,0], px[i,1]] = M[:,i]
    
    return map_array


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def EPPs(Y_tilde, Ybar):
    
    # Compute EPPs for an M x S matrix of centred data (here M = number of groups, S = spatial dimension)
    
    # Singular Value Decomposition of the centred data
    U, Lambda, V = np.linalg.svd(Y_tilde, full_matrices = False)
    
    # extract principal patterns and scores
    EPP_vectors = np.diag(Lambda) @ V
    scores = U
    
    # adjust patterns & scores to ensure that positive scores are associated with positive average value over the UK
    sign_adjustment = np.sign(EPP_vectors.mean(axis = 1)) 
    EPP_vectors = np.diag(sign_adjustment) @ EPP_vectors
    scores = sign_adjustment * scores
    
    # adjust patterns & scores to ensure that positive scores are associated with positive correlations with mean
    # sign_adjustment = [np.sign(np.corrcoef(EPP_vector[i,:], Ybar.stack(s = ("projection_y_coordinate", "projection_x_coordinate")).dropna("s", "any").values)[0,1]) for i in range(len(Lambda))]
    
    EPP_maps = reshape_to_map(EPP_vectors, to_map = Ybar, new_labels = {"epp" : ["EPP"+str(x+1) for x in range(len(scores))]})
    EPP_maps = xr.concat([Ybar.expand_dims(epp = ["Ensemble mean"]), EPP_maps.sel(epp = ["EPP1", "EPP2"])], "epp")
    
    # Compute % of variance explained, convert to formatted string for title
    var_explained = Lambda**2 / sum(Lambda**2) * 100
    
    return {"EPPs" : EPP_maps, "scores" : scores, "var_explained" : var_explained}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def EPP_plot(epps, scores, var_exp, cmap = "RdBu", markers = None, colours = None, cbar_label = None, vlim = None):
    
    # Custom plotting function
    
    # create array of subplots with appropriate projection
    fig, axs = plt.subplots(ncols = 4, figsize = (13,4), gridspec_kw = {'width_ratios' : [1,1,1,2]},
                            sharex = True, sharey = True, dpi= 100, facecolor='w', edgecolor='k', subplot_kw = {"projection" : crs_osgb})
    fig.subplots_adjust(top = 0.85, wspace = 0)
    fig.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAPS OF ENSEMBLE MEAN & FIRST TWO EPPS
    
    # if not provided, get range of values for colourbar
    if vlim is None: vlim = np.ceil(max([np.abs(x) for x in [epps.min(), epps.max()]]))
    
    # convert variance explained to string
    ve_string = [""] + [" ("+str(int(x))+"%)" for x in var_exp[:2]]
    
    # plot maps
    for i in range(3):
        cbar = epps.isel(epp = i).plot(ax = fig.axes[i], cmap = cmap, vmin = -vlim, vmax = vlim, add_colorbar = False)      # draw map
        fig.axes[i].set_title(epps.epp.values[i] + ve_string[i])                                                            # add title
    
        # set spatial extent of map axes, remove bounding box, add coastlines
        fig.axes[i].set_extent((-2e5, 7e5, -1e5, 12.2e5), crs = crs_osgb)                                                  # fix plot extent to reduce whitespace
        fig.axes[i].set_axis_off()                                                                                         # remove box around plot
        fig.axes[i].coastlines()                                                                                           # draw coastlines
    
    plt.colorbar(cbar, ax = fig.axes[:3], location = "bottom", pad = 0.04, fraction = 0.05, label = cbar_label)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SCATTERPLOT OF SCORES
    
    # add subplot (this is necessary because the other subplots are drawn on GeoAxes, with a specific projection)
    ax3 = fig.add_subplot(144)
    
    # add points individually so that unique markers/colours can be used
    if markers is None: 
        markers = np.repeat("o", len(scores[:,0]))
    elif len(markers) == 1:
        markers = np.repeat(markers, len(scores[:,0]))
        
    if colours is None: 
        colours = np.repeat("k", len(scores[:,0]))
    elif len(colours) == 1:
        colours = np.repeat(colours, len(scores[:,0]))
        
    [ax3.scatter(scores[i,0], scores[i,1], marker = markers[i], color = colours[i], edgecolor = "k", s = 70, zorder = 9) for i in range(len(scores[:,0]))]
    
    # fix plotting limits to be symmetric about 0
    score_range = [max([np.abs(x) * 1.2 for x in [scores[:,:2].min(), scores[:,:2].max()]]) * c for c in [-1,1]]
    ax3.set_xlim(*score_range)
    ax3.set_ylim(*score_range)
    
    # add gridlines & labels
    ax3.set_xlabel("First EPP score")
    ax3.set_ylabel("Second EPP score")
    ax3.set_title("Contribution from each pattern", fontsize = 12)
    ax3.axvline(0, linestyle = "--", color = "grey")
    ax3.axhline(0, linestyle = "--", color = "grey")
    ax3.set_aspect("equal", adjustable = "box")