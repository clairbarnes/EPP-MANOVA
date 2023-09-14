# load necessary modules
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import cartopy.crs as ccrs

matplotlib.rcParams['savefig.bbox'] = "tight"    #CB: always save with tight bounding box - means that text is no longer cut off when using tight_layout


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
        if new_labels is None: new_labels = {"n" : [list(range(M.shape[0]))]}
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
    # REC: amended next line to retain maps corresponding to all EPPs (previously just retained two)
    EPP_maps = xr.concat([Ybar.expand_dims(epp = ["Ensemble mean"]), EPP_maps], "epp")
    
    # Compute % of variance explained, convert to formatted string for title
    var_explained = Lambda**2 / sum(Lambda**2) * 100
    
    return {"EPPs" : EPP_maps, "scores" : scores, "var_explained" : var_explained}
       
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def EPP_plot(epps, scores, var_exp, which=[1,2], cmap = "RdBu", markers = None, colours = None, 
             cbar_label = None, vlim = None, LegInfo=None):
    
    # Custom plotting function. NB "which" controls the EPPs to plot
    
    # create array of subplots with appropriate projection - with extra space if a legend is needed
    
    if LegInfo is None:
        FigSize = (13, 4.5)
        RightMargin = 1
        PlotWidths = [1, 1, 1, 0.1, 0.8]  #CB: added an extra column to separate the scatterplot
        TitleHeight = 1.345 #CB: my *ahem* elegant solution to getting the titles to line up
    else:
        FigSize = (15, 4.5)
        RightMargin = 0.9
        PlotWidths = [1, 1, 1, 0.1, 0.8, 0.9]  #CB: added an extra column to separate the scatterplot
        TitleHeight = 1.4 #CB: my *ahem* elegant solution to getting the titles to line up

    fig, axs = plt.subplots(ncols = len(PlotWidths), figsize = FigSize, gridspec_kw = {'width_ratios' : PlotWidths},
                                sharex = True, sharey = True, dpi= 200, facecolor='w', 
                                edgecolor='k', subplot_kw = {"projection" : crs_osgb})
    fig.subplots_adjust(bottom=0.05, top = 0.8, right=RightMargin, wspace = 0)
    fig.tight_layout()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MAPS OF ENSEMBLE MEAN & FIRST TWO EPPS
    
    # if not provided, get range of values for colourbar
    ToPlot = [0] + which
    if vlim is None: vlim = np.ceil(max([np.abs(x) for x in [epps[ToPlot].min(), epps[ToPlot].max()]]))
    
    # convert variance explained to string
    ve_string = [""] + [" ("+str(round(x,1))+"%)" for x in var_exp]
    
    # Plot IDs
    PlotIDs = ["(a) ", "(b) ", "(c) "]

    # plot maps
    for i in range(3):
        CurMap = ToPlot[i]
        cbar = epps.isel(epp = CurMap).plot(ax = fig.axes[i], cmap = cmap,       # draw map
                                       vmin = -vlim, vmax = vlim, add_colorbar = False) 
        fig.axes[i].set_title(PlotIDs[i] + epps.epp.values[CurMap] + ve_string[CurMap])  # add title
        # set spatial extent of map axes, remove bounding box, add coastlines
        fig.axes[i].set_extent((-2e5, 7e5, -1e5, 12.2e5), crs = crs_osgb)             # fix plot extent to reduce whitespace
        # fig.axes[i].set_axis_off()                                                    # remove box around plot #CB: removed this because we can do it for all axes simultaneously
        fig.axes[i].coastlines()                                                      # draw coastlines
    
    for ax in axs: ax.set_axis_off()  #CB: turned off all axes in one fell swoop

    plt.colorbar(cbar, ax = fig.axes[:3], location = "bottom", pad = 0.04, fraction = 0.05, label = cbar_label)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SCATTERPLOT OF SCORES
    
    # fix plotting limits to be symmetric about 0
    score_range = [max([np.abs(x) * 1.2 for x in [scores[:,[i-1 for i in which]].min(), 
                                                  scores[:,[i-1 for i in which]].max()]]) * c for c in [-1,1]]

    # add subplot (this is necessary because the other subplots are drawn on GeoAxes, with a specific projection)
    ax4 = fig.add_subplot(1, len(PlotWidths), 5, xlim = score_range, ylim = score_range, aspect="equal", #CB: changed axis to account for dummy subplot
                          adjustable="box", anchor="C", xlabel = "EPP" + str(which[0]) + " score", 
                          ylabel = "EPP" + str(which[1]) + " score") # title fontsize should be 12? 
    ax4.set_title("(d) Contribution from each pattern", y = TitleHeight) #CB: set title separately to allow control of vertical alignment

    # add points individually so that unique markers/colours can be used
    if markers is None: 
        markers = np.repeat("o", len(scores[:,0]))
    elif len(markers) == 1:
        markers = np.repeat(markers, len(scores[:,0]))
        
    if colours is None: 
        colours = np.repeat("k", len(scores[:,0]))
    elif len(colours) == 1:
        colours = np.repeat(colours, len(scores[:,0]))
        
    [ax4.scatter(scores[i,which[0]-1], scores[i,which[1]-1], marker = markers[i], color = colours[i], 
                 edgecolor = "k", s = 70, zorder = 9) for i in range(len(scores[:,0]))]
       
    # add gridlines & labels
    ax4.axvline(0, linestyle = "--", color = "grey")
    ax4.axhline(0, linestyle = "--", color = "grey")
    
    if LegInfo is not None: # Legend in remaining plot panel
        ax5 = fig.add_subplot(166, xlim = [0,1], ylim = [0,1]) #CB: was 155
        FirstItem = list(LegInfo.keys())[0]
        if "_r1" in FirstItem: # GCMs #CB: made this a more general case
            handles = [matplotlib.lines.Line2D([], [], color = "w", marker = m, markersize = 6, 
                                               markeredgecolor = "black", linestyle = "None")
                       for gcm_nm, m in LegInfo.items()]
            LegTitle = "GCM"
        else: # RCMs #CB: since we don't have a catch-all 'else', I've made this it. If this is to be used for anything other than EuroCordex we should probably make an actual general case.
            handles = [matplotlib.lines.Line2D([], [], color = c, marker = 'o', markersize = 6, 
                                               markeredgecolor = "black", linestyle = "None") 
                       for rcm_nm, c in LegInfo.items()]
            LegTitle = "RCM"
    
        ax5.legend(handles = handles, labels=LegInfo.keys(), loc="center", frameon=False,
                   bbox_to_anchor = (0.5, 0.5), edgecolor = "white", title = LegTitle)
        ax5.set_axis_off() 
