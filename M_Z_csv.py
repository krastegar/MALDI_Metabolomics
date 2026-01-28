import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse as sp
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from SpaCoObject import SPACO
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
import plotly.express as px
from sklearn.cluster import KMeans
import umap
import igraph as ig
import leidenalg as la

class SpectrumData:
    def __init__(self, imzml_path, min_intensity=100, min_count=100, mz_tol=0.0042):
        self.parser = ImzMLParser(imzml_path)
        self.df =pd.DataFrame(
                    (   # neat trick to unpack the mzs and intensities directly into the row
                        # * is a unpacking operator called splat and it unpacks the tuple of mzs and intensities
                        # from getspectrum into individual elements in the new tuple
                        (*self.parser.getspectrum(idx), coord) for idx, coord in enumerate(self.parser.coordinates)
                    ),
                        columns=["mzs", "intensities", "coordinates"]
            )
        self.min_count = min_count
        self.mz_tol = mz_tol
        self.min_intensity = min_intensity
        self.long_df = None
    
    def min_max_mz(self, mz_min, mz_max) -> pd.DataFrame:
        """
        Filter the spectrum data based on a given m/z range. 

        :param mz_min: the minimum m/z value to keep
        :param mz_max: the maximum m/z value to keep
        :return: a filtered DataFrame containing only the spectrum data where any m/z is within [mz_min, mz_max]
        :rtype: pandas.DataFrame
        """

        # keep rows where any m/z is within [mz_min, mz_max]
        mask = self.df["mzs"].apply(lambda mzs: np.any((mzs >= mz_min) & (mzs <= mz_max)))
        filtered = self.df[mask].copy()

        # trim each row's mzs and intensities to the requested range
        def trim_row(row):
            mzs = row["mzs"]
            ints = row["intensities"]
            m = (mzs >= mz_min) & (mzs <= mz_max)
            row["mzs"] = mzs[m]
            row["intensities"] = ints[m]
            return row

        return filtered.apply(trim_row, axis=1)
    
    def plot_ion_image(self, mz, cmap='viridis', tol=0.1, z=1) -> None:
        """
        Plot the ion image for a given m/z value.

        :param mz: the m/z value to plot
        :param tol: the tolerance for the m/z value
        :param z: the z-coordinate to plot (default is 1)
        """
        ion_mat = getionimage(self.parser, mz, tol=tol, z=z)
        plt.imshow(ion_mat.T, cmap = cmap, interpolation='auto')
        plt.colorbar()
        plt.title(f"Ion image m/z {mz} ± {tol}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(f'ion_image_mz{mz:.4f}_tol{tol}.png')
        plt.show()
    
    def plot_sampling_comparison(self, mz, tol=0.1, z=1, tile_sizes=(3000, 10000, 5000), cmap='viridis', save_path=None, figsize_per_panel=(7,7)) -> None:
        """
        Show the original ion image next to downsampled (tiled) versions.
        tile_sizes: iterable of tile pixel-count targets (int). Each produces one sampled panel.
        save_path: optional Path or str to save the figure.
        """
        ion_mat = getionimage(self.parser, mz, tol=tol, z=z)
        n, m = ion_mat.shape

        # prepare images: first is original, then sampled for each tile_size
        panels = [('original', ion_mat)]
        for ts in tile_sizes:
            if ts <= 0:
                continue
            # Practically this takes every (n//new_n)-th row and every (m//new_m)-th column, 
            # producing a coarser image by striding through the original array.
            # Accomplished by using integer division (//) to determine the step size 
            # for rows and columns. Then taking the max of this step size and 1, 
            # to ensure that the step size is at least 1.
            k = np.sqrt(ts / (n * m))
            new_n = max(1, int(n * k))
            new_m = max(1, int(m * k))
            step_row = max(1, n // new_n)
            step_col = max(1, m // new_m)
            sampled = ion_mat[::step_row, ::step_col]
            panels.append((f'scaling factor ~= {k:.2f}', sampled))

        cols = len(panels)
        fig_w = figsize_per_panel[0] * cols
        fig_h = figsize_per_panel[1]
        fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h))
        if cols == 1:
            axes = [axes]

        im = None
        for ax, (title, img) in zip(axes, panels):
            im = ax.imshow(img, cmap=cmap, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')

        fig.suptitle(f"Ion image m/z {mz} ± {tol}")

        # remove extra white space
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()
        return
    
    def dbscan_1d_ish(self) -> pd.DataFrame:
        """
        Perform DBSCAN-like clustering on the m/z values in the dataframe. Clusters for now
        will be defined as contiguous regions where the difference between adjacent m/z values is less than mz_tol.
        Clusters with fewer than min_count points will be flagged as "noise".

        :param mz_tol: the tolerance for the m/z values
        :param min_intensity: the minimum intensity for a m/z to be considered
        :param min_count: the minimum number of points to form a cluster

        :return: a DataFrame with additional columns for delta_mz, in_dense_region, and cluster_labels
        :rtype: pandas.DataFrame
        """

        
        if self.long_df is None:

            # create long format dataframe if not already done
            print("Exploding dataframe to long format...")
            self.long_df = self.df.explode(['mzs', 'intensities']).reset_index(drop=True)

            # filter the long dataframe based on intensity
            self.long_df = self.long_df[self.long_df['intensities'] > self.min_intensity]
            
            # sorting the long dataframe by m/z values
            print(f"Sorting m/z values...{len(self.long_df)} values")
            self.long_df = self.long_df.sort_values(by=['mzs']).reset_index(drop=True)
            
            
            print(f'\nlong_df columns: {self.long_df.columns}\n')  # print the columns of self.long_df.columns)
        
        # check if long
        # find the deltas between adjacent m/z values
        print("calculating deltas")
        delta_values = -1 * self.long_df['mzs'].values[0:(len(self.long_df) - self.min_count)] + self.long_df['mzs'].values[self.min_count:(len(self.long_df) + self.min_count)]
        
        # 0 padding missing values to match the length of the dataframe
        self.long_df['delta_mz'] = np.concatenate([delta_values, [0] * self.min_count])
        del delta_values

        # creating a boolean mask for regions where delta_mz < mz_tol
        self.long_df['in_dense_region'] = (self.long_df['delta_mz'] < self.mz_tol).astype(int)

        # Calculating difference of boolean flags
        # Calculate diff without fillna to get proper length n-1
        difference = np.diff(self.long_df['in_dense_region'].values)
        difference_plus = np.maximum(difference, 0)
        running_clusternr = np.cumsum(difference_plus)
        del difference
        del difference_plus

        # intermediate vector of of boolean flags
        delta = self.long_df['in_dense_region'].values

        # Now both have compatible shapes: delta[:-1] and running_clusternr are both length n-1
        clusternr = delta[:-1] * running_clusternr
        del running_clusternr

        # 0 padding missing values to match the length of the dataframe
        self.long_df['cluster_labels'] = np.concatenate([[delta[0]], clusternr])
        del clusternr
        del delta

        # Remove noise clusters
        # going to prune the data
        self.long_df = self.long_df[self.long_df["cluster_labels"] != 0]

        # Remove clusters with fewer than N mz values
        self.long_df = self.long_df[self.long_df.groupby('cluster_labels') ['mzs'].transform('size') >= self.min_count]

        return self.long_df
    
    def _sample_feature_genration(self, agg_func="sum") -> pd.DataFrame:
        """
        Generate a sampled feature DataFrame from the clustered long DataFrame.
        :return: a DataFrame with sampled features
        :rtype: pandas.DataFrame
        """

        # lazy evaluation of dbscan_1d_ish
        if self.long_df is None:
            self.long_df=self.dbscan_1d_ish()
        
        # create the pivot table, our sampled features matrix 
        clustered_sample_feature_df = self.long_df.pivot_table(
            index='coordinates',
            columns='cluster_labels',
            values='intensities',
            aggfunc= agg_func # You might want to change the aggregation function depending on your needs
        ).fillna(0) # Fill NaN values with 0

        return clustered_sample_feature_df
    
    def _cluster_ion_image(self, cluster_df, cluster_number): 
        """
        Generate an ion image for a specific cluster.
        :param cluster_df: the DataFrame containing the sampled features for a specific cluster
        :param cluster_number: the cluster number to generate the ion image for
        
        :return: the ion image for the specified cluster
        :rtype: numpy.ndarray
        """

        # ensure cluster index coordinates are a tuple of coordinates 
        cluster_df.index = [t[:2] for t in cluster_df.index]

        # check if the cluster number is valid
        #if cluster_number not in cluster_df.columns:
            # for now we just raise an error if the cluster is not found
            #raise ValueError(f"Cluster {cluster_number} not found in the DataFrame.\nAvailable clusters: {cluster_df.columns}")
        
        # filter the dataframe based on specific cluster number
        specific_cluster_df = cluster_df[cluster_number]
        
        # changes index into columns, so it means that 
        specific_cluster_df = specific_cluster_df.reset_index() 

        # creates two new columns x and y. Still keeps index column
        # two new columns are the 1st and 2nd element of coords ex) index: (1,3) -> x:1 , y:3
        coords = np.array(specific_cluster_df["index"].tolist())
        specific_cluster_df["x"] = [x for x, _ in coords]
        specific_cluster_df["y"] = [y for _, y in coords]
        specific_cluster_df.drop('index', axis=1, inplace=True)
        
        # pivot the dataframe to create the ion image
        ion_image = specific_cluster_df.pivot(index='x', columns='y', values=cluster_number)

        return ion_image
    
    def tile_and_aggregate(self, df, grid_size, aggfunc='median'):
        """
        Group a MultiIndex DataFrame with (x,y) coordinates into tiles and aggregate.
        
        Parameters
        ----------
        df : pd.DataFrame
            Must have a MultiIndex with levels ["x","y"].
        tile_size : tuple (tx, ty)
            Tile size along x and y.
        aggfunc : callable or dict
            Aggregation function (np.mean, np.sum, etc.) or dict of column->func.
        
        Returns
        -------
        pd.DataFrame
            Aggregated DataFrame with one row per tile.
        """
        # reducing the tuples in the index
        df.index = [t[:2] for t in df.index]

        # calculate tile size
        xs = np.asarray([x for x, _ in df.index.to_list()])
        ys = np.asarray([y for _, y in df.index.to_list()])
        n = max(xs) + 1
        m = max(ys) + 1
        k = np.sqrt(grid_size/(n * m))

        # making sure that our tile sizes are valid
        # this is done by checking the shrinkage parameter k
        if k >= 1: 
            raise ValueError(f'shrinkage was miscalculated: {k}')
        
        # tile size 
        u = int(np.ceil(1/k))
        
        # assign each coordinate to a tile
        tile_x = xs // u
        tile_y = ys // u
        tile_id = pd.Series(list(zip(tile_x, tile_y)), index=df.index)
        
        # group by tile and aggregate
        grouped = df.groupby(tile_id).agg(aggfunc)
        
        # clean up index
        grouped.index = pd.MultiIndex.from_tuples(grouped.index, names=["tile_x","tile_y"])
        
        return grouped
    def plot_aggregated_ion_images(self, clustered_df_aggregated, cluster_col, cmap='Spectral', marker='s',s=20):
        """
        Code to specifically plot the ion images for a specific cluster. after aggregation

        :param cluster_df: the DataFrame containing the sampled features for a specific cluster
        :param cluster_number: the cluster number to generate the ion image for
        """

        x_plot = clustered_df_aggregated.index.get_level_values(0)
        y_plot = clustered_df_aggregated.index.get_level_values(1)

        plt.figure(figsize=(8, 7))
        #plt.hexbin(x, y, C=df[cluster_col], gridsize=120,cmap='viridis')
        plt.scatter(x_plot, y_plot, c=clustered_df_aggregated[cluster_col], s=s, marker=marker, cmap=cmap)
        plt.colorbar(label=f'Cluster: {cluster_col} intensity')
        plt.title(f'tiled grid colored by {cluster_col} # of spots: {clustered_df_aggregated.shape[0]}')
        plt.axis('equal')
        plt.show()

    def make_adjacency(self, points_list, neighbors=30): 
        """
        Compute the adjacency matrix from the coordinates of the spots and the neighboring spots.

        Parameters
        ----------
        coords : pandas.MultiIndex
            The coordinates of the spots.
        neigbors : pandas.MultiIndex
            The neighboring spots.

        Returns
        -------
        The adjacency matrix as a sparse matrix of shape (N, N) where N is the number of spots.
        """
        knn = NearestNeighbors(n_neighbors=neighbors, metric='euclidean', n_jobs=-1)
        knn.fit(points_list)

        # adjacency graph: sparse matrix of shape (N, N)
        adjacency_matrix = knn.kneighbors_graph(points_list, n_neighbors=neighbors, mode='distance')

        return adjacency_matrix

# ...existing code...
def compute_sparsity(data, by='mzs', intensity_thresh=None, plot=True, save_path=None, bins=50, scatter_sample=None):
    """
    Compute occupancy and sparsity per 'mzs' (default) or 'cluster_labels'.

    Parameters
    ----------
    data : SpectrumData or pd.DataFrame
        If SpectrumData instance, its .df and .min_intensity are used. If DataFrame, it must contain
        'mzs','intensities','coordinates' (wide or long).
    by : {'mzs','cluster_labels'}
    intensity_thresh : float or None
        If None uses data.min_intensity when SpectrumData is passed, otherwise 0.0.
    plot : bool
    save_path : str or Path or None
    bins : int
    scatter_sample : int or None
    Returns
    -------
    pd.DataFrame with columns ['count','occupancy','sparsity']
    """
    # detect SpectrumData vs DataFrame
    sd = None
    if hasattr(data, 'min_intensity') and hasattr(data, 'df'):
        sd = data

    # determine intensity threshold
    if intensity_thresh is None:
        intensity_thresh = float(sd.min_intensity) if sd is not None else 0.0

    # pick input dataframe
    if sd is not None:
        if by == 'cluster_labels':
            # ensure clustering was run
            if sd.long_df is None:
                sd.dbscan_1d_ish()
            df_input = sd.long_df
        else:
            df_input = sd.df
    elif isinstance(data, pd.DataFrame):
        df_input = data
    else:
        raise ValueError("data must be a SpectrumData instance or a pandas DataFrame")

    # ensure required columns
    if not {'mzs', 'intensities', 'coordinates'}.issubset(df_input.columns):
        raise ValueError("DataFrame must contain 'mzs','intensities','coordinates' columns")

    # convert wide->long if needed: check first row type for 'mzs'
    first_mzs = df_input['mzs'].iat[0]
    is_iterable_mzs = hasattr(first_mzs, '__iter__') and not isinstance(first_mzs, (str, bytes))
    if is_iterable_mzs:
        long = df_input.explode(['mzs', 'intensities']).reset_index(drop=True)
    else:
        long = df_input.reset_index(drop=True)

    # ensure numeric intensities
    long['intensities'] = pd.to_numeric(long['intensities'], errors='coerce')

    # build spot id (x,y)
    def _spot_id(coord):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            return (coord[0], coord[1])
        return coord
    long['spot'] = long['coordinates'].apply(_spot_id)

    # total number of spots
    if sd is not None:
        total_spots = len(sd.df)
    else:
        total_spots = long['spot'].nunique()

    if by == 'cluster_labels' and 'cluster_labels' not in long.columns:
        raise ValueError("cluster_labels not found in DataFrame; run clustering first or use by='mzs'")

    # presence mask and counts
    present = long['intensities'] > float(intensity_thresh)
    counts = long[present].groupby(by)['spot'].nunique()

    # ensure float index for mzs where possible
    if by == 'mzs':
        try:
            counts.index = counts.index.astype(float)
        except Exception:
            pass

    occupancy = counts / float(total_spots)
    sparsity = 1.0 - occupancy

    stats_df = pd.DataFrame({
        'count': counts,
        'occupancy': occupancy,
        'sparsity': sparsity
    }).sort_index()

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(stats_df['sparsity'].dropna(), bins=bins, color='C0', edgecolor='k', alpha=0.8)
        axes[0].set_xlabel('Sparsity (1 - occupancy)')
        axes[0].set_ylabel('Number of features')
        axes[0].set_title(f'Sparsity distribution ({by}), n_features={len(df_input)}')

        if by == 'mzs':
            x_label = 'm/z'
            x_vals = stats_df.index.values
        else:
            x_label = 'cluster_label'
            x_vals = stats_df.index.values

        if scatter_sample is not None and scatter_sample < len(stats_df):
            sampled = stats_df.sample(scatter_sample, random_state=0)
            x_plot = sampled.index.values
            y_plot = sampled['sparsity'].values
        else:
            x_plot = x_vals
            y_plot = stats_df['sparsity'].values

        axes[1].scatter(x_plot, y_plot, s=8, alpha=0.6)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('Sparsity')
        axes[1].set_title(f'Feature vs sparsity ({by})')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()

    return stats_df 


if __name__ == "__main__":
    # file path to the imzml file
    imzml_path = Path("/home/krastegar0/MALDI_Metabolomics/MSI_data_grant/Mass_Spec_data/20251012_old_liver_area.imzML")

    # initialize the SpectrumData class
    spectrum_data = SpectrumData(imzml_path, mz_tol=0.0042, min_count=25, min_intensity=0)

    # perform DBSCAN-like clustering
    clustered_df = spectrum_data.dbscan_1d_ish()
    #del clustered_df

    # generate sampled feature DataFrame
    SF_df = spectrum_data._sample_feature_genration(agg_func='sum')# usually use the median  

    # aggregate data into reduced resolution
    SF_reduced = spectrum_data.tile_and_aggregate(SF_df, grid_size=40000)

    # Get ion image for a specific cluster
    cluster_number = 1
    s = 17 # point plotting parameter
    #_ = spectrum_data.plot_aggregated_ion_images(SF_reduced, cluster_number, s=s)

    # attempting to run spaco on the reduced cluster data 
    adjacency_matrix = spectrum_data.make_adjacency(SF_reduced.index.to_list())
    coords = pd.DataFrame({'x': [x for x,_ in SF_reduced.index.to_list()],
                       'y': [y for _, y in SF_reduced.index.to_list()]})
    
    # run spaco
    # randomly pick 100 peaks / features
    random.seed(42)
    spaco = SPACO(sample_features = SF_reduced, neighbormatrix=adjacency_matrix, coords=coords)
    denoised_data, spaco_projections = spaco.spaco_projection()

    # plot the spacs
    spaco.plot_spatial_heatmap(spaco_projections[:,0], point_size=50, cmap='Spectral' ,title="spac_1")
    
    # inspect the features in a specific cluster
    spaco.feature_inspection_by_cluster(n_neighbors=5)
