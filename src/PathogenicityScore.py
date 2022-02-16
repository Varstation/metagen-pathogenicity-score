import xarray as xr
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
import pandas as pd

from ete3 import NCBITaxa
ncbi = NCBITaxa()

class FeatureExtractor:
    def __init__(self, n_components, *args, **kwargs):
        self.n_components = n_components
        self.nmf = NMF(n_components, *args, **kwargs)
    
    def fit(self, X):
        for coord in X.coords:
            if coord != "Sample":
                taxon = coord
        self.seen_taxa = X.coords[taxon].values

        self.nmf.fit(X.values)
        self.components_ = xr.DataArray(
            self.nmf.components_,
            {"Feature":range(1,self.nmf.n_components+1), taxon:X.coords[taxon].values},
            ["Feature", taxon])
        return self
    
    def transform(self, X):
        for coord in X.coords:
            if coord != "Sample":
                taxon = coord

        W = xr.DataArray(
            self.nmf.transform(X.reindex({taxon:self.seen_taxa}, fill_value=0).values),
            {"Sample":X.coords["Sample"].values, "Feature":range(1,self.nmf.n_components+1)},
            ["Sample", "Feature"])
        residuals = X - W @ self.components_
        
        # Organisms absent from the training data have to be considered separately.
        # Since they are not part of the learned components, all their abundance has
        # to be allocated to residuals.
        missing_cols = set(X.coords[taxon].values) - set(self.seen_taxa)
        residuals = xr.concat([residuals, X.sel(Species=list(missing_cols))], taxon)
        residuals = residuals.assign_coords(Feature="residual")
        
        features = (
            xr.concat([W*self.components_, residuals], dim="Feature")
            .reindex({taxon:X.coords[taxon].values})
            .stack({"Instance":["Sample",taxon]})
            .sel(Instance=X.stack({"Instance":["Sample",taxon]})>0)
            )
        return features
    
    def fit_transform(self, X):
        for coord in X.coords:
            if coord != "Sample":
                taxon = coord
        self.seen_taxa = X.coords[taxon].values

        self.nmf.fit(X.values)
        W = xr.DataArray(
            self.nmf.fit_transform(X.values),
            {"Sample":X.coords["Sample"].values, "Feature":range(1,self.nmf.n_components+1)},
            ["Sample", "Feature"])
        self.components_ = xr.DataArray(
            self.nmf.components_,
            {"Feature":range(1,self.nmf.n_components+1), taxon:X.coords[taxon].values},
            ["Feature", taxon])
        residuals = X - (W @ self.components_).values
        residuals = residuals.assign_coords(Feature="residual")
        
        features = xr.concat([W*self.components_, residuals], dim="Feature").stack({"Instance":["Sample",taxon]})
        return features.sel(Instance=X.stack({"Instance":["Sample",taxon]})>0)

