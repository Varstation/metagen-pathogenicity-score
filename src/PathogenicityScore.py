import xarray as xr
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

from ete3 import NCBITaxa
ncbi = NCBITaxa()

class FeatureExtractor:
    def __init__(self, n_components, *args, **kwargs):
        self.n_components = n_components
        self.nmf = NMF(n_components, *args, **kwargs)
    
    def fit(self, X):
        for coord in X.coords:
            if coord != "sample":
                taxon = coord
        self.seen_taxa = X.coords[taxon].values

        self.nmf.fit(X.values)
        self.components_ = xr.DataArray(
            self.nmf.components_,
            {"feature":range(1,self.nmf.n_components+1), taxon:X.coords[taxon].values},
            ["feature", taxon])
        return self
    
    def transform(self, X):
        for coord in X.coords:
            if coord != "sample":
                taxon = coord

        W = xr.DataArray(
            self.nmf.transform(X.reindex({taxon:self.seen_taxa}, fill_value=0).values),
            {"sample":X.coords["sample"].values, "feature":range(1,self.nmf.n_components+1)},
            ["sample", "feature"])
        residuals = X - W @ self.components_
        
        # Organisms absent from the training data have to be considered separately.
        # Since they are not part of the learned components, all their abundance has
        # to be allocated to residuals.
        missing_cols = set(X.coords[taxon].values) - set(self.seen_taxa)
        residuals = xr.concat([residuals, X.sel({taxon:list(missing_cols)})], taxon)
        residuals = residuals.assign_coords(feature="residual")
        
        features = (
            xr.concat([W*self.components_, residuals], dim="feature")
            .fillna(0.0)
            .reindex({taxon:X.coords[taxon].values})
            )
        return features
    
    def fit_transform(self, X):
        for coord in X.coords:
            if coord != "sample":
                taxon = coord
        self.seen_taxa = X.coords[taxon].values

        self.nmf.fit(X.values)
        W = xr.DataArray(
            self.nmf.fit_transform(X.values),
            {"sample":X.coords["sample"].values, "feature":range(1,self.nmf.n_components+1)},
            ["sample", "feature"])
        self.components_ = xr.DataArray(
            self.nmf.components_,
            {"feature":range(1,self.nmf.n_components+1), taxon:X.coords[taxon].values},
            ["feature", taxon])
        residuals = X - (W @ self.components_).values
        residuals = residuals.assign_coords(feature="residual")
        
        features = xr.concat([W*self.components_, residuals], dim="feature")
        return features

class PathogenicityClassifier:
    def __init__(self, clf=make_pipeline(RobustScaler(), LogisticRegression()), taxonomic_info=None, **kwargs):
        self.clf = clf(**kwargs_starting_with("clf_", kwargs))
        self.taxonomic_info = taxonomic_info
        self.kwargs = kwargs
    
    def input_validation(self, data):
        if "species" not in self.taxonomy_levels:
            raise ValueError("PathogenicityClassifier needs species-level abundance data.")
        for rank in self.taxonomy_levels:
            if f"{rank}_n_components" not in self.kwargs:
                raise ValueError(f"Number of components for rank {rank} was not specified.")
            if f"{rank}_abundance" not in data.keys():
                raise ValueError(f"Rank {rank} was specified, but no '{rank}_abundance' variable was not found in data")

    def fit(self, data):
        self.taxonomy_levels = set(data.coords) - {"sample"}
        self.input_validation(data)
        # - Generate NMF features for each taxonomy level using FeatureExtractor
        # - Put all xarray.DataArray's in the dict NMF_features
        self.NMF_ = {}
        self.NMF_features_ = {}
        for rank in self.taxonomy_levels:
            NMF_kwargs = kwargs_starting_with(f"{rank}_", self.kwargs)
            self.NMF_[rank] = FeatureExtractor(**NMF_kwargs)
            feature_names = [f"{rank}_{i+1}" for i in range(NMF_kwargs["n_components"])] + [f"{rank}_residual"]
            self.NMF_features_[rank] = self.NMF_[rank].fit_transform(data[f"{rank}_abundance"]).assign_coords(feature=feature_names)
            if rank != "species":
                self.NMF_features_[rank] = self.NMF_features_[rank].loc[{rank:self.taxonomic_info[rank]}].reset_coords(drop=True)
        
        # Train supervised algorithm with extracted features
        observed_species = data["species_abundance"].stack(instance=("species","sample")).to_pandas()>0
        self.predictors_ = xr.concat(self.NMF_features_.values(), dim="feature").stack(instance=("species","sample")).to_pandas().T
        self.target_ = data["classification"].stack(instance=("species","sample")).to_pandas()
        if (self.predictors_.index != self.target_.index).any():
            raise ValueError("Predictor matrix and target matrix should have equal indices")
        self.predictors_, self.target_ = self.predictors_.loc[observed_species], self.target_.loc[observed_species]
        self.clf_ = self.clf.fit(self.predictors_, self.target_)

        return self
    
    def predict_proba(self, data):
        self.taxonomy_levels = set(data.coords) - {"sample"}
        self.input_validation(data)
        
        self.NMF_features_ = {}
        for rank in self.taxonomy_levels:
            feature_names = [f"{rank}_{i+1}" for i in range(self.NMF_[rank].n_components)] + [f"{rank}_residual"]
            self.NMF_features_[rank] = self.NMF_[rank].transform(data[f"{rank}_abundance"]).assign_coords(feature=feature_names)
            if rank != "species":
                self.NMF_features_[rank] = self.NMF_features_[rank].loc[{rank:self.taxonomic_info[rank]}].reset_coords(drop=True)
        
        # Evaluate supervised algorithm with extracted features
        observed_species = data["species_abundance"].stack(instance=("species","sample")).to_pandas()>0
        self.predictors_ = xr.concat(self.NMF_features_.values(), dim="feature").stack(instance=("species","sample")).to_pandas().T
        self.predictors_ = self.predictors_.loc[observed_species]
        prob = xr.DataArray(
            self.clf_.predict_proba(self.predictors_),
            {"class":self.clf_.classes_, "instance":self.predictors_.index},
            ["instance", "class"])
        return prob

    def predict(self, data):
        return self.predict_proba(data).argmax("class")

def get_rank(species, rank):
    for tx, rk in ncbi.get_rank(ncbi.get_lineage(species)).items():
        if rk==rank:
            return tx
    return 0

def make_xarray(abundance_tables, classification):
    if "species" not in abundance_tables.keys():
        raise ValueError("make_xarray needs a species-level abundance table (tables for other taxonomic levels are optional)")
    taxonomic_info = {rank:xr.DataArray(abundance_tables["species"].columns.map(lambda x: get_rank(x, rank)), {"species":abundance_tables["species"].columns}) for rank, table in abundance_tables.items() if rank!="species"}
    abundance_tables = {f"{rank}_abundance":xr.DataArray(table, {"sample":table.index, rank:table.columns}) for rank, table in abundance_tables.items()}
    return taxonomic_info, xr.Dataset(abundance_tables | dict(classification = xr.DataArray(classification, {"sample":classification.index, "species":classification.columns})))

def kwargs_starting_with(prefix, kwargs):
    # searches keyword arguments containing some prefix ending in '_', removes
    # the underscore and returns a dictionary with the results
    return {"_".join(key.split("_")[1:]):value for key, value in kwargs.items() if key.startswith(prefix)}

