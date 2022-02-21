import unittest
from src.PathogenicityScore import FeatureExtractor
import xarray as xr
import numpy as np

if __name__ == "__name__":
    unittest.main()

class TestFeatureExtractor(unittest.TestCase):
    def test_missing_species(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"Sample":range(5),"Species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"Sample":range(10,20),"Species":list("abcde")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_new_species(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"Sample":range(5),"Species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"Sample":range(10,20),"Species":list("ghijk")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_new_species_only(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"Sample":range(5),"Species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"Sample":range(10,20),"Species":list("klmno")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_fit_transform_result_adds_up(self):
        sim_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"Sample":range(5),"Species":list("abcdefghij")})
        reconstruction = FeatureExtractor(10).fit_transform(sim_data)
        self.assertTrue(np.allclose(reconstruction.sum("Feature").to_pandas(), sim_data))
    
    def test_fit_result_adds_up(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"Sample":range(5),"Species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"Sample":range(10,20),"Species":list("ghijk")})
        reconstruction = FeatureExtractor(10).fit(train_data).transform(test_data)
        self.assertTrue(np.allclose(reconstruction.sum("Feature").to_pandas(), test_data))