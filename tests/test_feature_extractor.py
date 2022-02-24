import unittest
from metagen_pathogenicity_score.PathogenicityScore import FeatureExtractor
import xarray as xr
import numpy as np

if __name__ == "__name__":
    unittest.main()

class TestFeatureExtractor(unittest.TestCase):
    def test_missing_species(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"sample":range(10,20),"species":list("abcde")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_new_species(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"sample":range(10,20),"species":list("ghijk")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_new_species_only(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"sample":range(10,20),"species":list("klmno")})
        result = FeatureExtractor(3).fit(train_data).transform(test_data)
        self.assertFalse(result.isnull().any())
    
    def test_fit_transform_result_adds_up(self):
        sim_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"species":list("abcdefghij")})
        reconstruction = FeatureExtractor(10).fit_transform(sim_data)
        self.assertTrue(np.allclose(reconstruction.sum("feature").to_pandas(), sim_data))
    
    def test_fit_result_adds_up(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"species":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"sample":range(10,20),"species":list("ghijk")})
        reconstruction = FeatureExtractor(10).fit(train_data).transform(test_data)
        self.assertTrue(np.allclose(reconstruction.sum("feature").to_pandas(), test_data))

    def test_with_genus(self):
        train_data = xr.DataArray(np.abs(np.random.randn(5,10)), {"sample":range(5),"genus":list("abcdefghij")})
        test_data = xr.DataArray(np.abs(np.random.randn(10,5)), {"sample":range(10,20),"genus":list("ghijk")})
        reconstruction = FeatureExtractor(10).fit(train_data).transform(test_data)
        reconstruction_onestep = FeatureExtractor(10).fit_transform(test_data)
        self.assertTrue(np.allclose(reconstruction.sum("feature").to_pandas(), test_data))
        self.assertTrue(np.allclose(reconstruction_onestep.sum("feature").to_pandas(), test_data))