from SpaCoObject import SPACO
from M_Z_csv import SpectrumData
from pathlib import Path
import pandas as pd

class MultiModalRegistration(SpectrumData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for multimodal registration can go here

    def load_massi_spectrometry_data(self, full_data=False, sample_features=True):
        """
        Loads mass spectrometry data from imzML file into SpectrumData object.

        Parameters
        ----------
        full_data : bool, optional
            If True, returns the full SpectrumData object dataframe and the coordinates of the samples.
        sample_features : bool, optional
            If True, returns the sample features matrix generated using the _sample_feature_genration method and the coordinates of the samples.

        Returns
        -------
        If full_data is True, returns a tuple containing the SpectrumData object dataframe and a list of the coordinates of the samples.
        If sample_features is True, returns a tuple containing the sample features matrix and a list of the coordinates of the samples.
        """
        
        # transform to full dataframe
        if full_data: 
            return self.df
        
        # transform to sample features matrix 
        if sample_features: 
            return self._sample_feature_genration(agg_func="sum")
        


    # Additional methods for multimodal registration can go here
if __name__ == "__main__":
    print("This is a module for multimodal registration using SPACO and SpectrumData classes.")

    # loading data and pathways 
    imzml_path = Path("/home/krastegar0/MALDI_Metabolomics/MSI_data_grant/Mass_Spec_data/20251012_old_liver_area.imzML")

    # Example usage:
    multimodal_registration = MultiModalRegistration(
        imzml_path=imzml_path, 
        min_intensity=100, 
        min_count=100, 
        mz_tol=0.0042
        )
    
    # get the MALDI data matrix and coordinates
    maldi_data_matrix, coords = multimodal_registration.load_massi_spectrometry_data(full_data=False, sample_features=True)

    # make coordinates into pandas dataframe
    coords_df = pd.DataFrame(coords[0], columns=['X', 'Y'])
    
    # see if I can make this sampling heirarchy work
    # start with the subsetting the function i.e) make the masks 

    print("MALDI data matrix shape:", maldi_data_matrix.shape)




