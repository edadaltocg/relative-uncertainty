from .odin_postprocessor import ODINPostprocessor
from .base_postprocessor import BasePostprocessor
from .doctor_postprocessor import DoctorPostprocessor
from .partition_postprocessor import PartitionPostprocessor
from .random_forest_postprocessor import RandomForestPostprocessor
from .base_scikit_postprocessor import GenericScikitPostprocessor
from .mlp_postprocessor_scikit import MLPPostprocessor
from .conformal_postprocessor import ConformalPostprocessor


postprocessors = {
    "odin": ODINPostprocessor,
    "msp": BasePostprocessor,
    "doctor": DoctorPostprocessor,
    "partition": PartitionPostprocessor,
    "random_forest": RandomForestPostprocessor,
    "scikit": GenericScikitPostprocessor,
    "mlp": MLPPostprocessor,
    "conformal": ConformalPostprocessor,
    }

def get_postprocessor(postprocessor_name, model, cfg, result_folder, device):
    if postprocessor_name not in postprocessors:
        raise ValueError(f"Postprocessor {postprocessor_name} not found. Available postprocessors: {list(postprocessors.keys())}")
    return postprocessors[postprocessor_name](model, cfg, result_folder, device=device)
   
