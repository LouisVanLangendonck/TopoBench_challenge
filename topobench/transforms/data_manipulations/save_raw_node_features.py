"""Transform that saves raw node features before any encoding."""

import torch
import torch_geometric


class SaveRawNodeFeatures(torch_geometric.transforms.BaseTransform):
    r"""Save raw node features before any encoding.
    
    This transform stores the original node features in a separate attribute
    (x_raw) before any feature encoding or transformations are applied.
    This is useful for reconstruction-based self-supervised learning tasks
    like GraphMAE where we want to reconstruct the original features,
    not the encoded ones.
    
    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "save_raw_node_features"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Save raw node features.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.
        
        Returns
        -------
        torch_geometric.data.Data
            Data with x_raw attribute containing a copy of the original features.
        """
        # Save a copy of the original node features
        if hasattr(data, 'x') and data.x is not None:
            data.x_raw = data.x.clone()
        
        return data

