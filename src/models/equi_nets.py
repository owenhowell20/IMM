import torch
import torch.nn as nn
import escnn.nn as enn
from typing import List


# ----------------------------------------------------------------------------
# SO(3) equivariant linear layer


@persistence.persistent_class
class GeneralProjection(nn.Module):
    def __init__(
        self,
        input_multiplicities: List[torch.Tensor] = None,
        output_multiplicities: List[torch.Tensor] = None,
        group=None,
        device="cpu",
    ):
        super(GeneralProjection, self).__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = 0
        input_rep = []
        for i, mult in enumerate(input_multiplicities):
            input_rep += mult * [group.irrep(i)]
            input_dim += mult * input_dim
        self.input_rep = input_rep
        self.input_dim = input_dim

        output_rep = []
        output_dim = 0
        for i, mult in enumerate(output_multiplicities):
            output_rep += mult * [group.irrep(i)]
            output_dim += mult * output_dim
        self.output_rep = output_rep
        self.output_dim = output_dim

        self.in_type = enn.FieldType(group, self.input_rep)
        self.out_type = enn.FieldType(group, self.output_rep)

        self.proj = enn.Linear(self.in_type, self.out_type).to(self.device)

    def forward(self, x):

        # x is (b,n,input_dim) --> ( b, N , outputdim )
        x = enn.GeometricTensor(x.reshape(-1, x.shape[-1]), self.in_type).to(
            self.device
        )
        x = self.proj(x)

        return x
