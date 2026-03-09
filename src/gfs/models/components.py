import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    Multilayer perceptron (MLP) model.

    Args:
      input_size: number of inputs.
      output_size: number of outputs.
      hidden: list of hidden layer widths.
      activation: nonlinearity between layers.
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU()):
        super().__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation function.
        self.activation = activation

    def forward(self, x):
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        return self.fc[-1](x)


class FeatureRegularizer(nn.Module):
    def __init__(self, l1=0.1, panel_size=None, priority_score=None, pairs=None, alpha=0.5, beta=0.5, gamma=0.5, strict=True):
        super().__init__()
        self.l1 = 0.01 if l1 is None else l1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.n_features = panel_size if panel_size else None
        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = None

        self.strict = strict

    def forward(self, x):
        abs_x = torch.abs(x)
        reg = torch.tensor(0., dtype=x.dtype, device=x.device)

        # Force weights toward 0 or 1
        reg += torch.sum(abs_x * torch.abs(x - 1))

        # Panel size constraint
        if self.n_features is not None:
            if self.strict:
                reg += torch.abs(torch.sum(abs_x) - self.n_features) * self.alpha
            else:
                reg += torch.max(torch.sum(abs_x) - self.n_features, 0) * self.alpha

        # Pairwise selection
        if self.pairs is not None:
            # Similar to tf.tensordot(abs_x, pairs, axes=1)
            pair_reg = torch.matmul(abs_x, self.pairs)
            reg += torch.sum(torch.abs(pair_reg)) * self.gamma

        return self.l1 * reg
