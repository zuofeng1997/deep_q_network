import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__(in_features, out_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()
        bias = self.bias
        bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight+self.sigma_weight*self.epsilon_weight.data, bias)


class NoisyFactorizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.5):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        bias = bias +self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_in)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        if self.dueling:
            self.fc_val = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
        )
            self.fc_adv = self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
        )
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )

    def _get_conv_out(self, shape):
        test_tensor = self.conv(torch.zeros(1, *shape))
        return int(np.prod(test_tensor.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        if self.dueling:
            val = self.fc_val(conv_out)
            adv = self.fc_adv(conv_out)
            return val + adv - adv.mean()
        else:
            return self.fc(conv_out)
