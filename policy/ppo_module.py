import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):  # action inference, sample from NN.
        logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class ActorStd(Actor):
    def __init__(self, architecture, distribution, device='cpu'):
        super(ActorStd, self).__init__(architecture, distribution, device)

    def sample(self, obs):  # action inference, sample from NN.
        logits = self.architecture.forward(obs)  # [std, mean] for each environment.
        # shape (num_envs, act_dim)
        half = self.action_shape[0]
        std = logits[:, :half]
        mean = logits[:, half:]
        actions, log_prob = self.distribution.sample(mean, std=std)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        logits = self.architecture.forward(obs)
        half = self.action_shape[0]
        std = logits[:, :half]
        mean = logits[:, half:]
        return self.distribution.evaluate(obs, mean, actions, std=std)

    @property
    def action_shape(self):
        shape = self.architecture.output_shape.copy()
        shape[0] = int(shape[0]/2)
        return shape

    # TODO: implement
    def noiseless_action(self, obs):
        return None

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        #modules.append(nn.Tanh())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class HafnerActorModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(HafnerActorModel, self).__init__()

        self.modules = [nn.Linear(input_size, 256), nn.ELU(),
                   nn.LayerNorm(256),
                   nn.Linear(256, 256), nn.ELU(),
                   nn.Linear(256, 100), nn.ELU(),
                   nn.Linear(100, output_size), nn.ELU(),
                   nn.Linear(output_size, output_size), nn.Tanh()
                   ]
        scales = [np.sqrt(2)]*5 # scales for all layers weights
        self.architecture = nn.Sequential(*self.modules)

        self.init_weights(self.architecture, scales)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class HafnerActorModelStd(HafnerActorModel):
    def __init__(self, input_size, output_size):
        super(HafnerActorModelStd, self).__init__(input_size, output_size)
        self.modules.pop()  # pops the last activation.
        scales = [np.sqrt(2)]*5 # scales for all layers weights
        self.architecture = nn.Sequential(*self.modules)
        self.init_weights(self.architecture, scales)

    def forward(self, x):
        # adds the last custom activation.
        y = self.architecture(x)  # size [num_env, act_dim*2]
        output = y.clone()
        half = int(self.output_shape[0]/2)
        # std
        # TODO: should I add the backward pass?
        f = nn.Tanh()
        output[:, :half] = (f(y[:, :half])+1)*0.5*0.7+0.3  # in [0.3;1.0]
        # Mean
        output[:, half:] = f(y[:, half:])  # in [-1.0;1.0]
        return output


class HafnerCriticModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(HafnerCriticModel, self).__init__()

        modules = [nn.Linear(input_size, 400), nn.ELU(),
                   nn.LayerNorm(400),
                   nn.Linear(400, 400), nn.ELU(),
                   nn.Linear(400, 300), nn.ELU(),
                   nn.Linear(300, output_size)
                   ]
        scales = [np.sqrt(2)]*4 # scales for all layers weights
        self.architecture = nn.Sequential(*modules)

        self.init_weights(self.architecture, scales)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None

    def sample(self, logits, std=None):
        if std is not None:
            self.distribution = Normal(logits, std)
        else:
            self.distribution = Normal(logits, self.std.reshape(self.dim))
        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)
        return samples, log_prob

    def evaluate(self, inputs, logits, outputs, std=None):
        if std is not None:
            distribution = Normal(logits, std)
        else:
            distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
