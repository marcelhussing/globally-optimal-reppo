import math
from typing import Sequence, Union

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from reppo_alg.jaxrl import utils


def torch_he_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=jnp.float_,
):
    "TODO: push to jax"
    return nnx.initializers.variance_scaling(
        0.3333,
        "fan_in",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


class UnitBallNorm(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def normed_activation_layer(
    rngs, in_features, out_features, use_norm=True, activation=nnx.swish
):
    layers = [
        nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            kernel_init=torch_he_uniform(),
            rngs=rngs,
        )
    ]
    if use_norm:
        layers.append(nnx.RMSNorm(out_features, rngs=rngs))
    if activation is not None:
        layers.append(activation)
    return nnx.Sequential(*layers)


class Identity(nnx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class FCNN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm: bool = True,
        use_output_norm: bool = False,
        layers: int = 2,
        input_activation: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        if layers == 1:
            self.module = normed_activation_layer(
                rngs,
                in_features,
                out_features,
                use_norm=use_output_norm,
                activation=output_activation,
            )
        else:
            if input_activation:
                input_layer = nnx.Sequential(
                    # nnx.LayerNorm(in_features, rngs=rngs) if use_norm else Identity(),
                    hidden_activation,
                    normed_activation_layer(
                        rngs,
                        in_features,
                        hidden_dim,
                        use_norm=use_norm,
                        activation=hidden_activation,
                    ),
                )
            else:
                input_layer = nnx.Sequential(
                    normed_activation_layer(
                        rngs,
                        in_features,
                        hidden_dim,
                        use_norm=use_norm,
                        activation=hidden_activation,
                    )
                )
            hidden_layers = [
                normed_activation_layer(
                    rngs,
                    hidden_dim,
                    hidden_dim,
                    use_norm=use_norm,
                    activation=hidden_activation,
                )
                for _ in range(layers - 2)
            ]
            output_layer = normed_activation_layer(
                rngs,
                hidden_dim,
                out_features,
                use_norm=use_output_norm,
                activation=output_activation,
            )
            self.module = nnx.Sequential(
                input_layer,
                *hidden_layers,
                output_layer,
            )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.module(x)


class CriticNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        use_encoder_norm: bool = False,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=pred_layers,
            rngs=rngs,
        )

    def features(self, obs: jax.Array, action: jax.Array):
        state = jnp.concatenate([obs, action], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def forward(self, obs, action):
        features = self.features(obs, action)
        value = self.critic_head(features)
        return self.pred_module(features), value


class CategoricalCriticNetwork(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        use_norm: bool = True,
        use_encoder_norm: bool = False,
        use_simplical_embedding: bool = False,
        encoder_layers: int = 1,
        head_layers: int = 1,
        pred_layers: int = 1,
        num_bins: int = 51,
        vmin: float = -10.0,
        vmax: float = 10.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            rngs=rngs,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=self.num_bins,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            input_activation=not use_simplical_embedding,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            rngs=rngs,
        )
        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def features(self, obs: jax.Array, action: jax.Array):
        state = jnp.concatenate([obs, action], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features)  # + self.zero_dist.value * 40.0
        return cat

    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat(obs, action), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    def forward(self, obs, action):
        features = self.features(obs, action)
        value_cat = jax.nn.softmax(self.critic_head(features), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return self.pred_module(features), value

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        value_cat = jax.nn.softmax(self.critic_head(features), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        pred = self.pred_module(features)
        return value, value_cat, pred


class SACActorNetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim * 2,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    def __call__(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, std = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


class TD3ActorNetworks(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        ent_start: float = 0.1,
        kl_start: float = 0.1,
        use_norm: bool = True,
        layers: int = 2,
        min_std: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim * 2,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    def actor(
        self, obs: jax.Array, scale: float | jax.Array = 1.0
    ) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = (jnp.exp(log_std) + self.min_std) * scale
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        return pi

    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)


class TD3DeterministicDist(distrax.Distribution):
    def __init__(self, loc: jax.Array, scale: float | jax.Array):
        self.loc = loc
        self.scale = scale

    def sample(self, seed=None):
        return self.loc + self.scale * jax.random.normal(seed, self.loc.shape)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return jnp.zeros_like(value)

    def sample_and_log_prob(self, *, seed, sample_shape=...):
        sample = self.sample(seed=seed)
        log_prob = self.log_prob(sample)
        return sample, log_prob
