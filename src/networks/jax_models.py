import math
from typing import Sequence, Union

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.jaxrl import utils


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
        input_skip: bool = False,
        hidden_skip: bool = False,
        output_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = layers
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.input_skip = input_skip
        self.hidden_skip = hidden_skip
        self.output_skip = output_skip
        if layers == 1:
            hidden_dim = out_features
        self.input_layer = normed_activation_layer(
            rngs,
            in_features,
            hidden_dim,
            use_norm=use_norm,
            activation=hidden_activation,
        )
        self.main_layers = [
            normed_activation_layer(
                rngs,
                hidden_dim,
                hidden_dim,
                use_norm=use_norm,
                activation=hidden_activation,
            )
            for _ in range(layers - 2)
        ]
        self.norm = nnx.RMSNorm(in_features, rngs=rngs)
        self.output_layer = normed_activation_layer(
            rngs,
            hidden_dim,
            out_features,
            use_norm=use_output_norm,
            activation=output_activation,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        def _potentially_skip(skip, x, layer):
            if skip:
                return x + layer(x)
            else:
                return layer(x)

        if self.input_activation:
            # x = self.norm(x)
            x = self.hidden_activation(x)
        if self.layers == 1:
            return _potentially_skip(self.input_skip, x, self.input_layer)
        x = _potentially_skip(self.input_skip, x, self.input_layer)
        for layer in self.main_layers:
            x = _potentially_skip(self.hidden_skip, x, layer)
        return _potentially_skip(self.output_skip, x, self.output_layer)


class CriticNetwork(nnx.Module):
    @nnx.vmap(
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, 0)
    )
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        project_discrete_action: bool,
        use_norm: bool,
        use_encoder_norm: bool,
        use_simplical_embedding: bool,
        encoder_layers: int,
        head_layers: int,
        pred_layers: int,
        use_skip,
        rngs,
    ):
        rngs = nnx.Rngs(rngs)

        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
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
            input_skip=use_skip,
            input_activation=not use_simplical_embedding,
            hidden_skip=use_skip,
            layers=head_layers,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=utils.multi_softmax if use_simplical_embedding else None,
            use_norm=use_norm,
            use_output_norm=False,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            input_activation=not use_simplical_embedding,
            layers=pred_layers,
            rngs=rngs,
        )

    def features(self, obs: jax.Array, action: jax.Array):
        state = jnp.concatenate([obs, action], axis=-1)
        return self.feature_module(state)

    @nnx.vmap
    def critic_head(self, features: jax.Array) -> jax.Array:
        return self.critic_module(features)

    @nnx.vmap
    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    @nnx.vmap
    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    @nnx.vmap
    def forward(self, obs, action):
        features = self.features(obs, action)
        value = self.critic_head(features)
        pred = self.pred_module(features)
        pred_rew = pred[..., :1]
        pred_features = pred[..., 1:]
        return features, pred_features, pred_rew, value.squeeze(-1)


class CategoricalCriticNetwork(nnx.Module):
    @nnx.vmap(
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
        )
    )
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        project_discrete_action: bool,
        use_norm: bool,
        use_simplical_embedding: bool,
        encoder_layers: int,
        head_layers: int,
        pred_layers: int,
        num_bins: int,
        vmin: float,
        vmax: float,
        use_skip: bool,
        rngs: nnx.Rngs,
    ):
        rngs = nnx.Rngs(rngs)
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax

        self.use_skip = use_skip

        if project_discrete_action:
            self.action_embedding = nnx.Embed(
                num_embeddings=action_dim,
                features=hidden_dim // 2,
            )
            action_dim = hidden_dim // 2
        else:
            self.action_embedding = Identity()

        self.feature_module = FCNN(
            in_features=obs_dim + action_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=encoder_layers,
            hidden_skip=use_skip,
            output_skip=use_skip,
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
            input_skip=use_skip,
            hidden_skip=use_skip,
            rngs=rngs,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim + 1,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=None,
            layers=pred_layers,
            input_activation=not use_simplical_embedding,
            input_skip=use_skip,
            hidden_skip=use_skip,
            output_skip=False,
            rngs=rngs,
        )

        self.zero_dist = nnx.Param(
            utils.hl_gauss(jnp.zeros((1,)), num_bins, vmin, vmax)
        )

    def features(self, obs: jax.Array, action: jax.Array):
        action_embedding = self.action_embedding(action)
        state = jnp.concatenate([obs, action_embedding], axis=-1)
        return self.feature_module(state)

    def critic_head(self, features: jax.Array) -> jax.Array:
        cat = self.critic_module(features) + self.zero_dist.value * 40.0
        return cat

    def critic_cat_no_vmap(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    @nnx.vmap
    def critic_cat(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = self.features(obs, action)
        return self.critic_head(features)

    @nnx.vmap
    def critic(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        value_cat = jax.nn.softmax(self.critic_cat_no_vmap(obs, action), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        return value

    @nnx.vmap
    def forward(self, obs, action):
        features = self.features(obs, action)
        value_cat = jax.nn.softmax(self.critic_head(features), axis=-1)
        value = value_cat.dot(
            jnp.linspace(self.vmin, self.vmax, self.num_bins, endpoint=True)
        )
        preds = self.pred_module(features)
        pred_rew = preds[..., :1]
        pred_features = preds[..., 1:]
        if self.use_skip:
            pred_features = pred_features + features
        return features, pred_features, pred_rew, value


class SACActorNetworks(nnx.Module):
    @nnx.vmap(in_axes=(0, None, None, None, None, None, None, None, None, None, 0))
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        ent_start: float,
        kl_start: float,
        use_norm: bool,
        layers: int,
        min_std: float,
        use_skip: bool,
        rngs: nnx.Rngs,
    ):
        rngs = nnx.Rngs(rngs)
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
            hidden_skip=use_skip,
            rngs=rngs,
        )
        start_value = math.log(ent_start)
        kl_start_value = math.log(kl_start)
        self.temperature_log_param = nnx.Param(jnp.ones(1) * start_value)
        self.lagrangian_log_param = nnx.Param(jnp.ones(1) * kl_start_value)
        self.min_std = min_std

    @nnx.vmap
    def actor(self, obs: jax.Array) -> distrax.Distribution:
        loc = self.actor_module(obs)
        loc, log_std = jnp.split(loc, 2, axis=-1)
        std = jnp.exp(log_std) + self.min_std
        pi = distrax.Transformed(distrax.Normal(loc=loc, scale=std), distrax.Tanh())
        return pi

    @nnx.vmap
    def det_action(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, _ = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc)

    @nnx.vmap
    def temperature(self) -> jax.Array:
        return jnp.exp(self.temperature_log_param.value)

    @nnx.vmap
    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.lagrangian_log_param.value)

    @nnx.vmap
    def __call__(self, obs: jax.Array) -> jax.Array:
        loc = self.actor_module(obs)
        loc, std = jnp.split(loc, 2, axis=-1)
        return jnp.tanh(loc), std, self.temperature(), self.lagrangian()


class GumbleSoftmaxDistribution(distrax.Distribution):
    def __init__(self, logits: jax.Array, temperature: jax.Array):
        self.logits = logits
        self.temperature = temperature

    def sample(self, seed=None):
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).sample(seed=seed)

    def log_prob(self, value: jax.Array) -> jax.Array:
        return distrax.RelaxedOneHotCategorical(
            temperature=self.temperature, logits=self.logits
        ).log_prob(value)

    def sample_and_log_prob(self, *, seed, sample_shape=...):
        sample = self.sample(seed=seed)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class SACDiscreteActorNetworks(nnx.Module):
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
        use_skip: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor_module = FCNN(
            in_features=obs_dim,
            out_features=action_dim,
            hidden_dim=hidden_dim,
            hidden_activation=nnx.swish,
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            input_activation=False,
            hidden_skip=use_skip,
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
