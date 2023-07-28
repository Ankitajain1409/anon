import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
dist = tfp.distributions
import jax
from jax import jit, vmap
from jax.scipy.stats import norm


def gmm_mean_var(means_stack, sigmas_stack):
    means = jnp.stack(means_stack)
    final_mean = means.mean(axis=0)
    sigmas = jnp.stack(sigmas_stack)
    final_sigma = jnp.sqrt((sigmas**2 + means ** 2).mean(axis=0) - final_mean ** 2)
    return final_mean, final_sigma

def rmse(y,yhat):
  def rmse_loss(y,yhat):
      return (y-yhat)**2
  return jnp.sqrt(jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(y,yhat), axis = 0))

def NLL(mean,sigma,y):
    def loss_fn(mean, sigma, y):
      d = dist.Normal(loc=mean, scale=sigma)
      return -d.log_prob(y)
    return jnp.mean(jax.vmap(loss_fn, in_axes=(0, 0, 0))(mean, sigma, y))
    
def mae(y,yhat):
  def mae_loss(y,yhat):
      return jnp.abs(y-yhat)
  return jnp.mean(jax.vmap(mae_loss,in_axes=(0,0))(y,yhat), axis = 0)

def ace(dataframe):
    """
    dataframe : pandas dataframe with Ideal and Counts as column for regression calibration
    It can be directly used as 2nd output from calibration_regression in plot.py 
    """
    def rmse_loss(y,yhat):
      return jnp.abs(y-yhat)
    return jnp.mean(jax.vmap(rmse_loss,in_axes=(0,0))(dataframe['Ideal'].values,dataframe['Counts'].values))


@jit
def empirical_entropy(p):
    return jnp.mean(-jnp.log(p))

@jit
def entropy_of_normal(sigma):
    return 0.5 * (1 + jnp.log(2 * jnp.pi * sigma ** 2))


@jit
def entropy_of_GMM(means, sigmas):
    samples = jax.random.normal(shape=(1000, means.shape[0]), key  = jax.random.PRNGKey(0)) * sigmas + means
    probs = norm.pdf(samples, loc=means, scale=sigmas)
    gmm_probs = jnp.mean(probs, axis=1)
    return empirical_entropy(gmm_probs)

@jit
def get_all(means, sigmas):
    predictive_entropy = entropy_of_GMM(means, sigmas)
    expected_entropy = jnp.mean(entropy_of_normal(sigmas))
    mutual_information = predictive_entropy - expected_entropy
    return predictive_entropy, expected_entropy, mutual_information
