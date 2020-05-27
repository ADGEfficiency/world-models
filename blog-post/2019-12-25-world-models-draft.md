---
title: 'World Models (the long version)'
date: 2019-12-21
categories:
  - Python, Machine Learning, Reinforcement Learning
excerpt: Ha & Schmidhuber's World Models reimplemented in Tensorflow 2.0.

---

todo
-a pictures on the forward passes bit of the VAE
- lstm testing sections
- redo the monthly commits post, size, yaxis, plus other analysis of work done (put into methods section?)
- transitions & summary between all the major sections

we, I, our agent

<center>
	<img src="/assets/world-models/f0.gif">
<figcaption>Performance of the final agent on a conveniently selected random seed. The cumulative episode reward is shown in the lower left.  </figcaption>
<figcaption>This agent & seed achieves 893. 900 is solved.</figcaption>
</center>

<p></p>

# Table of Contents

todo

# Resources

> I didn't have time to write a short letter, so I wrote a long one instead - Mark Twain

The resources used to develop the code base and write this post are in [ADGEfficiency/rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

The reimplementation code base is here - [ADGEfficiency/world-models](https://github.com/ADGEfficiency/world-models).

I plan to follow up this longer post with a shorter version, when this is published I will link it here.

# Summary

> Don’t believe everything you think - BJ Miller

**My main side project in 2019 was this reimplementation of 2018's World Models by David Ha & Jürgen Schmidhuber**.  The paper introduced a novel machine learning algorithm that solved a previously unsolved continuous action space, pixel observation space control problem.

The paper trains an agent on two environments, `Car-Racing-v0` and `ViSDoom`.  The scope of this reimplementation is only of the `Car-Racing-v0` work.

Over the ten months, my work on the project averaged !! days per month.

On average, I committed 12.5 times per month.  You can see the variance of my monthly commitment below:

<center>
	<img src="/assets/world-models/commits-month.png">
	<figcaption>Commits per month (excludes blog post commits).  Not all commits are made equal.</figcaption>
  <div></div>
</center>

The distribution of the final agent performance is below (see more in Section !):

<center>
	<img src="/assets/world-models/final_hist.png">
<figcaption>Histogram of the best agent (Agent Five, generation 229) episode rewards across 48 random seeds.  900 is solved.</figcaption>
</center>

I spent a total of ! on the project (read more in !):

|*AWS costs*|   Cost [$] |
|:--------------------|-----------:|
| compute-total       |       2485 |
| storage-total       |       1162 |
| total               |       3648 |

# Motivations and Context

> In order to be a perfect and beautiful computing machine, it is not requisite to know what arithmetic is - Alan Turing

## Why reimplement a paper?

The idea came from Open AI job advertisement.  Seeing a tangible goal that could put me in the ballpark of a world class machine learning lab, I set out looking for a paper to reimplement.  

**The advertisement specified a high quality implementation**.  This requirement echoed in my mind as I developed the project.

## Why reimplement World Models?

World Models is one of three reinforcement learning papers that have been significant for me.

### DQN

The first paper was blew me away without even reading it.  I have a memory of seeing a YouTube video DQN playing the Atari game Breakout.  Even though I knew nothing of reinforcement learning, the significance of a machine could learn to play a video game from pixels was clear.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/V1eYniJ0Rnk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<figcaption>DQN playing Atari Breakout</figcaption>
</center>

I had no way of knowing that the algorithm I was watching would be one I implement four times, or that I would teach the mechanics of DQN and it's evolution into Rainbow over twenty times.

### AlphaGo Zero

The second paper was AlphaGo Zero.  The publication of AlphaGo Zero in October 2017 came out after I had taught my course on reinforcement learning twice.

<center>
	<img src="/assets/world-models/Zero_act_learn.png">
	<figcaption>
		<a href="https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ">Silver et. al (2017) Mastering the game of Go without human knowledge</a> with additional annotations
	</figcaption>
	<div></div>
</center>

At this stage (9 months after my transition into data science), I didn't fully grasp all of the mechanics in AlphaGo.  But I knew enough to understand the significance of the changes - *tabula rasa* learning among the most important.

### World Models

<center>
	<img src="/assets/world-models/ha-blog.png">
<figcaption><a href="https://worldmodels.github.io/">https://worldmodels.github.io/</a></figcaption>
</center>

The third paper was World Models. World Models is an example of strong technical work presented well. Ha & Schmidhuber's [2018 paper](https://github.com/ADGEfficiency/rl-resources/blob/master/world-models/2018_Ha_world_models.pdf) was accompanied by a [blog post](https://worldmodels.github.io/) that was both interactive and full of `.gif`, making the work engaging and impressive. 

Alongside the presentation sits technical work that we will explore in detail in this post. I had never worked with any of the techniques before reimplementing World Models. The influence has been visible in the projects of my students at [Data Science Retreat](https://www.datascienceretreat.com/).  

Shout out to Mack (who used a mixed density network to predict hospital triage codes), Samson (who used a variational auto-encoder to detect images of football games) and Stas (who combined the World Models vision & memory with a PPO controller).  **Thank you to them for allowing me to improve my understanding in parallel with theirs.**

## The promise of learning a model of the world

**A world model is an abstract representation of the spatial or temporal dimensions of our world**.  A world model can be useful in a number of ways.

**One use of a world model is to use their low dimensional, internal representations for control**.  We will see that the World Models agent uses it's vision and memory in this way.  The value of having these low dimensional representations is that both prediction and control are easier in low dimensional spaces.

**Another is to generate data for training**.  A model that is able to approximate the environment transition dynamics can be used recurrently to generate rollouts of simulated experience.

These two uses can be combined together, where world models are used to generate synthetic rollouts in the low dimensional, internal representation spaces.  This is learning within a dream, and is demonstrated by Ha & Schmidhuber on the `ViSDoom` environment.

The value of these different approaches is clear to anyone who has spent time building or learning environment models. The sample inefficiency of modern reinforcement learning agents means that an environment model is required for training - sampling from the real world is too slow.

I encountered this problem as an data scientist at Tempus Energy.  We had no simulator of our business problem, and struggled to learn one from limited amounts of customer data.  Part of the motivation of this reimplementation was to learn more about how environment models can be learnt.

Now that we understand the motivations and context of this project, we can look at one side of the Markov Decision Process (MDP) coin - the environment.  *If you aren't familiar with what an MDP is, take a look at the [Appendix]().*

# The Environment

> Do not seek to have events happen as you want them to, but instead want them to happen as they do happen, and your life will go well - Epictetus

Our agent interacts with the `car-racing-v0` environment from OpenAI's `gym` library.  I used the same version of `gym` as the paper codebase (`gym==0.9.4`).

`car-racing-v0` is a continous action space probelm.

## Working with `car-racing-v0` 

We can describe the `car-racing-v0` environment as a Markov Decision Process.  

In the `car-racing-v0` environment, the agents **observation space** is raw image pixels $(96, 96, 3)$.

The observation has both a spatial $(96, 96, 3)$ and temporal structure, given the sequential nature of sampling transitions from the environment.  An observation is always a single frame.

The **action space** has three continuous dimensions - `[steering, gas, break]`.  This is a continuous action space - the most challenging for control.

The **reward** function is $-0.1$ for each frame, $+1000 / N$ for each tile visited, where $N$ is the total tiles on track.  This reward function encourages quickly driving forward on the track.

The **horizon** (aka episode length) is set to $1000$ throughout the paper codebase.  

### Getting the resizing right

- this is cropped and resized to $(64, 64, 3)$.

This was where I made a mistake in my first agent - as you can see from the `.gif` below, the image renders much different than the images shown in the paper.  The reason is a different resampling filter.

<center>
	<img src="/assets/world-models/first.gif">
<figcaption>Performance of Agent One with the incorrect resizing.  Note that the poor driving is due to an exploration problem.</figcaption>
</center>

You can see more detail on the resizing in the notebook [world-models/notebooks/resizing-observation.ipynb](https://github.com/ADGEfficiency/world-models-dev/blob/master/notebooks/resizing-observation.ipynb).  The final function used is given below, and uses `PIL.Image.BILINEAR`:

```python
def process_frame(frame, screen_size=(64, 64), vertical_cut=84, max_val=255, save_img=False):
    """ crops, scales & convert to float """
    frame = frame[:vertical_cut, :, :]
    frame = Image.fromarray(frame, mode='RGB')
    obs = frame.resize(screen_size, Image.BILINEAR)
    return np.array(obs) / max_val
```

<center>
	<img src="/assets/world-models/f1-final.png">
	<figcaption>The raw observation (96, 96, 3) - the correctly resized observation (64, 64, 3) - the learnt latent variables (32,)</figcaption>
  <div></div>
</center>

### Avoiding corrupt observations

One important hack is to use `env.viewer.window.dispatch_events()` in the `reset()` and `step()` methods of the environment ([see the GitHub issue here](https://github.com/openai/gym/issues/976)).  If you don't use it you will get corrupt environment observations!

<center>
	<img src="/assets/world-models/corrupt.jpeg">
	<figcaption>If you see this, your environment observation is corrupt!</figcaption>
  <div></div>
</center>

```python
#  do this before env.step(), env.reset()
self.viewer.window.dispatch_events()
```

See the notebook [`worldmodels/notebooks/car_race_consistency.ipynb`](https://github.com/ADGEfficiency/world-models/blob/master/notebooks/car_race_consistency.ipynb) for more about avoiding corrupt `car-racing-v0` observations.


Below the code for sampling environment observations is given in full - see the source in ([world-models/dataset](https://github.com/ADGEfficiency/world-models/blob/master/worldmodels/dataset)).

```python
# worldmodels/dataset/car_racing.py
# worldmodels/dataset/*
```


# The Agent

> We shall not cease from exploration, and the end of all our exploring will be to arrive where we started and know the place for the first time - T.S. Eliot

<center>
	<img src="/assets/world-models/agent.png">
	<figcaption>The World Models agent</figcaption>
  <div></div>
</center>

[The World Models agent is a Popperian intelligence](https://adgefficiency.com/four-competences/) - able to improve via global selection, respond to reinforcement and to learn models of it's environment.  These models can be used for offline improvement, without having to sample more experience from the environment.

## The three components

The World Models agent has three components - vision, memory and a controller.

**The first component is vision**.  The vision component uses a Variational Autoencoder (VAE) to compress the environment observation $x$ into a latent space $z$ and then reconstruct it into $x'$.

The controller doesn't ever use the reconstruction $x'$ - instead it uses the lower dimensional latent representation $z$. This low dimensional, latent representation $z$ is used as one of the controllers two inputs.

**The second component is memory**.  This component uses a long short-term memory (LSTM) with a mixed density network to predict environment transitions in latent space - to predict $z'$ given $z$ and an action $a$.

The controller doesn't ever use the prediction $z'$ (which represents only one step in the future) but instead the hidden state $h$ of the LSTM. The hidden state $h$ contains information many steps into the future - it is the controllers second input.

**The third component is the controller**.  The controller is a linear function that maps the vision latent state $z$ and memory hidden state $h$ to an action $a$.  The controller parameters are found using an evolutionary algorithm called CMA-ES, with a fitness function of total episode reward.

The attentive reader will note that we do not ever use the vision or the memory in the traditional way - either for the vision's reconstructed environment observation or the memory's predicted next latent state.  **The controller only uses internal representations**.

These components are trained sequentially and independently:
- first train the vision on observations
- then train the memory using sequences of observations and actions.  The observations are encoded into latent space $z$ statistics, meaning we can sample a different $z$ each epoch
- finally train the controller, using the vision to provide $z$ and the memory to provide $h$, using total episode reward

This independent learning has a number of consequences.

Both the vision & memory components are trained without access to rewards - they learn only from observations and actions. Rewards are only used to learn parameters of the controller.

By training the vision and memory without access to rewards, these components are not required to concurrently learn both representations of the environment and do credit assignment.

This allows the vision and memory to focus on one task.  It can however mean that without the context of reward information, the vision & memory might learn features of the environment that are not useful for control.

By training the vision and memory on a fixed dataset, the training will be more stable than the morre common situation in reinforcement learning, where as the policy improves the data distribution changes.

Whether you learn from a fixed dataset or a non-stationary one, the exploration-exploitation dilemma cannot be ignored.  The fixed dataset will have been generated with some policy.

# Vision

> It's not what you look at that matters, it's what you see - Henry David Thoreau

<center>
	<img src="/assets/world-models/vae.png">
	<figcaption>The World Models vision</figcaption>
  <div></div>
</center>

The vision is a generative model, that models the conditional distribution of the environment observation $x$ and a latent representation $z$:

$$z \sim E_{\theta}(z \mid x)$$

## Why do we need to see?

We use vision to understand our environment - our agent does the same. **Here we define vision as dimensionality reduction** - the process of reducing high dimensional data into a lower dimensional space.

The World Models vision provides a low dimensional representation of the environment to the controller.  The value of this representation is that it is easier to make decisions in low dimensional spaces.

A canonical example in computer vision is image classification, where an image can be mapped throughout a convolutional neural network to a predicted class. A business can use that predicted class to make a decision. Another example is the flight or fight response, where visual information is mapped to a binary decision.

In our `car-racing-v0` environment, a low dimensional representation of the environment observation might be something like:

```python
observation = [on_road=1, corner_to_the_left=1, corner_to_the_right=0, on_straight=1]
```

Using this representation, we could imagine deriving a simple control policy.  Try to do this with $27,648$ numbers, even if they are arranged in a shape $(96, 96, 3)$.

Note that we don't know which variables to have in our latent representation. **The latent representation is hidden.** It is unobserved - for a given image, we have no labels for these variables.  We don't know how many there are - or if they exist at all!

## How does our agent see?

We have a definition of vision as reducing high dimensional data into a low dimension, so that we can use it to take actions.  How then does our agent see?

The vision of the World Models agent reduces the environment observation $x$ $(96, 96, 3)$ into a low dimensional, latent representation $z$ $(32,)$.

How do we learn this latent representation if we don't have examples? The World Models vision uses a Variational Autoencoder.

## The Variational Autoencoder

**A Variational Autoencoder (VAE) forms the vision of our agent.**

The VAE is a generative model that learns the data generating process.  The data generating process is $P(x,z)$ - the joint distribution over our data (the probability of $x$ and $z$ occurring together).

*[If the generative/discriminative concept is unfamiliar, take a look at Appendix !]().*

### The VAE in context

The VAE sits alongside the Generative Adversarial Network (GAN) as the state of the art in generative modelling.

The figure below shows the outstanding progress in image quality generated by GANs.  GANs typically outperform VAEs on reconstruction quality, with the VAE providing better support over the data (support meaning the number of different values a variable can take).

<center>
	<img src="/assets/world-models/gan.png">
	<figcaption>
		Progress in GANS - <a href="https://www.iangoodfellow.com/slides/2019-05-07.pdf">Adverserial Machine Learning - Ian Goodfellow - ICLR 2019</a>
	</figcaption>
  <div></div>
</center>

The VAE has less in common with classical (sparse or denoising) autoencoders, which both require the use of the computationally expensive Markov Chain Monte Carlo.

### What makes the VAE a good choice for the World Models agent?

A major benefit of generative modelling is the ability to generate new samples $x'$.  Yet our World Models agent never uses $x'$ (whether a reconstruction or a new sample).

The role of the VAE in our agent is to provide a compressed representation $z$ by learning to encode and decode a latent space.  This lower dimensional latent space is easier for our memory and controller to work with.

What qualities do we want in our latent space?  **One is meaningful grouping**.  This requirement is a challenge in traditional autoencoders, which tend to learn spread out latent spaces.

Meaningful grouping means that similar observations exist in the same part of the latent space, with samples that are close together in the latent space producing similar images when decoded.  This grouping means that even observations that the agent hadn't seen before could be responded to the same way.

**Meaningful grouping allows interpolation**.  Encoding similar observations close together makes the space between observed data meaningful.  

So how do we get meaningful encoding? The intuition behind autoencoders is to constrain the size of the latent space ($32$ variables for the World Models VAE).  The VAE takes this one step further by imposing a Kullback-Leibler Divergence on the latent space - we will see more on this in Section !.

## VAE structure

The VAE is formed of three components - an encoder, a latent space and a decoder.  

As the raw data is an image, the VAE makes use of convolution and deconvolution layers.  [If you need a quick refresher on convolution, see Appendix !]().

### Encoder

**The primary function of the encoder is recognition**.  The encoder is responsible for recognizing and encoding the hidden latent variables**.

The encoder is built from convolutional blocks that map from the input image ($x$) (64, 64, 3) to statistics (means & variances) of the latent variables (length 64 - 2 statistics per latent space variable).

### Latent space

Constraining the size of the latent space (length 32) is one way auto-encoders learn efficient compression of images.  All of the information needed to reconstruct a sample $x$ must exist in only 32 numbers!

The statistics parameterized by the encoder are used to form a distribution over the latent space - a diagonal Gaussian.  A diagonal Gaussian is a multivariate Gaussian with a diagonal covariance matrix.  This means that each variable is independent.

(is this enforcing a gaussian prior or posterior?)

This parameterized Gaussian is an approximation.  Using it will limit how expressive our latent space is,

$$z \sim P(z \mid x)$$

$$ z \mid x \approx \mathbf{N} \Big(\mu_{\theta}, \sigma_{\theta}\Big) $$

We can sample from this latent space distribution, making the encoding of an image $x$ stochastic.

Because the latent space fed to the decoder is spread (controlled by the parameterized variance of the latent space), it learns to decode a range of variatons for a given $x$.

Ha & Schmidhuber propose that the stochastic encoding leads to a more robust controller in the agent.

### Decoder

The decoder uses deconvolutional blocks to reconstruct the sampled latent space $z$ into $x'$.  In the World Models agent, we don't use the reconstruction $x'$ - we are interested in the lower dimensional latent space representation $z$.

The agent uses the latent space is used in two ways
- directly in the controller
- as features to predict $z'$ in the memory

But we aren't finished with the VAE yet - in fact we have only just started.

## The three forward passes

Now that we have the structure of the VAE mapped out, we can be specific about how we pass data through the model.

### Compression

$x$ -> $z$

- encode an image $x$ into a distribution over a low dimensional latent space
- sample a latent space $z \sim E_{\theta}(z \mid x)$

### Reconstruction

$x$ -> $z$ -> $x'$

- encode an image $x$ into a distribution over a low dimensional latent space
- sample a latent space $z \sim E_{\theta}(z \mid x)$
- decode the sampled latent space into a reconstructed image $x' \sim D_{\theta}(x' \mid z)$

### Generation

$z$ -> $x'$

- sample a latent space $z \sim P(z)$
- decode the sampled latent space into a reconstructed image $x' \sim D_{\theta}(x' \mid z)$

```python
# worldmodels/vision/vae.py
```

## The backward pass

*This section owes much to the excellent tutorial [What is a variational autoencoder? by Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/).*

We do the backward pass to learn - maximizing the joint likelihood of an image $x$ and the latent space $z$.

The VAE uses likelihood maximization to learn this joint distribution $P(x,z)$.  **Likelihood maximization maximizes the similarity between two distributions**.  In our case these distributions are over our training data (the data generating process, $P(x,z)$) and our parametrized approximation (a convolutional neural network $E_{\theta}(z \mid x))$.

Let's start with the encoder. We can write the encoder $E_{\theta}$ as model that given an image $x$, is able to sample the latent space $z$.  The encoder is parameterized by weights $\theta$:

$$ z \sim E_{\theta}(z \mid x) $$

The encoder is an approximation of the true posterior $P(z \mid x)$ (the distribution that generated our data).  Bayes Theorem shows us how to decompose the true posterior:

$$P(z \mid x) = \dfrac{P(x \mid z) \cdot P(z)}{P(x)}$$

The challenge here is calculating the posterior probability of the data $P(x)$ - this requires marginalizing out the latent variables. Evaluating this is exponential time:

$$P(x) = \int P(x \mid z) \cdot P(z) \, dz$$

The VAE sidesteps this expensive computation by *approximating* the true posterior $P(z \mid x)$ using a diagonal Gaussian:

$$ x \mid z \sim \mathbf{N} \Big(\mu_{\theta}, \sigma_{\theta}\Big) $$

$$P(x \mid z) \approx E(x \mid z ; \theta) = \mathbf{N} \Big(x \mid \mu_{theta}, \sigma_{\theta}\Big)$$

**This approximation is variational inference** - using a family of distributions (in this case Gaussian) to approximate the latent variables.  Using variational inference is is a key contribution of the VAE.

Now that we have made a decision about how to approximate the latent space distribution, we want to think about how to bring our parametrized latent space $E_{\theta}(z \mid x)$ closer to the true posterior $P(z \mid x)$.

In order to minimize the difference between our two distributions, we need way to measure the difference.  The VAE uses a Kullback-Leibler divergence ($\mathbf{KLD}$),  which has a number of interpretations:
- measures the information lost when using one distribution to approximate another
- measures a non-symmetric difference between two distributions
- measures how close distributions are

$$\mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z \mid x) \Big) = \mathbf{E}_{z \sim E_{\theta}} \Big[\log E_{\theta}(z \mid x) \Big] - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log P(x, z) \Big] + \log P(x)$$

This $\mathbf{KLD}$ is something that we can minimize - it is a loss function.  But our exponential time $P(x)$ (in the form of $\log P(x)$) has reappeared!

Now for another trick from the VAE.  We will make use of
- the Evidence Lower Bound ($\mathbf{ELBO}$)
- Jensen's Inequality

The $\mathbf{ELBO}$ is given as the expected difference in log probabilities when we are sampling our latent vectors from our encoder $E_{\theta}(z \mid x)$:

$$\mathbf{ELBO}(\theta) = \mathbf{E}_{z \sim E_{\theta}} \Big[\log P(x,z) - \log E_{\theta}(z \mid x) \Big]$$

Combining this with our $\mathbf{KLD}$ we can form the following:

$$\log P(x) = \mathbf{ELBO}(\theta) + \mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z \mid x) \Big) $$

Jensen's Inequality tells us that the $\mathbf{KLD}$ is always greater than or equal to zero. Because $\log P(x)$ is constant (and does not depend on our parameters $\theta$), a large $\mathbf{ELBO}$ requires a small $\mathbf{KLD}$ (and vice versa).

Remember that we have a $\mathbf{KLD}$ we want to minimize!  We have just shown that we can do this by ELBO maximization.  After a bit more mathematical massaging )) we can arrive at:

$$ \mathbf{ELBO}(\theta, \theta) = \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta}(x' \mid z) \Big] - \mathbf{KLD} \Big (E_{\theta}(z \mid x) \mid \mid P(z) \Big) $$

Note the appearance of our decoder $D_{\theta}(x \mid z)$.  The decoder is used to approximate the true posterior $P(x' \mid z)$ - the conditional probability distribution over the reconstruction of latent variables into a generated $x'$ (given $x$).

The last step is to convert this $\mathbf{ELBO}$ maximization into a more familiar loss function minimization.  We now have the VAE loss function's final mathematical form - in all it's tractable glory:

$$ \mathbf{LOSS}(\theta) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

The loss function has two terms - the log probability of the reconstruction (aka the decoder) and a $\mathbf{KLD}$ between the latent space (sampled from our encoder) and the latent space prior $P(z)$.

Remember that the loss function above is the result of minimizing the $\mathbf{KLD}$ between our encoder $E_{\theta}(z \mid x)$ and the data generating distribution $P(z \mid x)$.  What we have is a result of maximizing the log-likelihood of the data.

## Implementing the loss function in code

Although our loss function is in it's final mathematical form, we will make three more modifications before we implement it in code:
- convert the log probability of the decoder into a pixel wise reconstruction loss
- use a closed form solution to the $\mathbf{KLD}$ between our encoded latent space distribution and the prior over our latent space $P(x)$
- refactor the randomness using reparameterization

### First term - reconstruction loss

$$ - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] $$

The first term in the VAE loss function is the log-likelihood of reconstruction - given latent variables $z$, the distribution over $x'$.  The latent variables are sampled from our encoder (hence the expectation $\mathbf{E}_{z \sim E_{\theta}}$).

Minimizing the negative log-likelihood is equivalent to likelihood maximization.  In our case, the likelihood maximization maximizes the similarity between  the distribution over our training data $P(x \mid z)$ and our parametrized approximation.

[Section 5.1 of the Deep Learning textbook](http://www.deeplearningbook.org/) shows that for a Gaussian approximation, maximizing the log-likelihood is equivalent to minimizing mean square error ($\mathbf{MSE}$):

$$\mathbf{MSE} = \frac{1}{n} \sum \Big( \mid \mid x' - x \mid \mid \Big)^{2} $$

**This gives us the form of loss function that is often implemented in code - a pixel wise reconstruction loss** (also known as an L2 loss):

```python

```

### Second term - regularization

$$ \mathbf{LOSS}(\theta) = - \mathbf{E}_{z \sim E_{\theta}} \Big[ \log D_{\theta} (x' \mid z) \Big] + \mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big)  $$

The intuition of the second term in the VAE loss function is compression or regularization.

The second term in the VAE loss function is the $\mathbf{KLD}$ between the and the latent space prior $P(z)$.

We haven't yet specified what the prior over the latent space should be.  A convenient choice is a Standard Normal - a Gaussion with a mean of zero, variance of one.

Minimizing the $\mathbf{KLD}$ means we are trying to make the latent space look like random noise.  It encourages putting encodings near the center of the latent space.

The KL loss term further compresses the latent space.  This compression means that using a VAE to generate new images requires only sampling from noise!  This ability to sample without input is the definition of a generative model.

Because we are using Gaussians for the encoder $E_{theta}(z \mid x)$ and the latent space prior $ P(z) = \mathbf{N} (0, 1) $, the $\mathbf{KLD}$ has closed form solution ([see Odaibo - Tutorial on the VAE Loss Function](https://arxiv.org/pdf/1907.08956.pdf)).

$$\mathbf{KLD} \Big( E_{\theta} (z \mid x) \mid \mid P(z) \Big) = \frac{1}{2} \Big( 1 + \log(\sigma_{\theta}^{2}) - \sigma_{\theta}^{2} - \mu_{\theta} \Big)$$

This is how the $\mathbf{KLD}$ is implemented in the VAE loss.  Note that we use a clip on the KLD at half the latent space sizein order to balance the two losses:


```python

```

A note on the use of $\log \sigma^{2}$ - we force our network to learn this by taking the exponential later on in the program:

```python

```

### Reparameterization trick

Because our encoder is stochastic, we need one last trick - a rearrangement of the model architecture, so that we can backprop through it.  This is the **reparameterization trick**, and results in a latent space architecture as follows:

$$ n \sim \mathcal{N}(0, 1) $$

$$ z = \sigma_{theta} (x) \cdot n + \mu_{theta} (x) $$

After the refactor of the randomness, we can now take a gradient of our loss function and train the VAE.

### Final VAE loss function

todo

## Vision - Summary

Below a few 

<center>
	<img src="/assets/world-models/vae-training.png">
<figcaption>Training curve of the first iteration VAE over 8 epochs.  You can clearly see the effect of the `kl_tolerance=16`.</figcaption>
</center>

<center>
	<img src="/assets/world-models/vae-reconstructions.png">
<figcaption>Observations and their reconstructions from the final VAE, along with the reconstruction and KLD losses</figcaption>
</center>

<center>
	<img src="/assets/world-models/vae-noise.png">
<figcaption>Images reconstructed from sampling the latent space prior $z \sim \mathbf{N} (0, 1)$</figcaption>
</center>

The contributions of the VAE are:
- variational inference to approximate
- compression / regularization of the latent space using a KLD between our learnt latent space and a prior $P(z) = \mathbf{N} (0, 1)$
- stochastic encoding of a sample $x$ into the latent space $z$ and into a reconstruction $x'$

The reasons why we use the VAE in the World Models agent:
- learn the latent representation $z$

```python
# worldmodels/vision/vae.py

# worldmodels/vision/train_vae.py
```

# Memory

> Prediction is very difficult, especially if it's about the future - Nils Bohr

<center>
	<img src="/assets/world-models/memory.png">
	<figcaption>The World Models memory</figcaption>
  <div></div>
</center>


The memory is a discriminative model, that models the conditional probability of seeing an environment transition in latent space, from $z$ to $z'$, conditional on an action $a$:

$$ P(z' | z, a) $$

## Why do we remember?

Memory has many uses - some think that the entire purpose of a human life is to generate memories, to be looked back on after a life well lived.

falability of memory, generative / constructive

## What the agent remembers

Our agent remembers past transitions in order to predict future transitions.  However, the controller does not actually make use of the prediction, instead it makes use of an internal representation - the hidden state of an LSTM.

The memory is trained to predict transitions in the latent space learnt by the vision's VAE.

The primary role of the memory in the World Models agent is compression of the future into a low dimensional representation $h$.  This low compression of time $h$ is the hidden state of an LSTM.

The LSTM to trained only to predict the next latent vector $z'$, but learns a longer representation of time via hidden state (specifically $h$)

The memory's life is made easier by being able to predict in the low dimensional space learnt by the VAE.

The memory has two components - an LSTM and a Gaussian Mixture head.  **Both these together form a Mixed Density Network** (MDN).  The MDN was introduced in 1994 by Christopher Bishop in the context of fully connected networks.  The MDN was originally introduced with fully connected layers connected to a Gaussian Mixture head.

The motivation behind MDN's is being able to combine a neural network (that can represent arbitrary non-linear functions) with a mixture model (that can model arbitrary conditional distributions).

##  Gaussian Mixtures

In the section ! above we saw that if we make the assumption of Gaussian distributed data, we can derive the mean square error loss function from likelihood maximization.  This loss function leads to learning of the conditional average of the target.

Learning the conditional average can be useful, but also has drawbacks.  For multimodal data, taking the average is unlikely to be informative.

A primary motivation behind using a mixture of distributions is that we can approximate **multi-modal** distributions.

Bishop (?) shows that by training a neural network using a least squares loss function, we are able to learn two statistics.  One is the conditional mean, which is our prediction.  The second statistic is the variance, which we can approximate from the residual.  We can use these two statistics to form a Gaussian.

Being able to learn both the mean and variance motivates the paramterization of a mixture model with Gaussian kernels.

A mixture model is a linear combination of kernel functions:

$$ P(y \mid x) = \sum_{mixes} \alpha(x) \cdot \phi(y \mid x) $$

Where $\alpha$ are mixing coefficients, and $\phi$ is a conditional probability density.  Our kernel of choice is the Gaussian, which has a probability density function:

$$ \phi (z' \mid z, a) = \frac{1}{\sqrt{(2 \pi) \sigma(z, a)}} \exp \Bigg[ - \frac{\lVert z' - \mu(z, a) \rVert^{2}}{2 \sigma(z, a)^{2}} \Bigg] $$

The cool thing about Gaussian mixtures is there ability to approximate complex probability densities using Gaussian's with a diagonal covariance matrix.

Probability distribution output by a mixture can (in principle!) be calculated.  The flexibility is similar to a feed forward neural network, and likely has the same distinction between being able approximate versus being able to learn.

In practice, the mixture probabilities are parameterized as $log \pi$, recovering the probabilities by taking the exponential.  These probabilities are priors of the target having been generated by a mixture component.  These are transformed via a softmax:

$$ \pi = \frac {\exp (\pi)}{\sum exp(\pi)} $$

Meaning our mixture satisfies the constraint:

$$ \sum_{mixes} \pi(z, a) = 1 $$

As with the VAE, the memory $\theta$ parameters are found using likelihood maximization.

$$ M(z' \mid z, a) = \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

$$ \mathbf{LOSS} = - \log M(z' \mid z, a)$$

$$ \mathbf{LOSS} = - \log  \sum_{mixes} \alpha(z, a) \cdot \phi (z'| z, a) $$

In a more general setting, the variances learnt by a Gaussian mixture can be used as a measure of uncertainty.

A mixture model requires statistics (probabilities, means and variances) as input.  In the World Models memory, these statistics are supplied by a long short-term memory (LSTM) network.

If you want refresher on LSTM see Appendix Four!.

```python
#worldmodels/memory/memory.py - GaussianMix
```


```python
#worldmodels/memory/memory.py

#worldmodels/memory/train_memory.py
```

Note that even when the memory has the ability to think 'long-term', this still pales in comparison to the long term memorpwe posess.  The authors note that a higher caparity external memory is needed for explorinc moce complex worlds.

## Putting the Memory together

$$ M_{\theta}(z'| z, a, h, c) $$

## Implementing the Memory in code

From a software development perspective, development of the `Memory` class was done in two distinct approaches.

### Performance based testing

The first was testing the generalization of the MDN on a toy dataset.  The inspiration and dataset came directly from [Mixture Density Networks with TensorFlow](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/) by David Ha. It is in Tensorflow 1.0, which required updating to Tensorflow 2.0.  You can see the notebook I used to develop the MDN + LSTM at [world-models/notebooks/memory-quality-check.ipynb]().

To isolate any bugs I first tested the Gaussian mixture head with a fully connected net.

![]()

The next test was with an LSTM generating the statistics of the mixture:

![]()

### Unit testing

The performance based testing was combined with lower level unit style testing (the tests are still within the notebooks).  You can see the notebook I used to develop the MDN + LSTM at [world-models/notebooks/Gaussian-mix-kernel-check.ipynb]() - a short snippet is below.

```python
import math
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

sample = 0
mu = 0
sig = 0.1

nor = tfd.Normal(loc=mu, scale=sig)
nor.prob(sample)
# <tf.Tensor: id=124, shape=(), dtype=float32, numpy=3.989423>

constant = 1 / math.sqrt(2 * math.pi)
gaussian_kernel = np.subtract(sample, mu)
gaussian_kernel = tf.square(tf.divide(gaussian_kernel, sig))
gaussian_kernel = - 1/2 * gaussian_kernel
tf.divide(tf.exp(gaussian_kernel), sig) * constant
# <tf.Tensor: id=133, shape=(), dtype=float64, numpy=3.989422804014327>
```

I also wrote tests to check if I was saving and loading the memory correctly, and that I was able to control the passing of the hidden state correctly - see []().

### Memory - Summary

<center>
	<img src="/assets/world-models/memory-training.png">
<figcaption>Training curve for the memory for Agent Four (40 epochs)</figcaption>
</center>

```python
# mem, train mem

```

# Control

> Never let the future disturb you. You will meet it, if you have to, with the same weapons of reason which today arm you against the present - Marcus Aurelius

The final component of our agent is the controller.  The controller uses the compressed representations of the current and future environment.  These are provided by the vision and memory components.

$$ C_{\theta}(a' \mid z, h) $$

Both of these representations are learnt without access to the rewards information.  It is only the controller that has access to reward data, in the form of total episode reward.

The controller is a simple linear function that maps from these compressed representations ($z$ and $h$) to an action $a$.  We want to find values for these parameters that will maximize the expected reward of our agent in the `Car-Racing-v0` environment.

The algorithm used by the World Models agent for finding these parameters is Covariance Matrix Adapation Evolution Stragety (CMA-ES).

## Why do we need control?

Decision making requires a goal. In a Markov Decision Process, the goal is to maximize expected reward.

To maximize expected reward an agent must perform **credit assignment** - determining which actions lead to reward. An agent that understands how to assign credit can take good actions.

In reinforcement learning, this credit assignment is often learnt by a function that also learns other tasks, such as learning a low dimensional representation of a high dimensional image.  An example is DQN, the action-value function learns to both extract features from the observation and map them to an action.

In the World Models agent these tasks are kept separate, with vision responsible for learning a spatial representation and the memory learning a temporal representation.  This separation allows the use of a simple linear controller, completely dedicated to learning how to assign credit.

Having a simple, low parameter count controller opens up less sample efficient but more general methods for finding the model parameters.  The downside of this is that our vision and memory might use capacity to learn features that are not useful for control, or not learn features that are useful.

## How the agent controls

The agent controls using all three of it's components.  The vision takes the environment observation $x$ and encodes it into a latent representation $z$.  The memory uses the latent representation of the environment and the last action $a$ to predict $z'$, updating it's hidden state $h$ in the process.

The controller takes the latent representation on the current environment observation $z$ and the memory LSTM hidden state $h$ and maps to an action $a$.

The question is how we find good parameters for our linear controller. For MDPs, reinforcement learning is a common choice.  However, our low parameter count ($784$ parameters) controller opens up more options - the one chosen by the World Models agent chooses an evolutionary algorithm called CMA-ES.  Before we dive into the details of computational evolution and CMA-ES, we will consider evolution.

## Evolution

[Evolution is an example of Darwinian competence](https://adgefficiency.com/four-competences/), with agents that don't learn within their lifetime.  From a computational perspective, this means that the controller parameters are fixed within each generation.

Evolution is the driving force in our universe.  At the heart of evolution is a paradox - failure at a low level leading to improvement at a high level.  Examples include biology, business, training neural networks and personal development.

Evolution is iterative improvement using a generate, test, select loop:
- in the **generate** step a population is generated, using infomation from previous steps
- in the **test** step, the population interacts with the environment, and is assigned a single number as a score
- in the **select** step, members of the current generation are selected (based on their fitness) to be used to generate the next step

There is so much to learn from this evolutionary process:
- failure at a low level driving improvement at a higher level
- the effectiveness of iterative improvement
- the requirement of a dualistic (agent and environment) view


We now have an understanding of the general process of evolutionary learning.  Let's look at how we do this *in silico*.

## Computational evolution

Computational evolutionary algorithms are inspired by biological evolution.  They perform non-linear, non-convex and gradient free optimization.  Evolutionary methods can deal with the challenges that discontinuities, noise, outliers and local optima pose in optimization.

Computational evolutionary algorithms are often the successive process of sampling parameters of a function (i.e. a neural network) from a distribution.  This process can be further extended by other biologically inspired mechanics, such as crossover or mutation - known as genetic algorithms.

A common Python API for computation evolution is the **ask, evaluate and tell** loop.  This can be directly mapped onto the generate, test & select loop introduced above:

```python
for population in range(populations):
  #  generate
  parameters = solver.ask()
  #  test
  fitness = environment.evalute(parameters)
  #  select
  solver.tell(parameters, fitness)
```

From a practical standpoint, the most important features of computational evolutionary methods are:
- general purpose optimization that can handle noisy, ill-conditioned, non-linear problems
- poor sample efficiency, due to learning from a weak signal (total episode reward)
- parallelizable, due to the rollouts of each population member being independent of the other population members

### General purpose optimization

Evolutionary algorithms learn from a single number per generation - the total episode reward.  This single number serves as a measurement of a population's fitness.

This is why evolutionary algorithms are **black box** - unlike less general optimizers they don't learn from the temporal structure of the MDP.  They are also gradient free.

This black box approach, combined with a reliance on random search, allows evolutionary methods to be robust in challenging search spaces.  They can handle problems that other, more complex optimization methods struggle with, such as discontinuities, local optima and noise.

### Sample inefficiency

The cost of having a general purpose learning method is sample efficiency. By not exploiting information such as the temporal structure of an episode, or gradients, evolutionary methods must rely on lots of sampling to learn.

How sample efficient an algorithm is depends on how much experience (measured in transition between states in an MDP) an algorithm needs to achieve a given level of performance.  It is of key concern if compute is purchased on a variable cost basis.

The inherit sample inefficiency of evolutionary algorithms is counteracted by requiring less computation per episode (i.e. no gradient updates inbetween transitions) and being able to parallelize rollouts.

### Parallel rollouts

A key feature of Darwinian learning is fixed competence, with population members not learning within lifetime.  This feature means that each population member can learn independently, and hence be parallelized.  This is a major benefit of evolutionary methods, which helps to counteract their sample inefficiency.

## `ADGEfficiency/evolution`

I hadn't worked with evolutionary algorithms before this project.  Due to possessing a simple mind that relies heavily on empirical understanding, I often find implementing algorithms a requirement for understanding.

The simplest evolution strategy (ES) is $(1, \lambda)$-ES.  This basic algorithm involves sampling a population of parameters from a multivariate Gaussian, using the best performing member of the previous generation as the mean and a fixed, identity covariance matrix.

I implemented $(1, \lambda)$-ES and a wrapper around `pycma` in a separate repo [ADGEfficiency/evolution](https://github.com/ADGEfficiency/evolution) - refer to the repo for more on the algorithms and optimization problems implemented.  Below is the performance of $(1, \lambda)$-ES on the simple `Sphere` optimization problem:

<center>
	<img src="/assets/world-models/sphere-simple-solver.gif">
<figcaption></figcaption>
</center>

There are a number of problems with $(1, \lambda)$-ES.  A major one is a fixed covariance matrix - meaning that even after the approximation of the mean is good, the population is still spread with the same variance.  The evolutionary algorithm the World Models agent addresses this problem, by adapting the covariance matrix.  **This algorithm is CMA-ES**.

## CMA-ES

*This section was heavily influenced by the excellent [Hansen (2016) The CMA Evolution Strategy: A Tutorial](https://arxiv.org/pdf/1604.00772.pdf)*.

The Covariance Matrix Adapation Evolutionary Stragety (CMA-ES) is the algorithm used by our agent to find parameters of it's linear controller.

A key feature of CMA-ES is the successive estimation of a full covariance matrix.  **Unlike the algorithms we have discussed above, CMA-ES approximates a full covariance matrix of our parameter space**. This means that we model the pairwise dependencies between parameters - how one parameter changes with another.

This is different to the multivariate Gaussians we parameterized in the vision and memory components.  These have diagonal covariances, which mean each variable changes independently of the other variables.

If you are wondering whether CMA-ES could be useful for your control problem, David Ha suggests that CMA-ES is effective for up to 10k parameters, as the covariance matrix calculation is $O(n^{2})$.

We can describe CMA-ES in the context of the generate, test and select loop that defines evolutionary learning.

### Generate

The generation step involves sampling a population from a multivariate Gaussian, parameterized by a mean $\mu$ and covariance matrix $\mathbf{C}$:

$$ x \sim \mu + \sigma \cdot \mathbf{N} \Big(0, \mathbf{C} \Big) $$

### Test

The test step involves parallel rollouts of the population parameters in the environment.  In the World Models agent, each parameter is rolled out $16$ times, with the results being averaged to give the fitness for each set of parameters.  This leads to a total of $1,024$ rollouts per generation!

### Select

The selection step involves selecting the best $n_{best}$ members of the population.  These population members are used to update the statistics of our multivariate Gaussian.

We first update our estimate of the mean using a sample average over $n_{best}$ from the current generation $g$:

$$ \mu_{g+1} = \frac{1}{n_{best}} \sum_{n_{best}} x_{g} $$

Our next step is to update our covariance matrix $C$.  You can find a refresher on estimating a covariance matrix from samples in [Appendix Four]().

The CMA-ES covariance matrix estimation is more complex than this, and involves the combination of two updates known as rank-one and rank-$\mu$.  Combining these update strategies helps to prevent degeneration with small population sizes, and to improve performance on badly scaled or non-separable problems.

### Rank-one update

In the context of our World Models agent, we might estimate the covariance of our next population $g+1$ using our samples $x$ and taking a reference mean value from that population:

$$ \mathbf{C}_{g+1} = \frac{1}{N_{best} - 1} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g+1}} \Big) \Big( x_{g+1} - \mu_{x_{g+1}} \Big) $$

Using the mean of the actual sample $\mu_{g+1}$ leads to an estimation of the covariance within the sample. The approach used in a rank-one update instead uses a reference mean value from the **previous generation** $g$:

$$ \mathbf{C}_{g+1} = \frac{1}{N_{best}} \sum_{pop} \Big( x_{g+1} - \mu_{x_{g}} \Big) \Big( x_{g+1} - \mu_{x_{g}} \Big) $$

Using the mean of the previous generation $\mu_{g}$ leads to a covariance matrix that estimates the covariance of the **sampled step**.  The rank-one update introduces information of the correlations between generations using the history of how previous populations have evolved - known as the **evolution path**:

$$ p_{g+1} = (1-c_{c})p_{g} + \sqrt{c_{c} (2-c_{c} \mu_{eff})} \frac{\mu_{g+1} - \mu_{g}}{\sigma_{g}} $$

Where $c_{c}$ and $\mu_{eff}$ are hyperparameters.

The evolution path is a sum over all successive steps, but can be evaluated using only a single step - similar to how we can update a value function over a single transition.  The final form of the rank-one update is below:

$$ \mathbf{C}_{g+1} = (1-c_{1}) \mathbf{C}_{g} + c_{1} p_{g+1} p_{g+1}^{T} $$

### Rank-$\mu$ update

With the small population sizes required by CMA-ES, getting a good estimate of the covariance matirx using a rank-one update is challenging.  The rank-$\mu$ update uses a reference mean value that uses information from all previous generations.  This is done by taking an average over all previous estimated covariance matrices:

$$ \mathbf{C} = \frac{1}{g+1} \sum_{gens} \frac{1}{\sigma^{2}} \mathbf{C} $$

We can improve on this by using an exponential weights $w$ to give more influence to recent generations.  CMA-ES also includes a learning rate $c_{\mu}$, to control how fast we update:

$$ \mathbf{C}_{g+1} = (1-c_{\mu}) \mathbf{C} + c_{\mu} \sum_{gens} w \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big) \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big)^{T} $$

### CMA-ES step-size control

The covariance matrix estimation we see above does not explicitly control for scale.  CMA-ES implements an different evolution path ($p_{sigma}$) that is independent of the covariance matrix update seen above, known as **cumulative step length adaptation** (CSA).  This helps to prevent premature convergence.

The intuition CSA is:
- for short evolution paths, steps are cancelling each other out -> decrease the step size
- for long evolution paths, steps are pointing in the same direction -> increase the step size

To determine whether an observed evolution path is short or long, the path length is compared with the expected length under random selection.  Comparing the observed evolution path with a random (i.e. independent) path allows CMA-ES to determine how to update the step size parameter $c_{\sigma}$.

Our evolution path $p_{\sigma}$ is similar to the evolution path $p$ except it is a conjugate evolution path.  After some massaging, we end up with a step size update:

$$ \sigma_{g+1} = \sigma_{g} \exp \Big[ \frac{c_{\sigma}}{d_{\sigma}} \Big(  \frac{\mid\mid p_{\sigma, g} \mid\mid}{ \mathbf{E} \mid\mid \mathbf{N} (0, I)} - 1 \Big) \Big] $$

Where $c_{\sigma}$ is a hyperparameter controlling the backward time horizon, and $d_{\sigma}$ is a damping parameter.

### The final CMA-ES update

The mean is updated using a simple average of the $N_{best}$ population members from the previous generation:

$$ \mu_{g+1} = \frac{1}{N_{best}} \sum_{N_{best}} x_{g} $$

The covariance matirx is updated using 

$$ C_{g+1} = (1 - c_{1} - c_{\mu} \cdot \sum w) \cdot C_{g} + c_{1} \cdot p_{g+1} \cdot p_{g+1}^{T} + c_{\mu} \cdot \sum_{gens} w \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big) \cdot \Big( \frac{x_{g+1} - \mu_{g}}{\sigma_{g}} \Big)^{T} $$

These updates allow separate control of the mean, covariance and step-size:
- mean update controlled by $c_{m}$
- covariance matrix $\mathbf{C}$ update controlled by $c_{1}$ and $c_{\mu}$
- step size update controlled by damping parameter $d_{sigma}$

Phew!

## Implementing the controller & CMA-ES

Above we looked at some of the mechanics of CMA-ES.  

I did not need to reimplement CMA-ES from scratch - I used [`pycma`](https://github.com/CMA-ES/pycma).

Using `pycma` required only a simple wrapper class around the ask, evaluate and tell API of `pycma`.

For each generation the rollout of the linear controller parameters are parallelized using Python's `multiprocessing`.  When using `multiprocessing` with both `pycma` and `tensorflow`, care is required to import these packages at the correct place - **within the child process**.  Do these imports in the wrong place, and you are going to have a bad time.

The original runs 16 rollouts per generation, with the fitness for a population member being the average across the 16 rollouts.  With a population size of 64, this leads to 1024 rollouts per generation.

I also experienced a rather painful bug where some episode rollouts would stall.  This would lead to one of the 64 processes not returning, holding up all the other processes.

My solution to this was a band-aid - putting an alarm on AWS to terminate the instance if the CPU% fell below 50% for 5 minutes, along with code to restart the experiment from the latest generation saved in `~/world-models-experiments/control/generations/`.

```python
#worldmodels/control
```

# Timeline

Below is a rough outline of the work done on the 11 months of this project.  Eight months were spent on the technical reimplementation, with three writing this blog post.

### April 2019

The first commit I have for this project is **April 6th 2019**.  Work achieved in this month
- sampling a random policy
- the VAE model & training script finished
- memory development (LSTM hidden state)

29 - mdn (nan loss, cgant fine tune, using notebook, num mixes)
- linear combination of kernels

### May 2019

Work achieved in this month
- development of memory model & training scripts
- working on understanding evolutionary methods
- `tf.data`

### June 2019

I didn't work on this project in June - I was busy with lots of teaching for Batch 19 at Data Science Retreat.

### July 2019

Work achieved in this month
- development of memory
- first run of the full agent *Agent One* - achieved an average of 500

assets/world/first.png etc

### August 2019

- transfer from `ADGEfficiency/mono` to `ADGEfficiency/world-models-dev`.
- train second VAE with fixed resize

### September 2019

- train second memory

### October 2019

Very little work done in October - I was busy with lots of teaching for Batch 20 at Data Science Retreat.

- working on controller training
- move out of TF 2.0 beta

### November 2019

Work achieved this month:
- controller training development - saving parameters, ability to restart, random seeds for environment
- sampling episodes from trained controller
- train **Agent Two** - problem with the VAE not being able to encode images (i.e. off track), memory trains well - gets confused when on the edge of track
- train **Agent Three** - using data sampled from the controller (5000 episodes),
- train **Agent Four** - using data sampled from the controller, 40 epochs on mem

| Agent | Policy | Episodes | VAE epochs | Memory epochs |
|---|---|---|
|one| random | 10,000 | 10 | 20 |
|two| random | 10,000 | 10 | 20 |
|three| controller two | 5,000 | 10 | 20 |
|four| controller three |5,000 | 15 | 40 |
|five| controller three |5,000 | 15 | 80 |

### December 2019

> Know you don’t hit it on the first generation, don’t think you hit it on the second, on the third generation maybe, on the fourth & fifth, thats when we start talking - Linus Torvalds

This was the final month of technical work (finishing on December 19), where Agent Five was trained.  Work achieved this month:
- training Agent Five
- code to visualize the rollouts of the Agent Five controller
- code cleanup & refactors

### January 2020

- blog post writing
- refactors and code cleanup

### February 2020

- draft one done (13 Feb)
- readme cleanup, code cleanup

### March 2020

- code to download pretrained model
- draft two done (7 March)

# Methods

I believe in not duplicating text - the details on methodology are all held in the [reimplementation Github repo](https://github.com/ADGEfficiency/world-models).

## Training from scratch

The methodology for training the entire agent from scratch (including the second iteration) is given in the `readme.md` of the Github repo.  The basic methodology is:
- sample rollouts from a random policy
- train a VAE using the random policy rollouts
- sample the VAE statistics (mean & variances of the latent space) for the random policy data
- train the memory by sampling from the VAE statistics
- train the controller using the VAE & memory

## Using a pretrained vision, memory and controller

The methodology for using pretrained agent is given in the `readme.md` of the Github repo - it involves running a bash script `pretrained.sh` to download the pretrained vision, memory & controller from a Google Drive link.

## Main differences from the original code base

- requirement of a second it

# Final results

This section summarizes the performance of the final agent, along with training curves for the agent components. Due to the expense of training the controller (see the section on AWS costs below), [I was very glad to find the following from David Ha](http://blog.otoro.net/2018/06/09/world-models-experiments/):

> After 150-200 generations (or around 3 days), it should be enough to get around a mean score of ~ 880, which is pretty close to the required score of 900. 
> If you don’t have a lot of money or credits to burn, I recommend you stop if you are satistifed with a score of 850+ (which is around a day of training). 
> Qualitatively, a score of ~ 850-870 is not that much worse compared to our final agent that achieves 900+, and I don’t want to burn your hard-earned money on cloud credits. To get 900+ it might take weeks (who said getting SOTA was easy? :) 

The training curve for the controller.  We show a much worse performing minimum than Ha & Schmidhuber, perhaps due to the use of a much higher `sigma` in `pycma`.

<center>
	<img src="/assets/world-models/final.png">
<figcaption>Training of the controller for Agent Five</figcaption>
</center>

Performance of the best controller (generation 299):

<center>
	<img src="/assets/world-models/final_hist.png">
<figcaption>Histogram of the best agent (generation 229) episode rewards across 48 random seeds</figcaption>
</center>

An example of the debug gif I used at various stages of the project.

<center>
	<img src="/assets/world-models/rollout.gif">
<figcaption>A tool used for debugging - notice how noisy the memory prediction can be!</figcaption>
</center>

# Discussion

In this section I give a few miscellaneous thoughts on work done in this project.

## Requirement of an iterative training procedure

The most notable difference between this reimplementation and the 2018 paper is the requirement of iterative training.

Section 5 of Ha & Schmidhuber (2018) notes that they were able to train a world model using a random policy, and that more difficult environments would require an iterative training procedure.  This was not true with our reimplementation - we required two iterations - one using data from a random policy to train the full agent, then a second round using data sampled from the first agent.

The paper codebase implements a random policy by randomly initializing the VAE, memory and controller parameters.  The reimplementation [ctallec/world-models](https://github.com/ctallec/world-models) has two methods for random action sampling - white noise (using the `gym` `env.action_space.sample()` or as a Brownian motion ([see here](https://github.com/ctallec/world-models/blob/master/utils/misc.py)).  The Brownian motion action sampling is the default.

This suggests that slightly more care is needed than relying on a random policy.  An interesting next step would be to look at optimizing the frequency of the iterative training for a given budget of episode sampling.

## Important debugging steps

Most of the time I stuck to using the same hyperparameters as in the paper code base.  Hyperparameters I changed:
- batch size to 256 in the VAE training (originally !)
- CMA-ES `s0` set to 0.5
- amount of training data & epochs for the later iterations of VAE & memory training

- image antialiasing
- VAE not performing well when it went off track (loss + inspect the reconstruction) - exporation problem

## Thoughts on Tensorflow 2.0

This reimplementation was started during the beta of Tensorflow 2.0.  It is a massive improvement over Tensorflow 1.0.

The main learning points are adapting to the style of inheriting from `tf.keras.Model`, and then using the `__call__` method of your child class to implement the forward pass.

The only issue I had was with passing around the hidden state of the LSTM.  At the time [I wrote a short blog post about my experience](https://adgefficiency.com/tf2-lstm-hidden/) - I plan on revisiting this to see if moving out of beta has fixed my issue.

## Thoughts on `tf.data`

For datasets larger than memory, batches must loaded from disk as needed.  Holding a buffer of batches also makes sense to keep GPU utilization high.

One way to achieve this is using `tf.data`.  The API for this library is challenging, and we were required to use the `tf.data` at three different levels of abstraction (a tensor of floats, multiple tensors and a full dataset).

```python

```

Two types of `tfrecord` files were saved and loaded:
- observations and actions for an episode (random or controller policy) - used to train VAE
- VAE latent statistics for an episode - used to train memory

```python

```

Two configurations of `tf.data.Dataset` were used
- VAE trained using a dataset of shuffled observations
- memory trained using a dataset shuffled on the episode level (need to keep the episode sequence structure)

The coverage for our implementation of `tf.data` is tested - see [worldmodels/tests/test_tf_records.py']().

Occasionally I get corrupt records - a small helper utility is given in [worldmwodels/utils.py]():

It is possible to load the `.tfrecord` files directly from S3.  As neural network training requires multiple passes over the dataset, it makes more sense to pull these down onto the instance using the S3 CLI and access them locally.

## AWS lessons

I had experience running compute on AVS beefore this project, but not on setting up an entire account from scratch.  The progress was fairly painless, with a reasonable around of time configuring the infrastructure I needed.

The main tasks to get setup were:
- using IAM to create a user (seen as a best practice, even though my account has only one user)
- creating an S3 bucket, with permissions for the IAM user
- security group with ports open for SSH
- requesting allowances for instances - slightly frustrating, but the AWS support always got back within 24 hours
- using a `setup.txt` to automate some of the instance setup

Most of these tasks involved creating a few wrappers around the AWS CLI:

```bash
```

The `setup.txt` I used:

```bash
```

Didn't use spot at all!

Crashing of controller training

### AWS costs

Compute costs = EC2

Storage costs = EBS + S3

Breakdown of total costs per component (compute costs only) (TABLLE IS WRRONG)

| component         |   Cost [$] |   Cost [%] |
|:------------------|-----------:|-----------:|
| controller        |    1309.72 |      40.19 |
| vae-and-memory    |     263.04 |       8.07 |
| sample-experience |      56.68 |       1.74 |
| total             |    3258.87 |     100    |

Per component, per month:

| month   |   controller |   vae-and-memory |   sample-experience |   sample-latent-stats |   misc |   compute-total |    s3 |    ebs |   storage-total |   total |
|:--------|-------------:|-----------------:|--------------------:|----------------------:|-------:|----------------:|------:|-------:|----------------:|--------:|
| 1/04/19 |         0    |             0    |                0    |                  0    |   0    |            0    |  0    |   0    |            0    |    0    |
| 1/05/19 |         0    |             0    |                0    |                  0    |   0    |            0    |  0    |   0    |            0    |    0    |
| 1/06/19 |         0    |            90.94 |               29.37 |                  0    |  16.99 |          137.93 |  0    | 153.23 |          153.23 |  291.16 |
| 1/07/19 |         0    |           208.51 |                0.83 |                254.51 |   8.05 |          471.9  |  0    | 304.72 |          304.72 |  776.62 |
| 1/08/19 |         0    |           144.97 |               16.35 |                  0    |   0    |          162.62 | 15.41 | 447.38 |          462.79 |  625.41 |
| 1/09/19 |         0    |            25.25 |                0    |                  0    |   0    |           25.25 | 11.12 | 854.57 |          865.69 |  890.93 |
| 1/10/19 |       104.51 |             0    |                0    |                  0    |   0    |          104.51 | 11.12 | 108.17 |          119.29 |  223.8  |
| 1/11/19 |       673.23 |             0    |               48.33 |                  0    |   0    |          721.56 | 11.42 | 116.86 |          128.29 |  849.84 |
| 1/12/19 |       728.43 |           132.27 |                0.51 |                  0    |   0    |          861.72 |  4.91 | 285.53 |          290.44 | 1152.16 |

One painful mistake occured in September 2019 - leaving a ~ 1TB SSD volume sitting unconnected for a month, leading to a very expensive month!

<center>
	<img src="/assets/world-models/aws.png" width="300%" height="300%">
<figcaption></figcaption>
</center>

## Improvements

### Thoughts on future hyperparameter tweaking

As I was working on this project, a number of other hyperparameters that could be optimized came to mind:
- VAE loss balancing - in the paper implementation this is done using a `kl_tolerance` parameter of 0.5

```python
kl_loss = tf.reduce_mean(
		tf.maximum(unclipped_kl_loss, self.kl_tolerance * self.latent_dim)
)
```

- improving the random policy sampling
- number of mixtures of in the mixed density network
- number of rollouts per generation

Agent horizon
Changing this can have interesting effects on agent performance.  It is unclear how convenient this choice is - LSTM's (which power the agent's memory) can often struggle with very long sequences.

## What did I learn / takeaways

### Reimplement papers

This was the first machine learning paper I have reimplemented.  It is something I am going to do again, and would highly recommend.

### More tools in the machine learning toolbelt

I hadn't worked with a VAE or MDN before this project.  Both are now important tools in my toolbox - I expect MDN's in particular to be very useful in business context's, due to the estimation of uncertainty that the parameterized variance provides.

I now have a decent grasp of evolutionary methods. I know there are many more algorithms out there than CMA-ES, 

### Create bodies of work

One of the more pleasant insights I had while working on this project was discovering the many blog posts by David Ha on the various components of the World Models agent (in particular MDNs & evolutionary algorithms).


[Mixture Density Networks with TensorFlow](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/) by David Ha.  

It is inspiring to actually be able to see and follow some of the strands of the development.  It is also a perfect example of why working in machine learning is amazing - world class reseachers publically sharing content that helps others learn.

Another example of this insight isDavid Silver

Inspiring to see that greatness is built in small steps.  This reimplementation of World Models is one of my small steps.

# References

A full list of reference material for this project is kept in a repository I use to store my reinforcement learning resources - you can find the world models references at [ADGEfficiency/rl-resources/world-models](https://github.com/ADGEfficiency/rl-resources/tree/master/world-models).

# Appendix

## Appendix One - Markov Decision Process

A Markov Decision Process (MDP) is a mathematical framework for decision making.  Commonly the goal of an agent in an MDP is to maximize the expectation of future rewards.  It can be defined as a tuple:

$$ (\mathcal{S}, \mathcal{A}, \mathcal{R}, P, R, d_0, \gamma, H) $$

- set of states $\mathcal{S}$
- set of actions $\mathcal{A}$
- set of rewards $\mathcal{R}$
- state transition function $ P(s' \mid s,a) $
- reward transition function $ R(r \mid s,a,s') $
- distribution over initial states $d_0$
- discount factor $\gamma$
- horizion $H$

It is common to make the distinction between the state $s$ and observation $x$.  The state represents the true state of the environment and has the Markov property. The observation is what the agent sees.  The observation is less informative, and often not Markovian.

Because the World Models agent uses the total episode reward as a learning signal, there is no role for a discount rate $\gamma$.

The data collected by an agent interacting with an environment is a sequence of transitions, with a transition being a tuple of observation, action, reward and next state:

$$ (x, a, r, x') $$

- $x$ observation
- $a$ action
- $r$ reward
- $x'$ next observation

## Appendix Two - Generative versus discriminative models

To better understand the context of the VAE, let's take a quick detour into a useful categorization of predictive modelling - generative versus discriminative.

All approaches in predictive modelling can be categorized as either generative or discriminative.

### Generative models

**Generative models learn a joint distribution** $P(x, z)$ (the probability of $x$ and $z$ occurring together).  Generative models generate new, unobserved data $x'$.

We can derive this process for generating new data, from the definition of conditional probability:

$$ P(x \mid z) = \frac{P(x, z)}{P(z)} $$

Rearranging this definition gives us a decomposition of the joint distribution. This is the product rule of probability:

$$P(x, z) = P(x \mid z) \cdot P(z)$$

This decomposition describes the entire generative process.  First sample a latent representation:

$$z \sim P(z)$$

Then sample a generated data point $x'$, using the conditional probability $P(x \mid z)$:

$$x' \sim P(x \mid z)$$

These sampling and decoding steps only describe the generation of new data $x'$ from an unspecified generative model.  It doesn't describe the structure of the model we use to approximate $$P(x \mid z)$$.

### Discriminative models

Unlike generative models, **discriminative models learn a conditional probability** $P(x \mid z)$ (the probability of $x$ given $z$).  Discriminative models predict, using observed $z$ to predict $x$.  This is simpler than generative modelling.

A common discriminative computer vision problem is classification, where a high dimensional image is fed through convolutions and outputs a predicted class.

## Appendix Three - Convolution

<center>
	<img src="/assets/world-models/conv.png">
	<figcaption>2D convolution with a single filter</figcaption>
  <div></div>
</center>

Naturally we associate an image as having two dimensions - height & width.  Computers look at images in three dimensions - height, width and colour channels.

The kind of convolution used in neural networks to process images are therefore volume to volume operations - they take a volume as input and produce a volume as output.

At the heart of convolution is the filter (sometimes called a kernel).  These are usually defined as two dimensional, with the third dimension being set to match the number of channels in the image (3 for RGB).

Different kernels are learnt at different layers - shallower layers learn basic features such as edges, with later layers having filters that detect complex compositions of simpler features.

We can think about these kernels operating on tensors of increasing size:
- matrix (3, 3) * kernel (3, 3) -> scalar (1, )
- image (6, 6, 1) * kernel (3, 3, 1) -> image (6, 6, 1)
- image (6, 6, 1) * n kernels (n, 3, 3, 1) -> tensor (6, 6, 1, n)

Important hyperparameters in convolutional neural networks:
- size of filters (typically 3x3)
- number of filters per layer
- padding
- strides

Due to reusing kernels, the convolution neural network is translation invariant, meaning the features can be detected in different parts of the images.  This is ideal in image classification.  Max-pooling (commonly used to downsample the size of the internal representation) also produces translation invariance (along with a loss of infomation).

## Appendix Four - LSTM

*For a deeper look at LSTM's, I cannot reccomend the blog post [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) highly enough.*

The motivation for using an LSTM to approximate the transition dynamics of the environment is that an LSTM is a **recurrent neural network**.  In the `car-racing-v0` environment the data is a sequence of latent representations $z$ of the observation

Recurrent neural networks process data in a sequence:

$$ P(x' | x, h) $$

Where $h$ is the hidden state of the recurrent neural network.



The LSTM was introduced in 1997 by Hochreiter & Schmidhuber.  A key contribution of the LSTM was overcoming the challenge of long term memory with only a single representation of the future.

The LSTM is a recurrent neural network, that makes predictions based on the following:

$$ P(x' | x, h, c) $$

Where $h$ is the hidden state and $c$ is the cell state.  Using two variables for the LSTM's internal representation allows the LSTM to learn both a long and short term representation of the future.

The long term representation is the **cell state** $c$.  The cell state is an information superhighway.

The short term representation is the **hidden state** $h$.

Sigmoid often used as an activation for binary classification.  For LSTMs, we use the sigmoid to control infomation flow.

Tanh is used to generate data.  Neural networks like values in the range -1 to 1, which is exactly how a tanh generates data (with some non-linearity in between).

Infomation is added or removed from both the cell and hidden states using gates.

The gates are functions of the hidden state $h$ and the data $x$.

The gates can be thought of in terms of the methods of a REST API (GET, PUT and DELETE) or the read, update and delete functions in CRUD.

<center>
	<img src="/assets/world-models/lstm.png">
<figcaption> The LSTM - from [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)</figcaption>
</center>

### Forget gate

The first gate is the **forget gate**.  The forget gate works like the `DELETE` request in an REST API.

The forget gate multiplies the cell state by a sigmoid.  A gate value of 0 would mean forget the entire cell state.  A gate value of 0 would mean remember the entire cell state.

The sigmoid used to control the forget gate is a function of the hidden state $h$ and the data $x$.

### Input gate

The second gate is the **input gate**.  The input gate works like a `PUT` or `POST` request.

This gate determines how we will update the cell state from $c$ to $c'$.  The infomation added to the cell state is formed from a sigmoid (that controls which values to update) and a tanh (that generates the new values).

### Output gate

The final gate determines what the LSTM outputs.  This gate works like a `GET` request.

A sigmoid (based on the hidden state $h$) determines which parts of the cell state we will output.  This sigmoid is applied to the updated cell state $c'$, after the updated cell state $c'$ was passed through a tanh layer.

## Appendix Five - Estimating a covariance matrix

Before detailing how CMA-ES estimates it's covariance matrix, we can first review how we would estimate the covariance matrix of a distribution, if we were given only samples from that distribution (i.e. samples from a set of parameters that did well on a control task).

Let's imagine we have a parameter space with two variables, $x$ and $y$, along with samples from the distribution $P(x,y)$.  We can estimate the statistics needed for a covariance matrix as follows.  First the means:

$$ \mu_{x} = \frac{1}{N} \sum_{pop} x $$

$$ \mu_{y} = \frac{1}{N} \sum_{pop} y $$

Then the covariances on the diagonal:

$$ \sigma^{2}_{x} = \frac{1}{N-1} \sum_{pop} \Big( x - \mu_{x} \Big)^{2} $$

$$ \sigma^{2}_{y} = \frac{1}{N-1} \sum_{pop} \Big( y - \mu_{y} \Big)^{2} $$

And the covariance of how our two parameters vary together:

$$ \sigma_{xy} = \frac{1}{N-1} \sum_{pop} \Big( x - \mu_{x} \Big) \Big( y - \mu_{y} \Big) $$

This then gives us our estimated covariance matrix for our samples:

$$\mathbf{C} = \begin{bmatrix}  \sigma^{2}_{x} & \sigma_{xy} \\ \sigma_{yx} &  \sigma^{2}_{y}\end{bmatrix}$$

The method above will approximate the covariance matrix from data (in our case, a population of controller parameters).  We can imagine successively selecting only the best set of parameters, and approximating better covariance matrices.


---

Thanks for reading!
