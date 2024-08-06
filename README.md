# Bayesian Marketing Analysis
## Predicting ROI on a marketing campaign

In this project I derive statistical statements to inform a company's marketing decision. Using bayesian inference, I estimate and report uncertainty on the customer aquisition cost (CAC) based on advertising cost. For example, this can be used to make statements like "There is a 83% chance that a 250 thousand dollar advertising campaign will result in a CAC below the CLV.", where CLV is the customer lifetime value. The main result is visualized in figure 1.

<p align="center">
    <img src="https://github.com/user-attachments/assets/1015f59d-3075-43d9-b18b-2e966b2d950c" alt="Description" width="500"/>
</p>
<p align="center">Figures 1. CAC as a function of ad spending. The shaded region is where CAC>CLV which is not profitable The CAC distribution is profitable below around $100,000 in ad spending due to the fact that some customers will continue being acquired independent of ad spending. The region between $200,000 and $600,000 is where ad spending is most efficient. But with spending more than around $600,000, there will be diminishing returns. Please note that this data is simulated and is not real.</p>

I simulated a company's advertising spending and customer value in order to quantify the impact of a marketing campaign. It is based off ideas and examples given at talk of [Thomas Wieki at the PyData Conference in London, 2022](https://www.youtube.com/watch?v=twpZhNqVExc&t=973s) as well as a Bayesian Astronomy workshop by Jake Vanderplas [here](https://github.com/jakevdp/BayesianAstronomy/tree/3e83b466515df450dc7886109b6f444b55bf0238). Given a logistic growth model, I applied Bayesian inference using Markov chain Monte Carlo (MCMC) algorithms to find the best parameters to fit the data. Bayesian inference is especially useful when reporting both the best estimate as well as the uncertainty on that estimate. See figure 2 to see the simulated data.

<p align="center">
    <img src="https://github.com/user-attachments/assets/15a54763-cb38-4e96-aca4-770123e24fde" alt="Description" width="500"/>
</p>
<p align="center">Figures 2. A mock dataset simulating the number of customers acuquired by a company due to the cost of an advertising campaign. The true distribution is the black line and the mock data points are the blue points with errorbars. </p>

The question that I am trying to answer is: How much marketing money does the company have to spend in order to aquire a single customer. This will be quantified using the customer aquisition cost (CAC) and the customer lifetime value (CLV). So the company would want a CAC $<$ CLV

Using the following Bayesian statistical techniques, the company will be able to make informed decisions from quantitative results. For example, I quantify the probability that an advertising campaign would be profitable. While this project is simplified, it is meant to by a proof of concept for Bayesian inference and could be improved upon if necessary (especially with the addition of more domain knowledge).


### Assumptions

- CLV $= \$ 630$. This is arbitrary and could easily be changed with more domain knowledge
- There exists an underlying model relating marketing spend to customer value that is not highly sensitive to time. I will assume an underlying model that is an S-Shaped Response Curve (Logistic Growth Model).
- There is only one advertising channel (for simplicity)
- The data is simulated based off of numbers from the Thomas Wieki talk.
- errors are homoscedastic. This means the the variances are the same for each data point. 

### The model

We will be using this model to predict the number of new customers resulting from an ad campaign. 

$$ y(x) = A+\frac{L}{1+e^{-k(x-x_0)}} $$

where parameter vector is,

$$ \theta = [A,L,log(k),x_0].$$

$L$ is the curves maximum values, $k$ is the logistic growth rate/steepness, and $x_0$ is the value of the sigmoid midpoint. In this case, $x$ is for the amount of money spent on advertising (in thousands of dollars) and $y$ is the number of new customers aquired.

### Objective
We would like to enable the company to make a quantitatively informed decision using the CAC. Using the model above, the CAC would be calculated using,

$$ CAC = \frac{1000x}{y_{pred}(x)} $$

The plan is to sample the parameters for $y_{pred}(x)$, predict $y_{pred}(x)$, and then used that to find the $CAC$.

#### Bayesian statistics
We are doing Bayesian statistics so the end goal is to find the posterior probability distribution, $$P(\theta \mid D) = \frac{P(D\mid\theta)P(\theta)}{P(D)}.$$ Therefore, we need to know the prior and the likelihood (we can ignore the fully marginalized likelihood). 

First, let's think about the likelihood probability distributions, $P(D|\theta)$. We assumed that the probability for any single data point is a normal distribution about the true value. So,

$$
y_i \sim \mathcal{N}(y_M(x_i;\theta), \sigma)
$$

or, in other words,

$$
P(x_i,y_i\mid\theta) = \frac{1}{\sqrt{2\pi\varepsilon_i^2}} \exp\left(\frac{-\left[y_i - y_M(x_i;\theta)\right]^2}{2\varepsilon_i^2}\right)
$$

This reads as "the probability of data point $(x_i,y_i)$ given model parameter vector, $\theta$". We will assume that $\epsilon_i$ is known and measured. 
The full log-likelihood is,

$$
\log P(D\mid\theta) = -\frac{1}{2}\sum_{i=1}^N\left(\log(2\pi\varepsilon_i^2) + \frac{\left[y_i - y_M(x_i;\theta)\right]^2}{\varepsilon_i^2}\right)
$$



### Model parameter predictions 
These are the predicted model parameters with uncertainties that I report. The parameters are within 1 standard deviation of the true parameters. 
```
Best Estimates
A = 69.93 +/- 52.89
L = 1018.08 +/- 61.98
log10(k) = -1.87 +/- 0.07
x0 = 287.41 +/- 12.48

True values
A = 100.00
L = 1000.00
log10(k) = -1.90
x0 = 300.00
```
In figure 3 I show the result of the MCMC analysis which samples the posterior probability distribution for the parameters. 


<p align="center">
    <img src="https://github.com/user-attachments/assets/89e4c757-ea8b-4b40-baae-1120cc5c0593" alt="Description" width="500"/>
</p>
<p align="center">Figures 3. Corner plot showing marginal probability distributions for each of the model parameters.</p>


