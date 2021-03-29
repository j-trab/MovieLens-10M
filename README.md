# MovieLens-10M
Capstone Project using Movielens 10 millions entries dataset. 

In this Capstone project we created  a movie __recommendation system__ working with the __MovieLens 10M__, the largest version of the famously publicly available MovieLens, developed in 1997 by the _GroupLens Research Lab_. 

The main - and defining - challenges of the project resided in dealing with such a large dataset on a commercial laptop.

Unable to use most of the algorithms available in __R__ via the Caret package, we proposed a “Tidyverse” approach, producing a series of linear models which we adjusted by progressively adding a set of biases / effects. 

We finally tuned a __Matrix Factorization__ model by means of the extremely agile Reticulate package. 
