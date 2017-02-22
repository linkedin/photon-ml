# Random projections and scaling

Random projections can be helpful to reduce the number of degrees of freedom in the model when the
training data is insufficient and thus avoid over-fitting. For pure scaling, Game's strategy is to optimize
"by parts", by optimizing each random effect in parallel, so there is no need to use projections to tackle
scaling when there is enough training data.

The index map based projection (where all features are assigned integer ids and which does not reduce
the size of the feature space) is effective enough to scale up the model training by exploiting
the feature sparsity of the random effect model, so random projections do not appear to be critical
up to the following data size: ~ 100 millions of random effects, and ~1000 unique features per random effect.

As a result, our recommendation is to use the index map based projection (not random projections)
as default for model training, but possibly to try random projections if the training set is very limited.

## References

- http://scikit-learn.org/stable/modules/random_projection.html
- https://en.wikipedia.org/wiki/Random_projection

