# Photon Machine Learning (Photon-ML)
**Photon Machine Learning (Photon-ML)** is a machine learning library based upon [Apache Spark](http://spark.apache.org/) originally developed by the LinkedIn Machine Learning Algorithms team.

It's designed to be flexible, scalable and efficient, while providing handy analytical abilities to help modelers / data scientists make predictions easily and quickly.

<!-- MarkdownTOC autolink=true bracket=round depth=0 -->

- [Features](#features)
- [Experimental Features](#experimental-features)
  - [GAME - Generalized Additive Mixed Effect Model](#game---generalized-additive-mixed-effect-model)
- [How to build](#how-to-build)
- [Example](#example)
- [How to Contribute](#how-to-contribute)

<!-- /MarkdownTOC -->


## Features
**Photon-ML** currently supports:

1. Generalized Linear Model:
  * Logistic Regression with L1/L2/Elastic Net regularization
  * Poisson Regression with L1/L2/Elastic Net regularization
  * Lasso/Ridge Linear Regression

2. Boxed constraints towards model coefficients, e.g. [0.1 <= wi <= 0.9] where wi is the model coefficient at dimension i

3. Feature scaling and normalization:
  * Zero-mean, unit-variant normalization (with efficient optimization techniques that pertains vector sparsity)
  * Scaling by standard deviation
  * Scaling to range [-1, 1]

4. Offset training: a typical naive way of training multi-layer models. Offset is a special feature with a fixed model coefficient as 1. It's used to insert a smaller model's response into a global model. For example, when doing a typical binary classification problem, we could train a different model against a subset of all the features, and then set that model's response score as an offset of the global model training data. In this way, the global model will only learn against the residuals of the 1st layer model's response while having the benefits of combining the two models together.

5. Feature summarization: **note** it's a direct wrapper of Spark MLLIB Feature summarizer, providing typical metrics (mean, min, max, std, variance and etc.) on a per feature basis

6. Model diagnostic tools: metrics, plots and summarization page for diagnosing model performance. The supported functions include:
  * rocAUC, prAUC, precision, recall, F1, RMSE plotted under different regularization weights
  * Error / Prediction Independence Analysis
  * [Kendall Tau Independence Test](http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kend_tau.htm)
  * Coefficient Importance Analysis
  * Model fitting analysis, and bootstrap analysis
  * [Hosmer-Lemeshow Goodness-of-Fit Test](https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test) for Logistic Regression

## Experimental Features
Photon ML currently contains a number of experimental features that have not been fully tested, and as such should not be used in production. These features center mostly around the **GAME (Generalized Additive Mixed Effect)** modules.

### GAME - Generalized Additive Mixed Effect Model
GAME is a specific expansion of traditional Generalized Linear Models that further provides entity level (e.g., per-user/per-item) or segment level (e.g., per-country/per-category) coefficients, also known as random effects in the statistics literature, in addition to global coefficients. It manages to scale model training up to hundreds of billions of coefficients while still solvable within Spark's framework.

GAME models consist of three components:
  * One fixed effect model:
    * The fixed effect model is effectively a conventional generalized linear model. Its parameters are "global" in the sense that they apply uniformly to all entities.
  * Multiple random effect models:
    * Random effect models consist of "local" parameters – entity-specific coefficients that can be seen as random deviations from the global mean. In other words, they are personalized models.
  * Optionally a matrix factorization model:
    * The matrix factorization model captures interactions between the different random effect models.

The main difference between a GAME model and a conventional linear model is that GAME includes per-entity sub-models for personalization. An entity can be thought of as a logical grouping of data around some object or person, say a member. In GAME, each entity has its own RandomEffect optimization problem, where the training data have been grouped and partitioned by entity.

The relevant code can be found in the following namespaces:
 * com.linkedin.photon.ml.algorithm
 * com.linkedin.photon.ml.data
 * com.linkedin.photon.ml.optimization.game


## How to build
**Note**: Before building, please make sure environment variable ```JAVA_HOME``` is pointed at a Java 8 JDK properly. Photon ML is not compatible with JDK < 1.8.
The below commands are for Linux/Mac users, for Windows, please use ```gradlew.bat``` instead of ```gradlew```.

```bash
# Build binary jars only:
./gradlew assemble

# Build with all tests (unit and integration):
./gradlew clean build

# Run only unit tests:
./gradlew clean test -x integTest

# Run only integration tests:
./gradlew clean integTest -x test

# Check License with Apache Rat
./gradlew rat

# Check scala style
./gradlew scalastyle

# Check everything
./gradlew check
```
## Example
Upload ```example/run_photon_ml.driver.sh``` onto a Spark cluster.
Run command:
```bash
sh run_photon_ml.driver.sh [..options] hdfs_working_dir
```
Get detailed usage help:
```bash
sh run_photon_ml.driver.sh [-h|--help]
```
**Note**: not all configurations are currently exposed as options in the current script, please directly modify the confs if any customization is needed.

## How to Contribute
Contributions are always more than welcome. For starters, you could begin with reporting an issue, participating in discussions, or sending out a pull request addressing one. For major functionality changes, it is highly recommended to exchange thoughts and designs with reviewers beforehand. Well communicated changes will have the highest probability of getting accepted.
