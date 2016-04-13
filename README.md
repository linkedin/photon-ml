# Photon Machine Learning (Photon-ML)

[![Build Status](https://travis-ci.org/linkedin/photon-ml.svg?branch=master)](https://travis-ci.org/linkedin/photon-ml)
[![Coverage Status](https://coveralls.io/repos/github/linkedin/photon-ml/badge.svg?branch=master)](https://coveralls.io/github/linkedin/photon-ml?branch=master)

**Photon Machine Learning (Photon-ML)** is a machine learning library based upon [Apache Spark](http://spark.apache.org/) originally developed by the LinkedIn Machine Learning Algorithms team.

It's designed to be flexible, scalable and efficient, while providing handy analytical abilities to help modelers / data scientists make predictions easily and quickly.

<!-- MarkdownTOC autolink=true bracket=round depth=0 -->

- [Features](#features)
- [Experimental Features](#experimental-features)
  - [GAME - Generalized Additive Mixed Effect Model](#game---generalized-additive-mixed-effect-model)
- [How to Build](#how-to-build)
- [How to Use](#how-to-use)
  - [Avro Schemas](#avro-schemas)
    - [*What about other formats?*](#what-about-other-formats)
  - [Input Data Format](#input-data-format)
  - [Models](#models)
  - [Shaded Jar](#shaded-jar)
  - [Example Scripts](#example-scripts)
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


## How to Build
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
## How to Use
### Avro Schemas
Currently all serialized Photon ML data other than models is stored in [Apache Avro](https://avro.apache.org/) format. The detailed schemas are declared at [photon-avro-schemas](https://github.com/linkedin/photon-ml/tree/master/photon-avro-schemas/src/main/avro) module. And such Avro schemas are compiled for Avro 1.7.5+, be mindful that Avro 1.4 or any version < 1.7.5 might not be entirely compatible with the in-memory data record representations.

#### *What about other formats?*
While Avro does provide a unified and rigorous way of managing all critical data representations, we think it is also important to allow other data formats to make Photon ML more flexible. There are future plans to loosen such data format requirements. For the current version, Photon ML is assuming users will properly prepare input data in Avro formats.

### Input Data Format
Currently Photon ML uses Avro schema [TrainingExampleAvro](https://github.com/linkedin/photon-ml/blob/master/photon-avro-schemas/src/main/avro/TrainingExampleAvro.avsc) as the official input format. However, any Avro Generic Datum satisfying the following schema requirements are actually acceptable:
- The Avro record *must* have two fields:
  1. **label**: ```double```
  2. **features**: ```array: [{name:string, term:string, value:float}]```
- All the other fields are *optional*.
- We define a feature string to be represented as ```name + INTERNAL_DELIM + term```. For example, if we have a feature called ```age=[10,20]```, i.e. age between 10 years old to 20 years old. The feature can be represented as:
```
  name="age"
  term="[10,20]"
  value=1.0
```
- **term** field is optional, if you don't want to use two fields to represent one feature, feel free to set it as empty string.
- Train and validation datasets should be in the exact same format. e.g. it is probably a bad idea if train dataset contains the **offset** field while validation data does not.
- Intercept is not required in the training data, it can be optionally be appended via an option see [example scripts](#example-scripts) for more details;
- **weight** is an optional field that specifies the weight of the observation. Default is 1.0. If you feel some observation is stronger than the others, feel free to use this field, say making the weak ones 0.5.
- **offset** is an optional field. Default is 0.0. When it is non-zero, the model will learn coefficients beta by ```x'beta+offset``` instead of ```x'beta```.

Below is a sample of training/test data:
```
Avro Record 1:
  {
    "label" : 0,
    "features" : [
    {
      "name" : "7",
      "term" : "33",
      "value" : 1.0
    }, {
      "name" : "8",
      "term" : "151",
      "value" : 1.0
    }, {
      "name" : "3",
      "term" : "0",
      "value" : 1.0
    }, {
      "name" : "12",
      "term" : "132",
      "value" : 1.0
    }
   ],
    "weight" : 1.0,
    "offset" : 0.0,
    "foo" : "whatever"
 }
```

### Models
The trained model coefficients are output as text directly. It is intended as such for easy consumption. The current output format for Generalized Linear Models are as such:
```bash
# For each line in the text file:
[feature_string]\t[feature_id]\t[coefficient_value]\t[regularization_weight]
```
Future improvements are planned to make such model formats more flexible.

### Shaded Jar
[photon-all](https://github.com/linkedin/photon-ml/tree/master/photon-all) module releases a shaded jar containing all the required runtime dependencies of Photon ML other than Spark. Shading is a robust way of creating fat/uber jars. It does not only package all dependencies into one single place, but also smartly renames a few selected class packages to avoid dependency conflicts. Although ```photon-all.jar``` is not a necessity, and it is fine for users to provide their own copies of dependences, it is highly recommended to be used in cluster environment where complex dependency conflicts could happen between system and user jars. (See [Gradle Shadow Plugin](https://github.com/johnrengelman/shadow) for more about shading).

Below is a command to build the photon-all jar:
```bash
# Change 2.10 to 2.11 for Scala 2.11
./gradlew :photon-all_2.10:assemble
```

### Example Scripts
The below script is a simple demonstration of running a Logistic Regression training and validation job with minimal setups:
```bash
spark-submit \
  --class com.linkedin.photon.ml.Driver \
  --master yarn-cluster \
  --num-executors 50 \
  --driver-memory 10G \
  --executor-memory $memory \
  "photon-all_2.10-1.0.0.jar" \
  --training-data-directory "path/to/training/data" \
  --validating-data-directory "path/to/validating/data" \
  --output-directory "path/to/output/dir" \
  --task "LOGISTIC_REGRESSION" \
  --num-iterations 50 \
  --regularization-weights "0.1,1,10" \
  --job-name "demo_photon_ml_logistic_regression"
```

There is also a more complex script demonstrating advanced options and customizations of using Photon ML at  [example/run_photon_ml.driver.sh](https://github.com/linkedin/photon-ml/blob/master/examples/run_photon_ml_driver.sh).

Detailed usages are described via command:
```bash
./run_photon_ml.driver.sh [-h|--help]
```
**Note**: not all configurations are currently exposed as options in the current script, please directly modify the confs if any customization is needed.

## How to Contribute
Contributions are always more than welcome. For starters, you could begin with reporting an issue, participating in discussions, or sending out a pull request addressing one. For major functionality changes, it is highly recommended to exchange thoughts and designs with reviewers beforehand. Well communicated changes will have the highest probability of getting accepted.
