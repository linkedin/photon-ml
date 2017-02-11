# Photon Machine Learning (Photon ML)

[![Build Status](https://travis-ci.org/linkedin/photon-ml.svg?branch=master)](https://travis-ci.org/linkedin/photon-ml)

**New**: check out our [hands-on tutorial](https://github.com/linkedin/photon-ml/wiki/Photon-ML-Tutorial).

**Photon Machine Learning (Photon ML)** is a machine learning library based upon [Apache Spark](http://spark.apache.org/) originally developed by the LinkedIn Machine Learning Algorithms team.

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
  - [Try It Out!](#try-it-out)
    - [Install Spark](#install-spark)
    - [Get and Build the Code](#get-and-build-the-code)
    - [Grab a Dataset](#grab-a-dataset)
    - [Train the Model](#train-the-model)
  - [Running Photon ML on Cluster Mode](#running-photon-ml-on-cluster-mode)
- [How to Contribute](#how-to-contribute)
- [Reference](#reference)

<!-- /MarkdownTOC -->


## Features
**Photon ML** currently supports:

1. Generalized Linear Models:
  * Linear Regression
  * Logistic Regression
  * Poisson Regression

2. Regularization:
  * The LBFGS optimizer supports L1, L2, and Elastic Net regularization
  * The TRON optimizer supports L2 regularization

3. Boxed constraints towards model coefficients, e.g. [0.1 <= wi <= 0.9] where wi is the model coefficient at dimension i

4. Feature scaling and normalization:
  * Zero-mean, unit-variant normalization (with efficient optimization techniques that pertains vector sparsity)
  * Scaling by standard deviation
  * Scaling to range [-1, 1]

5. Offset training: a typical naive way of training multi-layer models. Offset is a special feature with a fixed model coefficient as 1. It's used to insert a smaller model's response into a global model. For example, when doing a typical binary classification problem, we could train a different model against a subset of all the features, and then set that model's response score as an offset of the global model training data. In this way, the global model will only learn against the residuals of the 1st layer model's response while having the benefits of combining the two models together.

6. Feature summarization: **note** it's a direct wrapper of Spark MLLIB Feature summarizer, providing typical metrics (mean, min, max, std, variance and etc.) on a per feature basis

7. Model diagnostic tools: metrics, plots and summarization page for diagnosing model performance. The supported functions include:
  * rocAUC, prAUC, precision, recall, F1, RMSE plotted under different regularization weights
  * Error / Prediction Independence Analysis
  * [Kendall Tau Independence Test](http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kend_tau.htm)
  * Coefficient Importance Analysis
  * Model fitting analysis, and bootstrap analysis
  * [Hosmer-Lemeshow Goodness-of-Fit Test](https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test) for Logistic Regression

## Experimental Features
Photon ML currently contains a number of experimental features that have not been fully tested, and as such should not be used in production. These features center mostly around the **GAME (Generalized Additive Mixed Effect)** modules.

#### Smoothed Hinge Loss Linear SVM
In addition to the Generalized Linear Models described above, Photon-ML also supports an optimizer-friendly approximation for linear SVMs as described [here](http://qwone.com/~jason/writing/smoothHinge.pdf) by Jason D. M. Rennie.

### GAME - Generalized Additive Mixed Effect Model
GAME is a specific expansion of traditional Generalized Linear Models that further provides entity level (e.g., per-user/per-item) or segment level (e.g., per-country/per-category) coefficients, also known as random effects in the statistics literature, in addition to global coefficients. It manages to scale model training up to hundreds of billions of coefficients while still solvable within Spark's framework.

Currently Photon-ML supports GAME models composed of the following three types of components:
  * Fixed effect model:
    * Each fixed effect model is effectively a conventional generalized linear model. Its parameters are "global" in the sense that they apply uniformly to all entities.
  * Random effect model:
    * Each random effect model consists of "local" parameters – entity-specific coefficients that can be seen as random deviations from the global mean. For example, a per-user random effect models each user's behavior through user-specific coefficients.
  * Matrix factorization model:
    * Conventional matrix factorization model that captures interactions between two types of random effects (e.g., user and item) in the latent space.

For example, a GAME model for movie recommendation can be formulated as fixed effect model + per-user random effect model + per-movie random effect model + user-movie matrix factorization model. More details on GAME models can be found [here](https://docs.google.com/presentation/d/1vHanpK3KLIVgdDIHYRehUeyb04Hc2AasbBHs4InVPSU).

One exemplary type of GAME model supported in Photon-ML is [GLMix](https://github.com/linkedin/photon-ml#reference), which has been adopted to serve the machine learning components of LinkedIn's core products, including: jobs search and recommendation, news feed ranking, Ads CTR prediction and "People Also Viewed". More details on GLMix models can be found [here](https://docs.google.com/presentation/d/1tYoelUma9-MMYdteWYS31LqVeoyPEncxJRk-k57gj0A/edit?usp=sharing).

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

### Try It Out!

The easiest way to get started with Photon ML is to try the tutorial we created to demonstrate how generalized linear mixed-effect models can be applied to build a personalized recommendation system. You can view the instructions on the wiki [here](https://github.com/linkedin/photon-ml/wiki/Photon-ML-Tutorial).

Alternatively, you can follow these steps to try Photon ML on your machine.

#### Install Spark

This step is platform-dependent. On OS X, you can install Spark with [Homebrew](http://brew.sh/) using the following command:

```
brew install apache-spark
```

For more information, see the [Spark docs](http://spark.apache.org/docs/latest/index.html).

#### Get and Build the Code

```
git clone git@github.com:linkedin/photon-ml.git
cd photon-ml
./gradlew build -x integTest
```

#### Grab a Dataset

For this example, we'll use the "a1a" dataset, acquired from [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). Currently the Photon ML dataset converter supports only the LibSVM format.

```
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t
```

Convert the data to the Avro format that Photon ML understands.

```
pip install avro
python dev-scripts/libsvm_text_to_trainingexample_avro.py a1a dev-scripts/TrainingExample.avsc a1a.avro
python dev-scripts/libsvm_text_to_trainingexample_avro.py a1a.t dev-scripts/TrainingExample.avsc a1a.t.avro
```

The first command might be different, depending on the configuration of your system. If it fails, try your platform's standard approach for installing a Python library.

#### Train the Model

Now we're ready to train the model with Photon ML on your local dev box. Run the following command from the "photon-ml" directory:

```
spark-submit \
  --class com.linkedin.photon.ml.Driver \
  --master local[*] \
  --num-executors 4 \
  --driver-memory 1G \
  --executor-memory 1G \
  "./build/photon-all_2.10/libs/photon-all_2.10-1.0.0.jar" \
  --training-data-directory "./a1a.avro" \
  --validating-data-directory "./a1a.t.avro" \
  --format "TRAINING_EXAMPLE" \
  --output-directory "out" \
  --task "LOGISTIC_REGRESSION" \
  --num-iterations 50 \
  --regularization-weights "0.1,1,10,100" \
  --job-name "demo_photon_ml_logistic_regression"
```

When this command finishes, you should have a new folder named "out" containing the model and a diagnostic report. On OS X:

```
open out/model-diagnostic.html
```

### Running Photon ML on Cluster Mode
In general, running Photon ML is no different from running other general Spark applications. As a result, using the
```spark-submit``` script in Spark’s ```bin``` directory we can run
Photon ML on [different cluster modes](http://spark.apache.org/docs/latest/cluster-overview.html) (e.g.,
[Spark Standalone Mode](http://spark.apache.org/docs/latest/spark-standalone.html),
[Mesos](http://mesos.apache.org/), [YARN](http://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/YARN.html)).

In the following we provide a simple demonstration of running a Logistic Regression
training and validation job with minimal setups on YARN. For running Photon ML on other cluster modes the relevant
arguments can be modified accordingly with the ```spark-submit``` script as detailed in
[http://spark.apache.org/docs/latest/submitting-applications.html](http://spark.apache.org/docs/latest/submitting-applications.html).


```bash
spark-submit \
  --class com.linkedin.photon.ml.Driver \
  --master yarn \
  --deploy-mode cluster \
  --num-executors $NUM_EXECUTORS \
  --driver-memory $DRIVER_MEMORY \
  --executor-memory $EXECUTOR_MEMORY \
  "./build/photon-all_2.10/libs/photon-all_2.10-1.0.0.jar" \
  --training-data-directory "path/to/training/data" \
  --validating-data-directory "path/to/validating/data" \
  --output-directory "path/to/output/dir" \
  --task "LOGISTIC_REGRESSION" \
  --num-iterations 50 \
  --regularization-weights "0.1,1,10" \
  --job-name "demo_photon_ml_logistic_regression"
```

There is also a more complex script demonstrating advanced options and customizations of using Photon ML at
[example/run_photon_ml.driver.sh](https://github.com/linkedin/photon-ml/blob/master/examples/run_photon_ml_driver.sh).

Detailed usages are described via command:
```bash
./run_photon_ml.driver.sh [-h|--help]
```
**Note**: not all configurations are currently exposed as options in the current script, please directly modify the confs if any customization is needed.

## Modules and directories
### Source code
- photon-all contains only a build.gradle, to build a shaded jar containing all of photon-ml.
- photon-avro-schemas contains all the Avro schemas used by photon-ml (e.g. when reading in training data).
- photon-ml contains the code for photon-ml itself
- photon-test-utils contains utility classes and functions used in tests and integration tests.

### Other
- build-scripts contains scripts used during the build of photon-ml.
- buildSrc contains Gradle plugins source code. Those plugins are used to build photon-ml.
- dev-scripts contains various scripts useful to developers of photon-ml.
- examples contains a script that demonstrates how to run photon ml from command line.
- gradle contains the gradle wrapper jar.

## IntelliJ IDEA setup
When set up correctly, all the tests (unit and integration) can be run from IntelliJ IDEA, which is very helpful for 
development (IntelliJ IDEA's debugger can be used with all the tests).
- Run ./gradlew first on the command line (some classes need to be generated once).
- Open project in IDEA with "Import Project" and import as a Gradle project. 

## How to Contribute
We welcome contributions. A good way to get started would be to begin with reporting an issue, participating in discussions, or sending out a pull request addressing an issue. For major functionality changes, it is highly recommended to exchange thoughts and designs with reviewers beforehand. Well communicated changes will have the highest probability of getting accepted.

## Reference
- XianXing Zhang, Yitong Zhou, Yiming Ma, Bee-Chung Chen, Liang Zhang and Deepak Agarwal. [GLMix: Generalized Linear Mixed Models For Large-Scale Response Prediction](http://www.kdd.org/kdd2016/papers/files/adf0562-zhangA.pdf). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
