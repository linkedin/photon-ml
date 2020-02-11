# Photon Machine Learning (Photon ML)


[![Build Status](https://travis-ci.org/linkedin/photon-ml.svg?branch=master)](https://travis-ci.org/linkedin/photon-ml)

**Check out our [hands-on tutorial](https://github.com/linkedin/photon-ml/wiki/Photon-ML-Tutorial).**

Photon ML is a machine learning library based on Apache Spark. It was originally developed by the LinkedIn Machine Learning Algorithms Team. Currently, Photon ML supports training different types of [Generalized Linear Models](https://en.wikipedia.org/wiki/Generalized_linear_model)(GLMs) and [Generalized Linear Mixed Models](https://en.wikipedia.org/wiki/Generalized_linear_mixed_model)(GLMMs/GLMix model): logistic, linear, and Poisson.

- [Features](#features)
  - [Generalized Linear Models](#generalized-linear-models)
  - [GAME - Generalized Additive Mixed Effects](#game---generalized-additive-mixed-effects)
  - [Configurable Optimizers](#configurable-optimizers)
  - [Regularization](#regularization)
  - [Feature scaling and normalization](#feature-scaling-and-normalization)
  - [Offset training](#offset-training)
  - [Feature summarization](#feature-summarization)
  - [Model validation](#model-validation)
  - [Warm-start training](#warm-start-training)
  - [Partial re-training](#partial-re-training)
- [Experimental Features](#experimental-features)
  - [Smoothed Hinge Loss Linear SVM](#smoothed-hinge-loss-linear-svm)
  - [Hyperparameter Auto-Tuning](#hyperparameter-auto-tuning)
  - [Regularize by Previous Model During Warm-Start Training](#regularize-by-previous-model-during-warm-start-training)
- [How to Build](#how-to-build)
- [How to Use](#how-to-use)
  - [Drivers](#drivers)
  - [API](#api)
  - [Avro Schemas](#avro-schemas)
  - [What about other formats?](#what-about-other-formats)
  - [Input Data Format](#input-data-format)
  - [Models](#models)
  - [Shaded Jar](#shaded-jar)
  - [Try It Out!](#try-it-out)
    - [Install Spark](#install-spark)
    - [Get and Build the Code](#get-and-build-the-code)
    - [Grab a Dataset](#grab-a-dataset)
    - [Train the Model](#train-the-model)
  - [Running Photon ML on Cluster Mode](#running-photon-ml-on-cluster-mode)
- [Modules and directories](#modules-and-directories)
  - [Source code](#source-code)
  - [Other](#other)
- [IntelliJ IDEA setup](#intellij-idea-setup)
- [How to Contribute](#how-to-contribute)
- [Reference](#reference)

## Features

#### Generalized Linear Models

  * Linear Regression
  * Logistic Regression
  * Poisson Regression
  
#### GAME - Generalized Additive Mixed Effects

The GAME algorithm uses coordinate descent to expand beyond traditional GLMs to further provide per-entity (per-user, per-item, per-country, etc.) coefficients (also known as random effects in statistics literature). It manages to scale model training up to hundreds of billions of coefficients, while remaining solvable within Spark's framework.

For example, a GAME model for movie recommendations can be formulated as (fixed effect model + per-user random effect model + per-movie random effect model + user-movie matrix factorization model). More details on GAME models can be found [here](https://docs.google.com/presentation/d/1vHanpK3KLIVgdDIHYRehUeyb04Hc2AasbBHs4InVPSU).

The type of GAME model currently supported by Photon ML is the GLMM or 'GLMix' model. Many of LinkedIn's core products have adopted GLMix models: jobs search and recommendation, news feed ranking, Ads CTR prediction and "People Also Viewed". More details on GLMix models can be found [here](https://docs.google.com/presentation/d/1tYoelUma9-MMYdteWYS31LqVeoyPEncxJRk-k57gj0A/edit?usp=sharing).

#### Configurable Optimizers
  * [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
  * [TRON](https://www.csie.ntu.edu.tw/~cjlin/papers/logistic.pdf)

#### Regularization
  * L1 (LASSO) regularization
  * L2 (Tikhonov) regularization (only type supported by TRON)
  * Elastic-net regularization

#### Feature scaling and normalization
  * Standardization: Zero-mean, unit-variant normalization
  * Scaling by standard deviation
  * Scaling by maximum magnitude to range [-1, 1]

#### Offset training
A typical naive way of training multi-layer models, it's used to insert another model's response into a global model. For example, when doing a typical binary classification problem, a model could be trained against a subset of all the features. Next, data could be scored with this model and the response scores set as 'offset' values. In this way, future models will learn against the residuals of the 1st layer model's response while having the benefits of combining the two models together.

#### Feature summarization
Provides typical metrics (mean, min, max, std, variance, etc.) on a per feature basis.
  
#### Model validation
Compute evaluation metrics for the trained models over a validation dataset, such as AUC, RMSE, or Precision@k.
  
#### Warm-start training
Load existing models and use their coefficients as a starting point for optimization. When training multiple models in succession, use the coefficients of the previous model.
  
#### Partial re-training
Load existing models, but lock their coefficients. Allows efficient re-training of portions of a GAME model.

## Experimental Features
Photon ML currently contains a number of experimental features that have not been fully tested.

#### Smoothed Hinge Loss Linear SVM
In addition to the Generalized Linear Models described above, Photon-ML also supports an optimizer-friendly approximation for linear SVMs as described [here](http://qwone.com/~jason/writing/smoothHinge.pdf) by Jason D. M. Rennie.

#### Hyperparameter Auto-Tuning
Automatically explore the hyperparameter space for your GAME model. Two types of search exist:
- Random search: Use Sobol sequences to randomly, but evenly, explore the hyperparameter space
- Bayesian search: Use a Gaussian process to perform a directed search throughout the hyperparameter space

#### Regularize by Previous Model During Warm-Start Training
Use the means and variances of an existing model to help approximate good coefficients when doing warm-start training. Allows multiple rounds of warm-start training on small datasets to achieve parity with cold-start training on a single large dataset.

## How to Build
**Note**: Before building, please make sure environment variable ```JAVA_HOME``` is pointed at a Java 8 JDK property. Photon ML is not compatible with JDK < 1.8.
The below commands are for Linux/Mac users, for Windows, please use ```gradlew.bat``` instead of ```gradlew```.

```bash
# Build binary jars only:
./gradlew assemble

# Build with all tests (unit and integration):
./gradlew clean build

# Build with only unit tests:
./gradlew clean build -x integTest

# Build with only integration tests:
./gradlew clean build -x test

# Build with no tests:
./gradlew clean build -x test -x integTest

# Run unit tests:
./gradlew test

# Run integration tests:
./gradlew integTest

# Check License with Apache Rat
./gradlew rat

# Check scala style
./gradlew scalastyle

# Check everything
./gradlew check
```
## How to Use

### Drivers
To use Photon ML from the command line, 3 default drivers exist: the Legacy Photon driver for GLM training, the GAME training driver, and the GAME scoring driver. Each of these have their own input parameters. We recommend using the GAME drivers, as a GLM is a special case of GAME model. The Legacy Photon driver has not been developed for some time and is deprecated.

### API
Photon ML can be imported just like Spark ML, and the API layer used directly. Where possible, we have tried to make the interfaces identical to those of Spark ML. See the driver source code for examples of how to use the Photon ML API.

### Avro Schemas
The currently available drivers read/write data in [Apache Avro](https://avro.apache.org/) format. The detailed schemas are declared at [photon-avro-schemas](https://github.com/linkedin/photon-ml/tree/master/photon-avro-schemas/src/main/avro) module.

### What about other formats?
LinkedIn uses primarily Avro formatted data. While Avro does provide a unified and rigorous way of managing all critical data representations, we think it is also important to allow other data formats to make Photon ML more flexible. Contributions of DataReaders for other formats to Photon ML are welcome and encouraged.

### Input Data Format

Photon ML reserves the following field names in the Avro input data:
1. **response**: `double` (required)
    - The response/label for the event
2. **weight**: `double` (optional)
    - The relative weight of a particular sample compared to other samples
    - Default = 1.0
3. **offset**: `double` (optional)
    - The residual score computed by some other model
    - Default = 0.0
    - Computed scores always take the form `(x * B) + offset`, where `x` is the feature vector and `B` is the coefficient vector
4. **uid**: `string`, `int`, or `long` (optional)
    - A unique ID for the sample
5. **metadataMap**: `map: [string]` (optional)
    - A map of non-feature metadata for the sample
6. **features**: `array: [FeatureAvro]` (required by Legacy Photon driver)
    - An array of features to use for training/scoring

All of these default names can be overwritten using the GAME drivers. However, they are reserved and cannot be used for purposes other than their default usage (e.g. cannot specify `response` as your weight column).

Additional fields may exist in the record, and in fact are necessary for certain features (e.g. must have ID fields to group data by for random effect models or certain validation metrics).

Features loaded through the existing drivers are expected to follow the LinkedIn naming convention. Each feature must be an Avro record with the following fields:
1. **name**: `string`
  - The feature name/category
2. **term**: `string`
  - The feature sub-category
3. **value**: `double`
  - The feature value
  
To demonstrate the difference between `name` and `term`, consider the following categorical features:
```
  name = "age"
  term = "0-10"
  value = 1.0
  
  name = "age"
  term = "11-20"
  value = 0.0
  
  ...
```

### Models

Legacy Photon outputs model coefficients directly to text:
```bash
# For each line in the text file:
[feature_string]\t[feature_id]\t[coefficient_value]\t[regularization_weight]
```

GAME models are output using the [BayesianLinearModelAvro](https://github.com/linkedin/photon-ml/blob/master/photon-avro-schemas/src/main/avro/BayesianLinearModelAvro.avsc) Avro schema.

### Shaded Jar
[photon-all](https://github.com/linkedin/photon-ml/tree/master/photon-all) module releases a shaded jar containing all the required runtime dependencies of Photon ML, other than Spark and Hadoop. Shading is a robust way of creating fat/uber jars. It does not only package all dependencies into one single place, but also smartly renames a few selected class packages to avoid dependency conflicts. Although ```photon-all.jar``` is not a necessity, and it is fine for users to provide their own copies of dependences, it is highly recommended to be used in cluster environment where complex dependency conflicts could happen between system and user jars. (See [Gradle Shadow Plugin](https://github.com/johnrengelman/shadow) for more about shading).

Below is a command to build the photon-all jar:
```bash
./gradlew :photon-all:assemble
```

### Try It Out!

The easiest way to get started with Photon ML is to try the tutorial we created to demonstrate how GLMix models can be applied to build a personalized recommendation system. You can view the instructions on the wiki [here](https://github.com/linkedin/photon-ml/wiki/Photon-ML-Tutorial).

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
./gradlew build -x test -x integTest
```

#### Grab a Dataset

For this example, we'll use the "a1a" dataset, acquired from [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html). Currently the Photon ML dataset converter supports only the LibSVM format.

```
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a
curl -O https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t
```

Convert the data to the Avro format that the Photon ML drivers use.

```
mkdir -p a1a/train
mkdir -p a1a/test
pip install avro
python dev-scripts/libsvm_text_to_trainingexample_avro.py a1a dev-scripts/TrainingExample.avsc a1a/train/a1a.avro
python dev-scripts/libsvm_text_to_trainingexample_avro.py a1a.t dev-scripts/TrainingExample.avsc a1a/test/a1a.t.avro
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
  --training-data-directory "./a1a/train/" \
  --validating-data-directory "./a1a/test/" \
  --format "TRAINING_EXAMPLE" \
  --output-directory "out" \
  --task "LOGISTIC_REGRESSION" \
  --num-iterations 50 \
  --regularization-weights "0.1,1,10,100" \
  --job-name "demo_photon_ml_logistic_regression"
```

Alternatively, to run the exact same training using the GAME training driver, use the following command:

```
spark-submit \
  --class com.linkedin.photon.ml.cli.game.GameTrainingDriver \
  --master local[*] \
  --num-executors 4 \
  --driver-memory 1G \
  --executor-memory 1G \
  "./build/photon-all_2.10/libs/photon-all_2.10-1.0.0.jar" \
  --input-data-directories "./a1a/train/" \
  --validation-data-directories "./a1a/test/" \
  --root-output-directory "out" \
  --feature-shard-configurations "name=globalShard,feature.bags=features" \
  --coordinate-configurations "name=global,feature.shard=globalShard,min.partitions=4,optimizer=LBFGS,tolerance=1.0E-6,max.iter=50,regularization=L2,reg.weights=0.1|1|10|100" \
  --coordinate-update-sequence "global" \
  --coordinate-descent-iterations 1 \
  --training-task "LOGISTIC_REGRESSION"
```

When this command finishes, you should have a new folder named "out" containing the trained model.

### Running Photon ML on Cluster Mode

In general, running Photon ML is no different from running other general Spark applications. As a result, using the ```spark-submit``` script in Spark’s ```bin``` directory we can run Photon ML on [different cluster modes](http://spark.apache.org/docs/latest/cluster-overview.html):
* [Spark Standalone Mode](http://spark.apache.org/docs/latest/spark-standalone.html)
* [Mesos](http://mesos.apache.org/)
* [YARN](http://hadoop.apache.org/docs/stable/hadoop-yarn/hadoop-yarn-site/YARN.html)

Below is a template for running a logistic regression training job with minimal setup on YARN. For running Photon ML using other cluster modes, the relevant arguments to ```spark-submit``` can be modified as detailed in [http://spark.apache.org/docs/latest/submitting-applications.html](http://spark.apache.org/docs/latest/submitting-applications.html).

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

TODO: This example should be updated to use the GAME training driver instead.
There is also a more complex script demonstrating advanced options and customizations of using Photon ML at [example/run_photon_ml.driver.sh](https://github.com/linkedin/photon-ml/blob/master/examples/run_photon_ml_driver.sh).

Detailed usages are described via command:
```bash
./run_photon_ml.driver.sh [-h|--help]
```
**Note**: Not all configurations are currently exposed as options in the current script. Please directly modify the configurations if any customization is needed.

## Modules and directories
### Source code
- TODO: Photon ML modules are in need of a refactor. Once this is complete, this section will be updated.

### Other
- `build-scripts` contains scripts for Gradle tasks
- `buildSrc` contains Gradle plugin source code
- `dev-scripts` contains various scripts which may be useful for development
- `examples` contains a script which demonstrates how to run Photon ML from the command line
- `gradle` contains the Gradle wrapper jar
- `travis` contains scripts for controlling Travis CI test execution

## IntelliJ IDEA setup
When set up correctly, all the tests (unit and integration) can be run from IntelliJ IDEA, which is very helpful for development (IntelliJ IDEA's debugger can be used with all the tests). 
- Run `./gradlew idea`
- Open project as "New/Project from Existing Source", choose Gradle project, and set Gradle to use the local wrapper.

## How to Contribute
We welcome contributions. The following are good ways to get started: reporting an issue, fixing an existing issue, or participating in a discussion. For major functionality changes, it is highly recommended to exchange thoughts and designs with reviewers beforehand. Well communicated changes will have the highest probability of getting accepted.

## Reference
- XianXing Zhang, Yitong Zhou, Yiming Ma, Bee-Chung Chen, Liang Zhang and Deepak Agarwal. [GLMix: Generalized Linear Mixed Models For Large-Scale Response Prediction](http://www.kdd.org/kdd2016/papers/files/adf0562-zhangA.pdf). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
