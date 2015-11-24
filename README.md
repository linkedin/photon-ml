## Photon Machine Learning (ML)
**Photon Machine Learning (ML)** is a machine learning library based upon [Apache Spark](http://spark.apache.org/) originally developed at the LinkedIn Offline Relevance Infrastructure team.

### Motivation
Photon ML is a complement to Spark ML/MLLIB. The development was started when there was only an early version of Spark MLLIB. Its original intention was to provide LinkedIn-specific learning functionalities needed by mutiple LinkedIn relevance teams. Over time, Photon ML has diveraged compared to Spark ML/MLLIB. It contains a few similar components but also a few distinct funcationalities. We'd like to open source Photon ML as an opportunity to bind more closely with Spark community, and also figure out ways of contributing those generally useful components back to Spark.

### Functions
**Photon ML** currently supports:

1. Generalized Linear Model:
  * Logistic Regression with L1/L2/Elastic Net regularization;
  * Possion Regression with L1/L2/Elastic Net regularization
  * Lasso/Ridge Linear Regression

2. Boxed constraints towards model coefficients, e.g. [0.1 <= wi <= 0.9] where wi is the model coefficient at dimension i

3. Feature scaling and normalization:
  * Zero-mean, unit-variant normalization (with efficient optimization techniques that pertains vector sparsity)
  * Scaling by standard deviation
  * Scaling to range [-1, 1]

4. Offset training: a typical naive way of training multi-layer models. Offset is a special feature with a fixed model coefficient as 1. It's used to insert a smaller model's response into a global model. For example, when doing a typical binary classification problem, we could train a different model against a subset of all the features, and then set that model's response score as an offset of the global model training data. In this way, the global model will only learn against the residuals of the 1st layer model's response while having the benefits of combining the two models together.

5. Feature summarization: **note** it's a direct wrapper of Spark MLLIB Feature summarizor, providing typical metrics (mean, min, max, std, variance and etc.) on a per feature basis

6. Model disgnostic tools: metrics, plots and summarization page for diagnosing model performance. The supported functions include:
  * rocAUC, prAUC, precision, recall, F1, RMSE plotted under different regularization weights
  * Error / Prediction Independence Analysis
  * [Kendall Tau Independence Test](http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/kend_tau.htm)
  * Coefficient Importance Analysis
  * Model fitting analysis, and boostrap analysis
  * [Hosmer-Lemeshow Goodness-of-Fit Test](https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test) for Logistic Regression

### How to build
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
```
### Example
Upload ```example/run_photon_ml.driver.sh``` onto a Spark cluster.
Run comamnd:
```bash
sh run_photon_ml.driver.sh [..options] hdfs_working_dir
```
Get detailed usage help:
```bash
sh run_photon_ml.driver.sh [-h|--help]
```
**Note**: not all configurations are currently exposed as options in the current script, please directly modify the confs if any customization is needed.