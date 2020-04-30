# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [19.0.1] - 2020-04-29
### Added
- Added publish to bintray functionality
### Changed
- Refactored feature normalization
- Refactored optimization states tracker
- Remove mean and variance type consistency check
### Fixed
- N/A

## [18.0.0] - 2020-03-06
### Added
- Added incremental training functionality
### Changed
- Refactored RDD join
- Refactored aggregators
### Fixed
- Fixed a bug to calculate RMSE off by sqrt(2)
- Fixed a bug which would keep L2 component of gradient and Hessian for intercept term
- Fixed a bug which would drop data which does not have initial random effect models
- Fixed a hyperparameter tuning dimension mismatch bug
- Fixed a bug which would filter out prior model coefficients for inactive features

## [17.0.0] - 2019-10-09
### Changed
- The intercept is no longer included in L2 regularization
### Deprecated
- Deprecated option to disable optimizer state tracking (it is now always enabled)
### Removed
- Removed unused/pointless modules
### Fixed
- Fixed bug which would treat data filtered by the 'active.data.lower.bound' as passive data

## [16.1.0] - 2019-06-17
### Added
- Added command line input for controlling minimum number of partitions for input data RDD to NameAndTermFeatureBagsDriver
### Fixed
- Fixed inefficient join in RandomEffectDataset construction which was causing excessive executor memory usage

## [16.0.0] - 2019-06-10
### Added
- Added a numerically stable function for computing Pearson correlation score to LocalDataset
### Changed
- Changed NameAndTermFeatureBagsDriver to read/write feature sets from/to text files using RDD functions
- Linear subspace projection is now the only kind of projection on random effect features and is now permanently enabled
### Fixed
- Warm start is now disabled for hyperparameter auto-tuning; this allows up to 10x the amount of iterations due to reduced size of RDD lineage

## [15.0.0] - 2019-05-15
### Changed
- Renamed 'updateModel' function in Coordinate class to 'trainModel'
### Removed
- Removed photon-diagnostics module
- Removed ModelTracker class
- Removed 'initializeModel' function from Coordinate class
### Fixed
- Fixed links in README
- Fixed bug which prevented training intercept-only models

## [14.0.0] - 2019-03-15
### Changed
- Improve performance of RandomEffectDataset construction
- Loosened permissions for Evaluator.evaluate function
### Removed
- Removed passive data lower bound as a tuning parameter
- Removed RDDLike.materializeOnce function
- Removed Coordinate.computeRegularizationTermValue function
### Fixed
- Fixed bug affecting reading Map types from AVRO using the AvroDataReader

## [13.0.0] - 2019-02-12
### Added
- Added option to compute approximate coefficient variances using the Hessian diagonal instead of the full Hessian matrix
### Changed
- Updated README
- Changed 'dataset' terminology to be consistent across library
- Refactored evaluation metric classes: created new EvaluationSuite class which manages a Set of Evaluators, the RDD they use, and the 'primary' Evaluator
- Replaced most references to 'mllib.linalg.Vectors' with 'ml.linalg.Vectors'
- Changed DataValidators to accept any numeric value, not just Double
- Map object loaded through AvroDataReader now drop any values that are non-String, non-numeric, or non-Boolean
### Removed
- Removed dependencies on photon-diagnostics module
### Fixed
- Fixed bug in MinHeapWithFixedCapacity which would clear the wrong heap
- Fixed various bugs in DataValidators
- Fixed bug which would ignore the L2 component of elastic net regularization

## [12.0.0] - 2018-09-11
### Added
### Changed
- BasicStatisticalSummary now stores index of intercept, if present
- Intercept column is ignored when writing BasicStatisticalSummary to disk
- Strengthened relationship between shifts and intercept in NormalizationContext
- Renamed DataSet and derived classes to Dataset and variants thereof
- Upgrade to Spark 2.3.0
- Disable Scala 2.10 crossbuild
### Removed
### Fixed
- Fixed inefficiency in NormalizationContext when neither shifts nor factors present

## [11.0.0] - 2018-08-27
### Added
- Added helper function to shrink search range when selecting hyperparamater vectors during Guassian auto-tuning
- Added hyperparameter tuner interface, for removing the direct dependency from Photon-ML on the hyperparameter tuning code
- Added Apache license check to Travis CI tests
- Added PhotonParams class, which contains shared parameter-management helper functions
### Changed
- Decouple hyperparameter tuning from the Photon-ML Evaluator family
- RandomEffectDataSetPartitioner now uses a single, lazy-evaluated backup partitioner instead of one generated per-instance requiring the backup partitioner
- Improved efficiency of MinHeapWithFixedCapacity merging algorithm
- Changed LBFGSB unit tests to be deterministic
- RandomEffectDataSetPartitioner now takes the active data upper bound into account, so that data is better distributed across partitions
### Removed
- Removed the 'NONE' training task
- Replaced Photon StorageLevel with Spark StorageLevel
- Removed unused feature filtering algorithms from LocalDataSet
### Fixed
- Fixed bug in using ExpectedImprovement for minimization problems

## [10.2.0] - 2018-07-31
### Added
- Enabled box constraint option for TRON in GLM driver
- Added evaluator for area under precision-recall curve (AUPR)
### Fixed
- Fixed bug in expected improvement

## [10.1.0] - 2018-07-17
### Added
- Added elastic net alpha to hyper-parameter auto-tuning vector
- Enabled Scala cross-building for versions 2.10.6 and 2.11.8
### Fixed
- Fixed bug which would unpersist models too early

## [10.0.0] - 2018-06-21
### Added
- Added `AvroDataWriter`; complement to the `AvroDataReader`, it writes a `DataFrame` to HDFS in AVRO format with the minimum columns expected by Photon ML for a fixed-effect-only model
- Added hyper-parameter rescaling in the covariance kernel
- Added parameter to control timezone for "days ago" date ranges
- Added full warm-start behaviour: an existing initial model can be loaded and its coefficients will be used as the starting point for optimization
- Hyper-parameter serialization !!!
- Added elastic net regularization alpha value to the hyper-parameter auto-tuning vector
### Changed
- Old warm-start behaviour is now enabled by default
- Ranges for hyper-parameter tuning are now unique to each coordinate and set in the coordinate configurations
### Removed
- Warm-start parameter removed from `GameTrainingDriver` and `GameEstimator`
- Hyper-parameter tuning range parameter removed
### Fixed
- Fixed inefficient assignment of unique IDs to samples
- Fixed partitioning bug in `AvroDataReader`
- Improved resilience of parsing code for `DoubleRange`
- Fixed performance issue in default index map generation code
- Fixed bug which allowed a single `DataFrame` column to be read for multiple fields (response, weight, offset, random effect type, etc.)
- Fixed bug which would unpersist models too early during warm-start training

## [9.0.0] - 2018-04-17
### Added
- Custom Kryo registrator for the shaded Avro `GenericData.Array` class
- Test source jars for `photon-lib` and `photon-api`
- Hyper-parameter tuning: added amplitude and noise to Bayesian hyper-parameter tuning kernel
- Hyper-parameter tuning: can load mean-centered prior observations
### Changed
- Moved `Coordinate` object creation logic out of the `GameEstimator` to a new `CoordinateFactory`
- Raw data `DataFrames` now persisted during training
- `ModelProcessingUtils` functions no longer produce `IndexMapLoaders`, they must be passed in as arguments
- Moved `Linalg` from `com.linkedin.photon.ml.hyperparameter` to `com.linkedin.photon.ml.util`
- Replaced `HessianDiagonalAggregator` with `HessianMatrixAggregator`
- Changed the `RandomSearch` API
### Removed
- `Coefficients` companion class apply methods used only by tests
### Fixed
- Fixed bug which caused `NormalizationContext` objects to never be unpersisted
- Fixed bug which would reject one-day long day ranges
- GAME training will no longer load all coordinates in the partial retraining base model
- GAME training will no longer attempt to build an index map using inefficient legacy code when a partial retrain coordinate is missing a feature shard definition
- `AvroDataReader` will perform an explicit repartition call if the minimum number of requested partitions is not loaded by `AvroUtils`
- `AvroDataReader` will no longer drop fields that are a union of `String` and numeric types
- Variances are now approximated using the full Hessian matrix instead of the diagonal only
- Variances are now computed on the final iteration only

## [8.0.0] - 2018-02-16
### Added
- Warm-start training: re-use `GameModel` objects in between multiple training runs for faster convergence
- Style: added IDEA style XML
- Hyper-parameter tuning: added handling for discrete parameters
- Partial re-training: coordinates can be loaded from an existing model and locked in place while new coordinates are trained
- CLI parameter parsing: added several Scopt parameter parsers, custom Scopt `Read` objects, and parsing helper functions
### Changed
- Complete overhaul of the command line interfaces used to interact with the `GameTrainingDriver`, `GameScoringDriver`, `FeatureIndexingDriver`, and `NameAndTermFeatureBagsDriver`
- Complete overhaul of the API for the `GameEstimator`
- Began replacing `spark.mllib.Vector` with `spark.ml.Vector`
- Model sparsity threshold when reading/writing now configurable
- Many classes that previously expected Spark `Broadcast` objects now expect Photon `BroadcastWrapper` objects - a wrapper class around objects which may or may not be broadcast to executors depending on training settings
- Renamed all integration test classes to have 'IntegTest' in the class name
- Updated integration test documentation
- `DataReader` and `AvroDataReader` now have an option for adding an intercept dummy feature to loaded data instead of doing so by default
- Began replacing `SparkContext` with `SparkSession`
### Removed
- Custom `Params` class for managing input parameters
- `Timer` class
- Unused factored random effects and matrix factorization models
### Fixed
- Potential `ClassCastExceptions` in `DataValidators`
- Fixed bug which would cause job to fail when output mode is 'BEST' but validation is disabled: most recently trained model is selected as best
- Better defined equality between `Evaluator` objects
- Better defined equality between `GeneralizedLinearModel` objects
- Combination of normalization and random effect projection will no longer cause failures due to feature mismatch in projected spaces
- Fixed style and import errors in integration tests

## [7.0.0] - 2017-11-20
### Added
- Hyper-parameter tuning: automatic tuning of regularization hyper-parameters using random search or Bayesian search
- Output: can write data to S3 (previously assumed HDFS file system)
### Changed
- Improved data validation logging
### Fixed
- Fixed bug which would unpersist `GameModel` objects still in use
- Fixed bug which would cause per-iteration validation metrics to be logged incorrectly
- Fixed bug which would cause models to start learning from poor coefficients and potentially fail when normalization is enabled
- Fixed failures when reading Snappy-compressed Avro data
- Fixed random effect projection bug for `DenseVector`
- Fixed bug causing random effect projection to produce non-deterministic results
- Improved performance of GAME scoring
- Fixed bug causing log-likelihoods to be computed incorrectly during validation
- Fixed bug causing mean to be computed incorrectly during validation
- Fixed bug that prevented intercept-only GAME models

[18.0.0]: https://github.com/linkedin/photon-ml/compare/72baee78...aa8fccee
[17.0.0]: https://github.com/linkedin/photon-ml/compare/b84b0ed9...72baee78
[16.1.0]: https://github.com/linkedin/photon-ml/compare/bab5d3dc...b84b0ed9
[16.0.0]: https://github.com/linkedin/photon-ml/compare/c839b2bb...bab5d3dc
[15.0.0]: https://github.com/linkedin/photon-ml/compare/4030f196...c839b2bb
[14.0.0]: https://github.com/linkedin/photon-ml/compare/6a94a117...4030f196
[13.0.0]: https://github.com/linkedin/photon-ml/compare/31866ced...6a94a117
[12.0.0]: https://github.com/linkedin/photon-ml/compare/1b4ed2e1...31866ced
[11.0.0]: https://github.com/linkedin/photon-ml/compare/44d13906...1b4ed2e1
[10.2.0]: https://github.com/linkedin/photon-ml/compare/ccc5b74a...44d13906
[10.1.0]: https://github.com/linkedin/photon-ml/compare/a72ff073...ccc5b74a
[10.0.0]: https://github.com/linkedin/photon-ml/compare/ca4ae356...a72ff073
[9.0.0]: https://github.com/linkedin/photon-ml/compare/4295e15f...ca4ae356
[8.0.0]: https://github.com/linkedin/photon-ml/compare/176b810e...4295e15f
[7.0.0]: https://github.com/linkedin/photon-ml/compare/490a279f...176b810e
