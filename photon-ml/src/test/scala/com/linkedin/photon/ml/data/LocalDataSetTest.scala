package com.linkedin.photon.ml.data

import java.util.Random

import breeze.linalg.{SparseVector, Vector}
import org.testng.Assert._
import org.testng.annotations.Test

import com.linkedin.photon.ml.constants.MathConst


class LocalDataSetTest {

   @Test(dependsOnGroups = Array[String]("testReservoirSampling", "testCore"))
   def testReservoirSamplingOnAllSamples(): Unit = {
     val numSamples = 10
     val random = new Random(MathConst.RANDOM_SEED)
     val weight = random.nextDouble()
     val localDataSet = LocalDataSet(Array.tabulate(numSamples)(i => (i.toLong, new LabeledPoint(label = 1.0, features = Vector(), weight = weight))))

     // don't keep any sample
     val filteredDataPoints0 = localDataSet.reservoirSamplingOnAllSamples(numSamplesToKeep = 0).dataPoints
     assertEquals(filteredDataPoints0.length, 0)

     // keep 1 sample
     val filteredDataPoints1 = localDataSet.reservoirSamplingOnAllSamples(numSamplesToKeep = 1).dataPoints
     assertEquals(filteredDataPoints1.length, 1)
     filteredDataPoints1.foreach { case(_, labeledPoint) => assertEquals(labeledPoint.weight, numSamples * weight, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD) }

     // keep numSamples samples
     val filteredDataPoints2 = localDataSet.reservoirSamplingOnAllSamples(numSamplesToKeep = numSamples).dataPoints
     assertEquals(filteredDataPoints2.length, numSamples)
     filteredDataPoints2.foreach { case(_, labeledPoint) => assertEquals(labeledPoint.weight, weight, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD) }

     // keep numSamples + 1 samples
     val filteredDataPoints3 = localDataSet.reservoirSamplingOnAllSamples(numSamplesToKeep = numSamples + 1).dataPoints
     assertEquals(filteredDataPoints3.length, numSamples)
     filteredDataPoints3.foreach { case(_, labeledPoint) => assertEquals(labeledPoint.weight, weight, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD) }

   }

   @Test(dependsOnGroups = Array[String]("testPearsonCorrelationScore", "testCore"))
   def testFilterFeaturesByPearsonCorrelationScore(): Unit = {
     val numSamples = 10
     val random = new Random(MathConst.RANDOM_SEED)
     val labels = Array.fill(numSamples)(if (random.nextDouble() > 0.5) 1.0 else -1.0)
     val numFeatures = 10
     // Each data point has 10 features, and each of them is designed as following:
     // 0: Intercept
     // 1: Positively correlated with the label
     // 2: Negatively correlated with the label
     // 3: Un-correlated with the label
     // 4: Dummy feature 1
     // 5: Dummy feature 2
     // 6-9: Missing features
     val intercept = 1.0
     val variance = 0.001
     val featureIndices = Array(0, 1, 2, 3, 4, 5)
     val features = Array.tabulate(numSamples) { i =>
       val featureValues = Array(
         intercept,
         labels(i) + variance * random.nextGaussian(),
         -labels(i) + variance * random.nextGaussian(),
         random.nextDouble(),
         1.0,
         1.0)
       new SparseVector[Double](featureIndices, featureValues, numFeatures)
     }
     val localDataSet = LocalDataSet(Array.tabulate(numSamples)(i => (i.toLong, LabeledPoint(labels(i), features(i), offset = 0.0, weight = 1.0))))

     // don't keep any features
     val filteredDataPoints0 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 0).dataPoints
     assertEquals(filteredDataPoints0.length, numSamples)
     assertTrue(filteredDataPoints0.forall(_._2.features.activeSize == 0))

     // keep 1 feature
     val filteredDataPoints1 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 1).dataPoints
     val filteredDataPointsKeySet1 = filteredDataPoints1.flatMap(_._2.features.activeKeysIterator).toSet
     assertEquals(filteredDataPoints1.length, numSamples)
     assertTrue(filteredDataPoints1.forall(_._2.features.activeSize == 1))
     assertTrue(filteredDataPointsKeySet1.size == 1 &&
       (filteredDataPointsKeySet1.contains(0) || filteredDataPointsKeySet1.contains(4) || filteredDataPointsKeySet1.contains(5)),
       s"$filteredDataPointsKeySet1")

     // keep 3 features
     val filteredDataPoints3 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 3).dataPoints
     val filteredDataPointsKeySet3 = filteredDataPoints3.flatMap(_._2.features.activeKeysIterator).toSet
     assertEquals(filteredDataPoints3.length, numSamples)
     assertTrue(filteredDataPoints3.forall(_._2.features.activeSize == 3))
     assertTrue(filteredDataPointsKeySet3.size == 3 && filteredDataPointsKeySet3.contains(1) && filteredDataPointsKeySet3.contains(2) &&
       (filteredDataPointsKeySet3.contains(0) || filteredDataPointsKeySet3.contains(4) || filteredDataPointsKeySet3.contains(5)),
       s"$filteredDataPointsKeySet3")

     // keep 5 features
     val filteredDataPoints5 = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = 5).dataPoints
     val filteredDataPointsKeySet5 = filteredDataPoints5.flatMap(_._2.features.activeKeysIterator).toSet
     assertEquals(filteredDataPoints5.length, numSamples)
     assertTrue(filteredDataPoints5.forall(_._2.features.activeSize == 5))
     assertTrue(filteredDataPointsKeySet5.forall(_ < 6), s"$filteredDataPointsKeySet5")

     // keep all features
     val filteredDataPointsAll = localDataSet.filterFeaturesByPearsonCorrelationScore(numFeaturesToKeep = numFeatures).dataPoints
     assertEquals(filteredDataPointsAll.length, numSamples)
     assertTrue(filteredDataPointsAll.forall(dataPoint => dataPoint._2.features.activeKeysIterator.toSet == Set(0, 1, 2, 3, 4, 5)))
   }

   @Test(dependsOnGroups = Array[String]("testCore"))
   def testFilterFeaturesBySupport(): Unit = {
     val numSamples = 4
     val numFeatures = 4
     val random = new Random(MathConst.RANDOM_SEED)
     val labels = Array.fill(numSamples)(if (random.nextDouble() > 0.5) 1.0 else -1.0)
     val features = Array(
       new SparseVector[Double](Array(0, 1, 2, 3), Array.fill(4)(random.nextDouble()), numFeatures),
       new SparseVector[Double](Array(0, 1, 2), Array.fill(3)(random.nextDouble()), numFeatures),
       new SparseVector[Double](Array(0, 1), Array.fill(2)(random.nextDouble()), numFeatures),
       new SparseVector[Double](Array(0), Array.fill(1)(random.nextDouble()), numFeatures)
     )
     val localDataSet = LocalDataSet(Array.tabulate(numSamples)(i => (i.toLong, LabeledPoint(labels(i), features(i), offset = 0.0, weight = 1.0))))

     // don't keep any features
     val filteredDataPoints0 = localDataSet.filterFeaturesBySupport(minNumSupportThreshold = 5).dataPoints
     assertEquals(filteredDataPoints0.length, numSamples)
     assertTrue(filteredDataPoints0.forall(_._2.features.activeSize == 0))

     // keep 1 feature
     val filteredDataPoints1 = localDataSet.filterFeaturesBySupport(minNumSupportThreshold = 4).dataPoints
     assertEquals(filteredDataPoints1.length, numSamples)
     filteredDataPoints1.zip(localDataSet.dataPoints).foreach { case((_, filteredLabeledPoint), (_, labeledPoint)) =>
       assertEquals(filteredLabeledPoint.features.activeSize, 1)
       assertTrue(filteredLabeledPoint.features.activeKeysIterator.toSet.contains(0))
     }

     // keep 3 features
     val filteredDataPoints3 = localDataSet.filterFeaturesBySupport(minNumSupportThreshold = 2).dataPoints
     assertTrue(filteredDataPoints3.length == numSamples)
     filteredDataPoints3.zip(localDataSet.dataPoints).foreach { case((_, filteredLabeledPoint), (_, labeledPoint)) =>
       assertTrue(filteredLabeledPoint.features.activeSize == 3 || filteredLabeledPoint.features.activeSize == labeledPoint.features.activeSize)
       assertTrue(!filteredLabeledPoint.features.activeKeysIterator.toSet.contains(3))
     }

     // keep all features
     val filteredDataPoints4 = localDataSet.filterFeaturesBySupport(minNumSupportThreshold = 0).dataPoints
     assertTrue(filteredDataPoints4.length == numSamples)
     filteredDataPoints4.zip(localDataSet.dataPoints).foreach { case((_, filteredLabeledPoint), (_, labeledPoint)) =>
       assertTrue(filteredLabeledPoint.features.activeSize == numFeatures || filteredLabeledPoint.features.activeSize == labeledPoint.features.activeSize)
     }
   }

   /**
    * Test the Pearson correlation score
    */
   @Test(groups = Array[String]("testPearsonCorrelationScore", "testCore"))
   def testPearsonCorrelationScore(): Unit = {
     //test input data
     val labels = Array(1.0, 4.0, 6.0, 9.0)
     val features = Array(
       Vector(0.0, 0.0, 2.0), Vector(5.0, 0.0, -3.0), Vector(7.0, 0.0, -8.0), Vector(0.0, 0.0, -1.0)
     )
     val expected = Map(0 -> 0.05564149, 1 -> 1.0, 2 -> -0.40047142)
     val labelAndFeatures = labels.zip(features)
     val computed = LocalDataSet.computePearsonCorrelationScore(labelAndFeatures.iterator)
     computed.foreach { case (key, value) =>
       assertEquals(expected(key), value, MathConst.LOW_PRECISION_TOLERANCE_THRESHOLD,
         s"Computed Pearson correlation score is $value, while the expected value is ${expected(key)}.")
     }
   }

   /**
    * Test the reservoir sampling
    */
   @Test(groups = Array[String]("testReservoirSampling", "testCore"))
   def testReservoirSampling(): Unit = {
     val random = new Random(MathConst.RANDOM_SEED)

     val input = Seq.fill(100)(random.nextInt())

     // input size < k
     val sample1 = LocalDataSet.reservoirSampling(input.iterator, 150)
     assertEquals(sample1.length, 100, s"Expected number of samples is 100, while ${sample1.length} is obtained")
     assertTrue(input == sample1.toSeq)

     // input size == k
     val sample2 = LocalDataSet.reservoirSampling(input.iterator, 100)
     assertEquals(sample2.length, 100, s"Expected number of samples is 100, while ${sample2.length} is obtained")
     assertTrue(input.zip(sample2).forall { case(i1, i2) => i1 == i2 })

     // input size > k
     val sample3 = LocalDataSet.reservoirSampling(input.iterator, 10)
     assertEquals(sample3.length, 10, s"Expected number of samples is 10, while ${sample3.length} is obtained")
   }
 }
