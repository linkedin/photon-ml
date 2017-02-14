/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.linkedin.photon.ml.hyperparameter

import breeze.linalg.DenseVector
import breeze.numerics.log
import breeze.stats.distributions.{ContinuousDistr, Gaussian, Laplace}
import org.apache.commons.math3.distribution.{LaplaceDistribution, NormalDistribution, RealDistribution}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest
import org.testng.Assert._
import org.testng.annotations.{DataProvider, Test}

/**
 * Test cases for the SliceSampler class
 */
class SliceSamplerTest {

  val alpha = 0.001
  val numBurnInSamples = 100
  val numSamples = 2000
  val seed = 0

  @DataProvider
  def distributionDataProvider() =
    Array(
      Array(new Gaussian(0.0, 0.5), new NormalDistribution(0.0, 0.5)),
      Array(new Gaussian(0.0, 1.0), new NormalDistribution(0.0, 1.0)),
      Array(new Gaussian(1.0, 4.0), new NormalDistribution(1.0, 4.0)),
      Array(new Gaussian(-2.0, 2.1), new NormalDistribution(-2.0, 2.1)),
      Array(new Gaussian(2.5, 2.3), new NormalDistribution(2.5, 2.3)),
      Array(new Laplace(0.0, 0.5), new LaplaceDistribution(0.0, 0.5)),
      Array(new Laplace(0.0, 1.0), new LaplaceDistribution(0.0, 1.0)),
      Array(new Laplace(1.0, 4.0), new LaplaceDistribution(1.0, 4.0)),
      Array(new Laplace(-2.0, 2.1), new LaplaceDistribution(-2.0, 2.1)),
      Array(new Laplace(2.5, 2.3), new LaplaceDistribution(2.5, 2.3)))

  @Test(dataProvider = "distributionDataProvider")
  def testSampledDistribution(
      sourceDistribution: ContinuousDistr[Double],
      testDistribution: RealDistribution): Unit = {

    def logp(x: DenseVector[Double]) = log(sourceDistribution.pdf(x(0)))
    val sampler = new SliceSampler(logp, seed = seed)

    // Sampler burn-in
    val init = (0 until numBurnInSamples)
      .foldLeft(DenseVector(0.0)) { (currX, _) =>
        sampler.draw(currX)
      }

    // Draw the real samples
    val (_, samples) = (0 until numSamples)
      .foldLeft((init, List.empty[Double])) { case ((currX, ls), _) =>
        val x = sampler.draw(currX)
        (x, ls :+ x(0))
      }

    // Run a Kolmogorov-Smirnov test to confirm the distribution of the samples
    val tester = new KolmogorovSmirnovTest(new MersenneTwister(seed))
    val pval = tester.kolmogorovSmirnovTest(testDistribution, samples.toArray)
    assertTrue(pval > alpha)
  }
}
