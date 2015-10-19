package com.linkedin.photon.ml.io

import com.linkedin.mlease.spark.test.SparkTestUtils
import com.linkedin.photon.ml.test.{TestTemplateWithTmpDir, SparkTestUtils}
import org.apache.spark.SparkContext
import org.testng.Assert.assertEquals
import org.testng.annotations.Test


/**
 * A simple check for [[LogWriter]]
 * @author dpeng
 */
class LogWriterIntegTest extends SparkTestUtils with TestTemplateWithTmpDir {
  private def checkContent(file: String, content: String, sc: SparkContext): Unit = {
    val actual = sc.textFile(file, 1).collect().mkString("\n")
    assertEquals(actual, content)
  }

  @Test
  def testLogWriter(): Unit = sparkTest("LogWriter") {
    val path = getTmpDir + "/" + getClass.getSimpleName
    val file = path + "/log-message.txt"
    val logger = new LogWriter(path, sc)
    logger.print("Test print")
    logger.println("Test println")
    logger.println("Test println 2")
    logger.flush()
    logger.close()
    checkContent(file, "Test printTest println\nTest println 2", sc)
  }
}
