package com.linkedin.photon.ml.test

import java.io.File
import java.nio.file.{Paths, Files}
import org.apache.commons.io.FileUtils
import org.testng.annotations.AfterMethod

/**
 * Thread safe test template to provide a temporary directory per method.
 * @author dpeng
 */
trait TestTemplateWithTmpDir {
  /**
   * Return the temporary directory as a string.
   * @return the temporary directory as a string.
   */
  def getTmpDir: String = TestTemplateWithTmpDir._tmpDirThreadLocal.get()

  @AfterMethod
  def afterMethod(): Unit = {
    FileUtils.cleanDirectory(new File(getTmpDir))
  }
}

private object TestTemplateWithTmpDir {
  private val _tmpDirThreadLocal: ThreadLocal[String] = new ThreadLocal[String] {
    protected override def initialValue(): String = {
      val parentDir = Paths.get(FileUtils.getTempDirectoryPath)
      val prefix = Thread.currentThread().getId + "-" + System.currentTimeMillis()
      val dir = Files.createTempDirectory(parentDir, prefix).toFile
      FileUtils.forceDeleteOnExit(dir)
      dir.toString
    }
  }
}
