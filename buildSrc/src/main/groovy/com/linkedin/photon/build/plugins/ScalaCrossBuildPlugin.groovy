/*
 * Copyright 2015 LinkedIn Corp. All rights reserved.
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
package com.linkedin.photon.build.plugins

import org.gradle.api.Plugin
import org.gradle.api.initialization.Settings
import static ScalaUtils.getScalaVersionSuffix

/**
 * This plugin provides Scala cross-build capability, it creates multiple projects with different scala version suffixes
 * that share the same module directory.
 *
 * @author cfreeman
 */
class ScalaCrossBuildPlugin implements Plugin<Settings> {

  private static def includeSuffixedProject(settings, module, scalaVersion) {
    def path = module + getScalaVersionSuffix(scalaVersion)
    settings.include(path)
    def project = settings.findProject(path)
    project.projectDir = new File(project.projectDir.parent, module.split(':').last())
  }

  void apply(Settings settings) {

    def scalaCrossBuild = settings.extensions.create('scalaCrossBuild', ScalaCrossBuildExtension, settings.startParameter.projectProperties)

    scalaCrossBuild.projectsToNotCrossBuild.all { module ->
      settings.include(module)
    }

    scalaCrossBuild.projectsToCrossBuild.all { module ->
      if (scalaCrossBuild.buildDefaultOnly) {
        includeSuffixedProject(settings, module, scalaCrossBuild.defaultScalaVersion)
      }
      else {
        scalaCrossBuild.targetScalaVersions.each { v -> includeSuffixedProject(settings, module, v) }
      }
    }

    settings.gradle.projectsLoaded { g ->
      g.rootProject.subprojects {
        def projectScalaVersion = scalaCrossBuild.targetScalaVersions.find { name.contains(getScalaVersionSuffix(it)) }
        def scalaVersion = projectScalaVersion ? projectScalaVersion : scalaCrossBuild.defaultScalaVersion
        def scalaSuffix = getScalaVersionSuffix(scalaVersion)
        ext.scalaVersion = scalaVersion
        ext.scalaSuffix = scalaSuffix
        ext.defaultScalaVersion = scalaCrossBuild.defaultScalaVersion
        ext.defaultScalaSuffix = getScalaVersionSuffix(scalaCrossBuild.defaultScalaVersion)

        // map the output directories in a way such that outputs won't overlap
        buildDir = "${g.rootProject.buildDir}/$name"
      }
    }

  }
}