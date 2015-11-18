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
import org.gradle.api.Project
import static ScalaUtils.getScalaVersionSuffix

class ScalaResolverPlugin implements Plugin<Project> {

  Project project

  def initResolver(String scalaVersion) {
    def scalaSuffix = getScalaVersionSuffix(scalaVersion)
    project.configurations.all {
      resolutionStrategy.eachDependency { dep ->
        if (dep.target.group == 'org.scala-lang' && dep.target.version != scalaVersion) {
          dep.useVersion scalaVersion
        }
        def scalaPattern = dep.target.name =~ /(.+)(_2(\.[0-9]{1,2}){1,2})/
        if (scalaPattern.matches()) {
          def moduleName = scalaPattern.group(1)
          def scalaVariant = scalaPattern.group(2)
          if (scalaVariant != scalaSuffix) {
            println("replacing binary incompatible dependency ${dep.target.name} with ${moduleName + scalaSuffix}")
            dep.useTarget group: dep.target.group, name: moduleName + scalaSuffix, version: dep.target.version
          }
        }
      }
    }
  }

  void apply(Project project) {
    this.project = project
    project.extensions.create('scalaResolver', ScalaResolverExtension, this.&initResolver)
  }
}
