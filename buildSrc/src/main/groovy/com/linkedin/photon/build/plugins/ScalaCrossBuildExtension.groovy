/*
 * Copyright 2016 LinkedIn Corp. All rights reserved.
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

import org.gradle.api.file.FileTree
import org.gradle.api.internal.DefaultDomainObjectSet

/**
 * Provides configurable extension for ScalaCrossBuildPlugin.
 */
class ScalaCrossBuildExtension {

  private String _defaultScalaVersion
  private List<String> _targetScalaVersions
  private boolean _buildDefaultOnly = false
  private Map<String, String> _overrides

  def projectsToNotCrossBuild = new DefaultDomainObjectSet(String)
  def projectsToCrossBuild = new DefaultDomainObjectSet(String)

  ScalaCrossBuildExtension(Map<String, String> overrides) {
    if (overrides.defaultScalaVersion) _defaultScalaVersion = overrides.defaultScalaVersion
    if (overrides.targetScalaVersions) _targetScalaVersions = overrides.targetScalaVersions.split(',') as List<String>
    if (overrides.buildDefaultOnly) _buildDefaultOnly = overrides.buildDefaultOnly.toBoolean()
    _overrides = overrides
  }

  // Find all projects by doing a DFS from the root directory and finding all build.gradle files.  Infer that a project
  // is a "Scala" project if it has a scala directory in its main source set.
  def discoverProjects(FileTree root) {

    def allProjects = [] as Set<String>
    def scalaProjects = [] as Set<String>
    def curProject = ''
    root.visit { file ->
      if (file.name == 'build.gradle') {
        if (file.relativePath.segments.length > 1) {
          curProject = ':' + file.relativePath.segments[0..-2].join(':')
          allProjects.add(curProject)
        }
      }
      else if (file.relativePath.toString().endsWith('/src/main/scala')) {
        scalaProjects.add(curProject)
      }
    }

    projectsToNotCrossBuild(allProjects - scalaProjects)
    projectsToCrossBuild(scalaProjects)
  }

  void defaultScalaVersion(String defaultScalaVersion) {
    if (!_overrides.defaultScalaVersion) _defaultScalaVersion = defaultScalaVersion
  }

  void targetScalaVersions(String... targetScalaVersion) {
    if (!_overrides.targetScalaVersions) _targetScalaVersions = targetScalaVersion as List<String>
  }

  void buildDefaultOnly(boolean buildDefaultOnly) {
    if (!_overrides.buildDefaultOnly) _buildDefaultOnly = buildDefaultOnly
  }

  void projectsToNotCrossBuild(Collection<String> projectPaths) {
    projectPaths = projectPaths.collect { it.startsWith(':') ? it : ':' + it }
    projectsToNotCrossBuild.addAll(projectPaths - projectsToCrossBuild)
  }

  void projectsToNotCrossBuild(String... projectPaths) {
    projectsToNotCrossBuild(projectPaths as List<String>)
  }

  void projectsToCrossBuild(Collection<String> projectPaths) {
    projectPaths = projectPaths.collect { it.startsWith(':') ? it : ':' + it }
    projectsToCrossBuild.addAll(projectPaths - projectsToNotCrossBuild)
  }

  void projectsToCrossBuild(String... projectPaths) {
    projectsToCrossBuild(projectPaths as List<String>)
  }

  String getDefaultScalaVersion() {
    return _defaultScalaVersion
  }

  List<String> getTargetScalaVersions() {
    return _targetScalaVersions
  }

  boolean getBuildDefaultOnly() {
    return _buildDefaultOnly
  }

}