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
package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class NumberedListToHTMLRenderer(renderStrategy: RenderStrategy[PhysicalReport, Node],
                                 numberingContext: NumberingContext,
                                 namespaceBinding: NamespaceBinding,
                                 htmlPrefix: String,
                                 svgPrefix: String) extends AbstractListToHTMLRenderer[NumberedListPhysicalReport]("ol", renderStrategy, numberingContext, namespaceBinding, htmlPrefix, svgPrefix)