package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class BulletedListToHTMLRenderer(
    renderStrategy: RenderStrategy[PhysicalReport, Node],
    numberingContext: NumberingContext,
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends AbstractListToHTMLRenderer[BulletedListPhysicalReport](
    "ul", renderStrategy, numberingContext, namespaceBinding, htmlPrefix, svgPrefix)
