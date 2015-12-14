package com.linkedin.photon.ml.diagnostics.reporting.html

import com.linkedin.photon.ml.diagnostics.reporting._

import scala.xml._

class NumberedListToHTMLRenderer(
    renderStrategy: RenderStrategy[PhysicalReport, Node],
    numberingContext: NumberingContext,
    namespaceBinding: NamespaceBinding,
    htmlPrefix: String,
    svgPrefix: String)
  extends AbstractListToHTMLRenderer[NumberedListPhysicalReport](
    "ol", renderStrategy, numberingContext, namespaceBinding, htmlPrefix, svgPrefix)
