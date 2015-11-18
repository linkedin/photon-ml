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
package com.linkedin.photon.ml.diagnostics.reporting

import java.awt.image.BufferedImage
import java.io.{PrintWriter, FileWriter}
import java.util.UUID
import com.linkedin.photon.ml.diagnostics.reporting.html.HTMLRenderStrategy
import com.linkedin.photon.ml.diagnostics.reporting.text.StringRenderStrategy
import com.xeiam.xchart.{StyleManager, ChartBuilder}
import org.testng.annotations.{Test, DataProvider}

import scala.xml._

/**
 * Generalized rendering test.
 */
class RenderStrategyTest {

  import org.testng.Assert._

  /**
   * If you are building a new physical report type, add it here.
   * @return
   */
  def generatePhysicalReports(): Map[String, PhysicalReport] = {
    // Scenarios:
    //   "Scalar" / "Simple"
    //      - Plot
    //      - Simple text
    //      - Rasterized image
    //      - link
    //   "Composite" / "Hard"
    //      - Bulleted list
    //        - Empty
    //        - Bulleted list containing scalars
    //        - Bulleted list nesting bulleted list
    //        - Bulleted list nesting numbered list
    //      - Numbered list
    //        - Empty
    //        - Numbered list containing scalars
    //        - Numbered list nesting bulleted list
    //        - Numbered list nesting numbered list
    //      - Section
    //        - Empty
    //        - Single scalar or list
    //        - Multiple scalars or lists
    //        - Nesting sections
    //      - Chapter
    //        - Empty
    //        - Single section (re-use)
    //        - Multiple sections (re-use)
    //      - Document
    //        - Empty
    //        - Single chapter
    //        - Multiple chapters
    val simpleText = new SimpleTextPhysicalReport(s"This is an example of simple text.\nParagraphs are marked with new-lines.")
    val chart = (new ChartBuilder()).chartType(StyleManager.ChartType.Bar)
      .title("This is a very boring plot")
      .theme(StyleManager.ChartTheme.XChart)
      .xAxisTitle("Category")
      .yAxisTitle("Amount")
      .build();
    chart.addSeries("Stuff", Array(1.0, 2.0, 3.0), Array(0.1, -0.2, 0.3))
    val plotWithCaption = new PlotPhysicalReport(chart, caption = Some("This is a plot with a caption"))
    val plotWithoutCaption = new PlotPhysicalReport(chart)

    val image = new BufferedImage(1280, 960, BufferedImage.TYPE_4BYTE_ABGR)
    val imageWithCaption = new RasterizedImagePhysicalReport(image, caption = Some("This is an image with a caption"))
    val imageWithoutCaption = new RasterizedImagePhysicalReport(image)

    val simpleBulletedTextList = new BulletedListPhysicalReport(Seq(
      new SimpleTextPhysicalReport("This is one item"),
      new SimpleTextPhysicalReport("This is another item")
    ))

    val simpleBulletedMixedList = new BulletedListPhysicalReport(Seq(
      simpleText,
      plotWithCaption,
      plotWithoutCaption,
      imageWithCaption,
      imageWithoutCaption))

    val nestedBulletedInsideBulletedList = new BulletedListPhysicalReport(Seq(
      new SimpleTextPhysicalReport("This is the first item in the report.\nIt should span several paragraphs."),
      simpleBulletedMixedList,
      new SimpleTextPhysicalReport("These are more items")
    ))

    val emptyBulletedList = new BulletedListPhysicalReport(Seq.empty)

    val simpleNumberedTextList = new NumberedListPhysicalReport(Seq(
      new SimpleTextPhysicalReport("This is one item"),
      new SimpleTextPhysicalReport("This is another item")
    ))

    val simpleNumberedMixedList = new NumberedListPhysicalReport(Seq(
      simpleText,
      plotWithCaption,
      plotWithoutCaption,
      imageWithCaption,
      imageWithoutCaption))

    val nestedNumberedInsideNumberedList = new NumberedListPhysicalReport(Seq(
      new SimpleTextPhysicalReport("This is the first item in the report.\nIt should span several paragraphs."),
      simpleNumberedMixedList,
      new SimpleTextPhysicalReport("These are more items")
    ))

    val emptyNumberedList = new NumberedListPhysicalReport(Seq.empty)

    val nestedNumberedInsideBulletedList = new BulletedListPhysicalReport(Seq(simpleNumberedTextList, simpleNumberedMixedList))

    val nestedBulletedInsideNumberedList = new NumberedListPhysicalReport(Seq(simpleBulletedTextList, simpleBulletedMixedList))

    val sectionWithText = new SectionPhysicalReport(Seq(simpleText, new ReferencePhysicalReport(simpleText, "This is a reference")), title = "This is a section")
    val sectionWithPlot = new SectionPhysicalReport(Seq(plotWithCaption, plotWithoutCaption), title = "I am a section that contains several plots")
    val sectionWithImg = new SectionPhysicalReport(Seq(imageWithCaption, imageWithoutCaption), title = "I am a section that contains several images")
    val emptySection = new SectionPhysicalReport(Seq.empty, title = "I am an empty section")
    val sectionWithStuff = new SectionPhysicalReport(
      Seq(
        simpleBulletedTextList,
        simpleBulletedMixedList,
        nestedBulletedInsideBulletedList,
        emptyBulletedList,
        simpleNumberedTextList,
        simpleNumberedMixedList,
        nestedNumberedInsideBulletedList,
        nestedBulletedInsideNumberedList,
        simpleText),
      title = "I am a section with just about everything")
    val nestedSections = new SectionPhysicalReport(
      Seq(
        emptySection,
        sectionWithImg,
        simpleBulletedTextList,
        sectionWithPlot,
        sectionWithImg),
      title = "I am a section with nested sections")

    val emptyChapter = new ChapterPhysicalReport(Seq.empty, title = "I am an empty chapter")
    val singleSectionChapter = new ChapterPhysicalReport(Seq(nestedSections), title = "I am a chapter with nested sections")
    val multipleSectionChapter = new ChapterPhysicalReport(Seq(nestedSections, sectionWithText, sectionWithStuff), title = "I am a chapter with multiple sections")

    val emptyDoc = new DocumentPhysicalReport(Seq.empty, title = "I am an empty document")
    val docWithStuff = new DocumentPhysicalReport(Seq(singleSectionChapter, emptyChapter, multipleSectionChapter), title = "I am a document with lots of stuff")

    Map(
      "Simple text" -> simpleText,
      "Plot with caption" -> plotWithCaption,
      "Plot without caption" -> plotWithoutCaption,
      "Image with caption" -> imageWithCaption,
      "Image without caption" -> imageWithoutCaption,
      "Simple bulleted list with text" -> simpleBulletedTextList,
      "Simple bulleted list with mixed contents" -> simpleBulletedMixedList,
      "Bulleted list containing numbered list" -> nestedNumberedInsideBulletedList,
      "Bulleted list containing bulleted list" -> nestedBulletedInsideBulletedList,
      "Simple numbered list with text" -> simpleNumberedTextList,
      "Simple numbered list with mixed contents" -> simpleNumberedMixedList,
      "Numbered list containing numbered list" -> nestedNumberedInsideNumberedList,
      "Numbered list containing numbered list" -> nestedNumberedInsideNumberedList,
      "Section with text" -> sectionWithText,
      "Section with plot" -> sectionWithPlot,
      "Section with image" -> sectionWithImg,
      "Empty section" -> emptySection,
      "Section with complex content" -> sectionWithStuff,
      "Empty chapter" -> emptyChapter,
      "Single section chapter" -> singleSectionChapter,
      "Multiple section chapter" -> multipleSectionChapter,
      "Empty document" -> emptyDoc,
      "Multiple chapter document" -> docWithStuff)
  }

  /**
   * If you are building a new render strategy, add it here.
   */
  def generateRenderStrategies(): Map[String, (RenderStrategy[PhysicalReport, Any], Any => Unit)] = {
    val binding = NamespaceBinding("svg", "http://www.w3.org/2000/svg", NamespaceBinding(null, "http://www.w3.org/1999/xhtml", TopScope))

    Map(
      "simple text" ->(new StringRenderStrategy(), (x: Any) => {
        x match {
          case s: String =>
            assertTrue(s.length > 0, s"Generated length [${s.length}] is non-zero")
          // Leaving this here in case it is useful to review the output

          //            val out = new PrintWriter(new FileWriter(UUID.randomUUID().toString + ".txt"))
          //
          //            try {
          //              out.println(s)
          //            } finally {
          //              out.close()
          //            }
          case null =>
            fail("Got a null")
          case _ =>
            fail(s"Expected to see a string, got a [${x.getClass.getName}] instead")
        }
      }: Unit),
      "HTML" ->(
        new HTMLRenderStrategy(
          null,
          "svg",
          binding),
        (x: Any) => {
          x match {
            case n: Node =>
            // Same as above -- leaving this here in case it's useful to review the output
            // vacuously demonstrated that this isn't empty (hopefully...)
            //              val out = new PrintWriter(new FileWriter(UUID.randomUUID().toString + ".html"))
            //              val indent = 2
            //              val maxWidth = 80
            //              try {
            //                val pp = new PrettyPrinter(maxWidth, indent)
            //                val buffer = new StringBuilder()
            //                pp.format(n, binding, buffer)
            //                out.println(buffer.toString)
            //                out.flush()
            //              } finally {
            //                out.close()
            //              }
            case null =>
              fail("Got a null")
            case _ =>
              fail(s"Expected to see a Node, got a [${x.getClass.getName}] instead")
          }
        }: Unit)
    )
  }

  /**
   * Generate test cases by building the cartesian product of all possible report fragments with
   * all possible (render strategy, validator) tuples
   */
  @DataProvider
  def generateHappyCaseScenarios(): Array[Array[Any]] = {
    (for {rep <- generatePhysicalReports(); ren <- generateRenderStrategies()} yield {
      Array[Any](s"Report: ${rep._1}, renderer: ${ren._1}", rep._2, ren._2._1, ren._2._2)
    }).toArray
  }

  @Test(dataProvider = "generateHappyCaseScenarios")
  def checkRender[P <: PhysicalReport](desc: String, report: P, renderStrategy: RenderStrategy[P, Any], validator: Any => Unit) = {
    assertTrue(report != null, "Report is not null")
    assertTrue(renderStrategy != null, "Render strategy is not null")
    assertTrue(validator != null, "Validator is not null")
    val renderer = renderStrategy.locateRenderer(report)
    assertTrue(renderer != null, "Renderer is not null")
    val result = renderer.render(report)
    validator(result)
  }
}
