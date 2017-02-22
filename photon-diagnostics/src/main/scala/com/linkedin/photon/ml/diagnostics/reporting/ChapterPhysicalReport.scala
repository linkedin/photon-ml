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
package com.linkedin.photon.ml.diagnostics.reporting

/**
 * A container report representing a collection of sections
 *
 * @param sections
 */
class ChapterPhysicalReport(sections:Seq[SectionPhysicalReport], val title:String)
  extends SequencePhysicalReport[SectionPhysicalReport](sections) {

  override def toString():String = s"CHAPTER [ID: ${getId()}, TITLE:$title]"
}
