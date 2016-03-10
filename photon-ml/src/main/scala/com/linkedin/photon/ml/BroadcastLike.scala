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
package com.linkedin.photon.ml

/**
 * A trait to hold some simple operations on the Broadcasted variables
 * @author xazhang
 */
trait BroadcastLike {
  /**
   * Asynchronously delete cached copies of this broadcast on the executors
   * @return This object with all its broadcasted variables unpersisted
   */
  def unpersistBroadcast(): this.type
}
