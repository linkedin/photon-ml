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
package org.apache.spark.photon;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

import org.apache.avro.generic.GenericData;
import org.apache.spark.serializer.KryoRegistrator;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.serializers.CollectionSerializer;

/**
 * Registrator for Kryo serialization. It adds support for serialization of the Avro GenericData.Array class. The
 * SpecificInstanceCollectionSerializer is copied from FLINK-1391.
 *
 * Required for DataFrame operations on Avro GenericData containing lists.
 *
 * Sample usage: set conf
 *
 * --conf spark.kryo.registrator=org.apache.spark.serializer.AvroGenericArrayRegistrator
 */
public class AvroGenericArrayRegistrator implements KryoRegistrator {

  @SuppressWarnings({"unchecked", "rawtypes"})
  @Override
  public void registerClasses(Kryo kryo) {
    kryo.register(GenericData.Array.class, new SpecificInstanceCollectionSerializer(ArrayList.class));
  }

  /**
   * Copied from FLINK-1391
   * Special serializer for Java collections enforcing certain instance types.
   * Avro is serializing collections with an "GenericData.Array" type. Kryo is not able to handle
   * this type, so we use ArrayLists.
   */
  public static class SpecificInstanceCollectionSerializer<T extends java.util.ArrayList<?>> extends CollectionSerializer implements Serializable {
    private static final long serialVersionUID = 1L;
    private Class<T> type;

    public SpecificInstanceCollectionSerializer(Class<T> type) {
      this.type = type;
    }

    @SuppressWarnings("rawtypes")
    @Override
    protected Collection create(Kryo kryo, Input input, Class<Collection> type) {
      return kryo.newInstance(this.type);
    }

    @SuppressWarnings("rawtypes")
    @Override
    protected Collection createCopy(Kryo kryo, Collection original) {
      return kryo.newInstance(this.type);
    }
  }
}
