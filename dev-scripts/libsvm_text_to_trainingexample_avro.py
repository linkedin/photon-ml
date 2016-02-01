# Copyright 2016 LinkedIn Corp. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain a
# copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""
[Doc]:
This script converts a text file in libsvm format into TrainingExample avro.
For each feature, the name is set as id, and the term is empty.

[Usage]:
python libsvm_text_to_trainingexample_avro.py [input_path] [output_schema_path] [output_path] (optional: -r for regression)
"""
import os
import sys
import getopt

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter

def main():
  if len(sys.argv) <= 1:
    print __doc__
    sys.exit(0)

  reg = False

  # parse command line options
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hr", ["help", "regression"])
  except getopt.error, msg:
    print msg
    print "for help use --help"
    sys.exit(2)
  # process options
  for o, a in opts:
    if o in ("-h", "--help"):
      print __doc__
      sys.exit(0)
    if o in ("-r", "--regression"):
      reg = True

  # process arguments
  input_path = args[0]
  output_schema_path = args[1]
  output_path = args[2]

  if os.path.exists(output_path):
    os.remove(output_path)

  schema = avro.schema.parse(open(output_schema_path).read())
  writer = DataFileWriter(open(output_path, "w"), DatumWriter(), schema)

  with open(input_path, 'r') as f:
    count = 0
    for line in f:
      count += 1
      r = {}
      i = 0
      feature_arr = []
      for token in line.strip().split(' '):
        if i == 0:
          if reg:
            r['label'] = float(token)
          else:
            r['label'] = int(token)
            if r['label'] <= 0:
              r['label'] = 0
            else:
              r['label'] = 1
        else:
          t = token.split(':')
          ft = {}
          ft['name'] = t[0]
          ft['term'] = ''
          ft['value'] = float(t[1])
          feature_arr.append(ft)
        i += 1
      r['features'] = feature_arr
      writer.append(r)
    print "converted " + str(count) + " examples"
  writer.close()

if __name__ == "__main__":
  main()
