This directory contains a dataset with user-defined column names that are not standard.

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import os
import json

input_file = homedir + '/photon-client/src/integTest/resources/DriverIntegTest/input/heart.avro'
output_file = homedir + '/photon-client/src/integTest/resources/DriverIntegTest/input/different-column-names/diff-col-names.avro'
heart_data_new_schema = avro.schema.parse(open('heart_data_diff_col_names.avsc', 'rb').read())

reader = DataFileReader(open(input_file, "rb"), DatumReader())
writer = DataFileWriter(open(output_file, "wb"), DatumWriter(), heart_data_new_schema)

for record in reader:
    new_record = {"uid": record["uid"], "the_label": "", "w": 0, "intercept": 0, "features": record["features"]}
    writer.append(new_record)

reader.close()
writer.close()