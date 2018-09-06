This directory contains datasets that contain bad weight values for the samples, i.e. negative or 0 weights. The data
is the heart data, with only the weights modified via the following python script:

import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import os
import json

input_file = homedir + '/photon-client/src/integTest/resources/DriverIntegTest/input/heart.avro'
output_file = homedir + '/photon-client/src/integTest/resources/DriverIntegTest/input/zero-weights.avro'
heart_data_new_schema = avro.schema.parse(open('heart_data_game_schema.avsc', 'rb').read())

reader = DataFileReader(open(input_file, "rb"), DatumReader())
writer = DataFileWriter(open(output_file, "wb"), DatumWriter(), heart_data_new_schema)

for record in reader:
    new_record = record
    new_record['weight'] = 0
    writer.append(new_record)

reader.close()
writer.close()