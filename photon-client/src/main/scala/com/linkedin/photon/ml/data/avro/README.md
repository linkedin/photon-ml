Refactorings needed here:
TODO Simplify avro field names and allow user to pass in the "label" column name
TODO Split ModelProcessingUtils into what's avro and what's not, locate accordingly

Note:
- ModelProcessingUtils does not belong here, as this module is for "data", as in "training data", "validation data". It
should go into photon.ml.model.
