package com.linkedin.photon.ml.data

import breeze.linalg.Vector

/**
 * The data structure is used to represent an entry of a sparse matrix
 * @param rowId row Id
 * @param colId column Id
 * @param label label of this entry
 * @param offset offset
 * @param weight weight
 */
case class MatrixEntry(
    rowId: String,
    colId: String,
    override val label: Double,
    override val features: Vector[Double],
    override val offset: Double = 0,
    override val weight: Double = 0)
  extends LabeledPoint(label, features, offset, weight)
