package com.linkedin.photon.ml.projector


trait ProjectorType

object ProjectorType extends Enumeration {
  val RANDOM, INDEX_MAP, IDENTITY = Value
}

case class RandomProjection(projectedSpaceDimension: Int) extends ProjectorType

case object IndexMapProjection extends ProjectorType

case object IdentityProjection extends ProjectorType
