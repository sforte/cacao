package vectors

import breeze.linalg.{NumericOps, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.{
  Vector => MllibVector, SparseVector => MllibSparseVector, DenseVector => MllibDenseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import math._

case class LabelledPoint(val label: Double, val features: Vector[Double]) {
  def toMllib = features match {
    case (f: DenseVector[Double]) => LabeledPoint(label, new MllibDenseVector(f.data))
    case (f: SparseVector[Double]) => LabeledPoint(label, new MllibSparseVector(f.size, f.index, f.data))
  }
}

object VectorOps {

  def l2norm(w: Vector[Double]) = w dot w
  def l1norm(w: Vector[Double]) = w.reduce(abs(_) + abs(_)).apply(0)
  def toBreeze(w: MllibVector) : Vector[Double] = w match {
    case (w: MllibDenseVector) => toBreeze(w)
    case (w: MllibSparseVector) => toBreeze(w)
  }
  def toBreeze(w: MllibDenseVector): DenseVector[Double] = new DenseVector[Double](w.values)
  def toBreeze(w: MllibSparseVector): SparseVector[Double] = new SparseVector[Double](w.indices,w.values,w.size)
}