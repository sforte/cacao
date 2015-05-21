package models.regularizer

import breeze.linalg.Vector
import vectors.LazyMappedVector
import vectors.VectorOps.{l2norm,l1norm}
import scala.math.{abs,signum}
import models.Regularizer

/**
 * Created by simone on 08/05/15.
 */
class L1Regularizer(sigma: Double, epsilon: Double) extends Regularizer {

  require(sigma > 0.0 && epsilon > 0.0)

  val B = 1.0 / sigma
  val lambda = epsilon / (B * B)
  val p = sigma / lambda

  def primal(w: Vector[Double]) = 1.0/2*l2norm(w) + p*l1norm(w)

  def dualGradient(v: Vector[Double]) = {
    new LazyMappedVector(v, {case(i,x) => if (abs(x) > p) x - p*signum(x) else 0.0})
  }

  override def toString = s"${0.5*lambda} |w|^2_2 + $sigma |w|_1"
}
