package models.regularizer

import breeze.linalg.Vector
import vectors.LazyMappedVector
import vectors.VectorOps.{l2norm,l1norm}
import scala.math.{abs,signum}
import models.Regularizer

/**
 * Created by simone on 08/05/15.
 */
class ElasticNet(sigma: Double, epsilon: Double) extends Regularizer {

  val B = 1.0 / sigma
  val lambda = epsilon / (B * B)
  val p = sigma / lambda
//  println(s"0.5 $lambda |w|^2_2 + $sigma |w|_1")
  def primal(w: Vector[Double]) = 1.0/2*l2norm(w) + p*l1norm(w)

  def dualGradient(v: Vector[Double]) = {
    new LazyMappedVector(v, x => if (abs(x) > p) x - p*signum(x) else 0.0)
  }
}
