package models.regularizer

import breeze.linalg.Vector
import models.Regularizer
import vectors.LazyMappedVector
import vectors.VectorOps.{l1norm, l2norm}

import scala.math.{abs, signum}

class ElasticNet(lam: Double, alpha:Double) extends Regularizer {

  require(0 < alpha && alpha <= 1 && lam > 0)

  val lambda = 2*alpha*lam
  val p = (1-alpha)/(2*alpha)

  def primal(w: Vector[Double]) = 1.0/2*l2norm(w) + p*l1norm(w)

  def dualGradient(v: Vector[Double]) = {
    new LazyMappedVector(v, {case(i,x) => if (abs(x) > p) x - p*signum(x) else 0.0})
  }

  override def toString = s"${alpha*lam} |w|^2_2 + ${(1-alpha)*lam} |w|_1"
}
