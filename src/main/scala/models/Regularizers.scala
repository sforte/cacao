package models

import breeze.linalg.Vector
import vectors.LazyMappedVector
import vectors.VectorOps._

import scala.math._

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

class L2Regularizer(val lambda: Double) extends Regularizer {

  def primal(w: Vector[Double]) = l2norm(w) / 2
  def dualGradient(w: Vector[Double]) = w

  def dual2(v: Vector[Double]) = l2norm(v)/2
}