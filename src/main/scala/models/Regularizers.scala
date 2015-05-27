package models

import breeze.linalg.Vector
import vectors.LazyMappedVector
import vectors.VectorOps._

import scala.math._

class ElasticNet(eta: Double) extends Regularizer {

  require(0 < eta && eta <= 1)

  val p = (1-eta)/(2*eta)

  def primal(w: Vector[Double]) = eta*l2norm(w) + (1-eta)*l1norm(w)

  def dualGradient(v: Vector[Double]) = ???
//  {
//    new LazyMappedVector(v, {case(i,x) => if (abs(x) > p) x - p*signum(x) else 0.0})
//  }

  override def toString = s"$eta |w|^2_2 + ${1-eta} |w|_1"
}

//class L1Regularizer(sigma: Double, epsilon: Double) extends Regularizer {
//
//  require(sigma > 0.0 && epsilon > 0.0)
//
//  val B = 1.0 / sigma
//  val lambda = epsilon / (B * B)
//  val p = sigma / lambda
//
//  def primal(w: Vector[Double]) = 1.0/2*l2norm(w) + p*l1norm(w)
//
//  def dualGradient(v: Vector[Double]) = {
//    new LazyMappedVector(v, {case(i,x) => if (abs(x) > p) x - p*signum(x) else 0.0})
//  }
//
//  override def toString = s"${0.5*lambda} |w|^2_2 + $sigma |w|_1"
//}

class L2Regularizer extends Regularizer {

  def primal(w: Vector[Double]) = l2norm(w) / 2
  def dualGradient(w: Vector[Double]) = w

  def dual2(v: Vector[Double]) = l2norm(v)/2
}