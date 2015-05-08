package models

import breeze.linalg.Vector
import optimizers.SingleCoordinateOptimizer
import vectors.{LazyScaledVector, LabelledPoint}

/*
  Ridge regression model
 */
case class RidgeRegressionModel(n: Long, var regularizer: Regularizer)
  extends DualModelWithFirstDerivative {

  def primalLoss = RidgeLoss
  def dualLoss = RidgeLossConjugate
}

object RidgeLoss extends Loss {
  def apply(y: Double, x: Double) = (x - y) * (x - y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugate extends DifferentiableLoss {
  def apply(y: Double, a: Double) = a*a/4 + a*y
  def derivative = RidgeLossConjugateDerivative
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugateDerivative extends Loss {
  def apply(y: Double, a: Double): Double = -a/2 - y
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}