package models

import breeze.linalg.Vector
import optimizers.SingleCoordinateOptimizer
import vectors.{LazyScaledVector, LabelledPoint}

/*
  SVM Classification model
 */
case class SVMClassificationModel(n: Long, var regularizer: Regularizer)
  extends DualModelWithFirstDerivative {

  def primalLoss = HingeLoss

  def dualLoss = HingeLossConjugate
}



object HingeLoss extends Loss {
  def apply(y: Double, x: Double) = math.max(0, 1 - x*y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object HingeLossConjugate extends DifferentiableLoss {
  def apply(y: Double, a: Double) = y*a
  def derivative = HingeLossConjugateDerivative
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object HingeLossConjugateDerivative extends Loss {
  def apply(y: Double, a: Double): Double = -y
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}