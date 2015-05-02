package models

import math.{log,exp}

/*
  Logistic regression classification model
 */
case class LogisticRegressionModel(n: Long, var regularizer: Regularizer)
  extends DualModelWithSecondDerivative {

  def primalLoss = LogisticLoss

  def dualLoss = LogisticLossConjugate
}

object LogisticLoss extends Loss {
  def apply(y: Double, x: Double) = log(1 + exp(-x*y))
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object LogisticLossConjugate extends DoublyDifferentiableLoss {
  def apply(y: Double, a: Double) = {
    if (a == 0 || y*a == -1) 0.0
    else (1+a*y)*log(1+a*y) - a*y*log(-a*y)
  }
  def derivative = LogisticLossConjugateDerivative
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object LogisticLossConjugateDerivative extends DifferentiableLoss {
  def apply(y: Double, a: Double): Double = y*(log(1+a*y) - log(-a*y))
  def derivative = LogisticLossConjugateSecondDerivative
  def domain(y: Double) = LogisticLossConjugate.domain(y)
}

object LogisticLossConjugateSecondDerivative extends Loss {
  def apply(y: Double, a: Double) = y*(1/a - y/(1+y*a))
  def domain(y: Double) = LogisticLossConjugate.domain(y)
}