package models

/*
  Logistic regression classification model
 */
case class LogisticRegressionModel(n: Long, lambda: Double) extends PrimalDualModelWithSecondDerivative with PrimalModel {

  def primalLoss = LogisticLoss

  def dualLoss = LogisticLossConjugate

  def initAlpha(y: Double) = 0.5*y

  def primalLossGradient = null
}

object LogisticLoss extends RealFunction {
  def apply(y: Double, x: Double) = math.log(1 + math.exp(-x*y))
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object LogisticLossConjugate extends DoublyDifferentiableRealFunction {
  def apply(y: Double, a: Double) = -a*y*math.log(-a*y) + (1+a*y)*math.log(1+a*y)
  def derivative = LogisticLossConjugate
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object LogisticLossConjugateDerivative extends DifferentiableRealFunction {
  def apply(y: Double, a: Double): Double = y*(math.log(-a*y) - math.log(1+a*y))
  def derivative = LogisticLossConjugateDerivative
  def domain(y: Double) = LogisticLossConjugate.domain(y)
}

object LogisticLossConjugateSecondDerivative extends RealFunction {
  def apply(y: Double, a: Double) = y*(1/a - y/(1+y*a))
  def domain(y: Double) = LogisticLossConjugate.domain(y)
}