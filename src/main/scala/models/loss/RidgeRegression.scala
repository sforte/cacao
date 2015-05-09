package models.loss

import models.{DifferentiableRealFunction, Loss, RealFunction}

class RidgeLoss extends Loss[RealFunction,DifferentiableRealFunction] {
  def apply(y: Double) = new RidgeLossPrimalFunction(y)
  def conjugate = new RidgeLossConjugate
}

class RidgeLossConjugate extends Loss[DifferentiableRealFunction,RealFunction] {
  def apply(y: Double) = new RidgeLossConjugateFunction(y)
  def conjugate = new RidgeLoss
}

class RidgeLossPrimalFunction(y: Double) extends RealFunction {
  def apply(x: Double) = (x - y) * (x - y)
  def domain = (Double.NegativeInfinity, Double.PositiveInfinity)
}

class RidgeLossConjugateFunction(y: Double) extends DifferentiableRealFunction {
  def apply(a: Double) = a*a/4 + a*y
  def derivative = new RidgeLossConjugateDerivative(y)
  def domain = (Double.NegativeInfinity, Double.PositiveInfinity)
}

class RidgeLossConjugateDerivative(y: Double) extends RealFunction {
  def apply(a: Double) = a/2 + y
  def domain = (Double.NegativeInfinity, Double.PositiveInfinity)
}