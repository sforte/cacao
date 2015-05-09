package models.loss

import models.{DifferentiableRealFunction, RealFunction, Loss}

class HingeLoss extends Loss[HingeLossPrimalFunction,HingeLossConjugateFunction] {
  def apply(y: Double) = new HingeLossPrimalFunction(y)
  def conjugate = new HingeLossConjugate
}

class HingeLossConjugate extends Loss[HingeLossConjugateFunction,HingeLossPrimalFunction] {
  def apply(y: Double) = new HingeLossConjugateFunction(y)
  def conjugate = new HingeLoss
}

class HingeLossPrimalFunction(y: Double) extends RealFunction {
  def apply(x: Double) = math.max(0, 1 - x*y)
  def domain = (Double.NegativeInfinity, Double.PositiveInfinity)
}

class HingeLossConjugateFunction(y: Double) extends DifferentiableRealFunction {
  def apply(a: Double) = y*a
  def derivative = new HingeLossConjugateDerivative(y)
  def domain = if (y == 1) (-1,0) else (0,1)
}

class HingeLossConjugateDerivative(y: Double) extends RealFunction {
  def apply(a: Double) = y
  def domain = if (y == 1) (-1,0) else (0,1)
}