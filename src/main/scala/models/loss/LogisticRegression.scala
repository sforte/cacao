package models.loss

import models.{DifferentiableRealFunction, DoublyDifferentiableRealFunction, Loss, RealFunction}

import scala.math.{exp, log}

class LogisticLoss extends Loss[RealFunction,DoublyDifferentiableRealFunction] {
  def apply(y: Double) = new LogisticLossPrimalFunction(y)
  def conjugate = new LogisticLossConjugate
}

class LogisticLossConjugate extends Loss[DoublyDifferentiableRealFunction,RealFunction] {
  def apply(y: Double) = new LogisticLossConjugateFunction(y)
  def conjugate = new LogisticLoss
}

class LogisticLossPrimalFunction(y: Double) extends RealFunction {
  def apply(x: Double) = log(1 + exp(-x*y))
  def domain = (Double.NegativeInfinity, Double.PositiveInfinity)
}

class LogisticLossConjugateFunction(y: Double) extends DoublyDifferentiableRealFunction {
  def apply(a: Double) = {
    if (a == 0 || y*a == -1) 0.0
    else (1+a*y)*log(1+a*y) - a*y*log(-a*y)
  }

  def derivative = new LogisticLossConjugateDerivative(y)
  def domain = if (y == 1) (-1,0) else (0,1)
}

class LogisticLossConjugateDerivative(y: Double) extends DifferentiableRealFunction {
  def apply(a: Double): Double = y*log(1+a*y) - y*log(-a*y)
  def derivative = new LogisticLossConjugateSecondDerivative(y)
  def domain = if (y == 1) (-1,0) else (0,1)
}

class LogisticLossConjugateSecondDerivative(y: Double) extends RealFunction {
  def apply(a: Double) = y/(1+y*a) + log(1+a*y) - y/a - log(-a*y)
  def domain = if (y == 1) (-1,0) else (0,1)
}