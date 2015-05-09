package models

import breeze.linalg.Vector

trait RealFunction extends (Double => Double) with Serializable {
  def domain: (Double,Double)
}

trait DifferentiableRealFunction extends RealFunction {
  def derivative: RealFunction
}

trait DoublyDifferentiableRealFunction extends DifferentiableRealFunction {
  def derivative: DifferentiableRealFunction
}

/*
  A function that maps a label y and a value x to a loss value.
  Used both to represent a primal and a dual loss.
 */
trait Loss[+P<:RealFunction,+D<:RealFunction] extends (Double => P) with Serializable {
  def conjugate: Loss[D,P]
}

trait Regularizer extends Serializable {
  def primal(w: Vector[Double]): Double
  def dualGradient(w: Vector[Double]): Vector[Double]
  def lambda: Double

  def dual(v: Vector[Double]) = {
    val w = dualGradient(v)
    (w dot v) - primal(w)
  }
}