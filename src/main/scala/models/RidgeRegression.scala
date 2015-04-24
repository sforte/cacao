package models

import distopt.utils.VectorOps._
import localsolvers.SingleCoordinateOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/*
  Ridge regression model
 */
case class RidgeRegressionModel(n: Long, lambda: Double) extends PrimalDualModelWithFirstDerivative with PrimalModel {

  def primalLoss = RidgeLoss

  def dualLoss = RidgeLossConjugate

  def initAlpha(y: Double) = 0.0

  def primalLossGradient: Gradient = null
}

/*
  An ad-hoc single coordinate optimizer for Ridge; the optimization is
  solvable in closed form.
 */
class RidgeOptimizer extends SingleCoordinateOptimizer[RidgeRegressionModel] {
  def optimize(model: RidgeRegressionModel, pt: LabeledPoint, alpha: Double, w: Vector): (Double, Vector) = {

    val n = model.n
    val lambda = model.lambda

    val x = pt.features
    val y = pt.label

    val deltaAlpha = (-dot(x,w) - alpha/2 + y) / (dot(x,x)/(lambda*n) + 1.0/2)

    (deltaAlpha, times(x, deltaAlpha/(lambda*n)))
  }
}

object RidgeLoss extends RealFunction {
  def apply(y: Double, x: Double) = (x - y) * (x - y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugate extends DifferentiableRealFunction {
  def apply(y: Double, a: Double) = a*a/4 + a*y
  def derivative = RidgeLossConjugateDerivative
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugateDerivative extends RealFunction {
  def apply(y: Double, a: Double): Double = -a/2 - y
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}