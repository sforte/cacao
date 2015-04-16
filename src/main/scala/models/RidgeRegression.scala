package models

import distopt.utils.VectorOps._
import localsolvers.SingleCoordinateOptimizerTrait
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

case class RidgeRegressionModel(lambda: Double) extends PrimalDualModel {

  def primalLoss = LogisticLoss

  def dualLoss = LogisticLossConjugate

  def initAlpha(y: Double) = 0.0
}

class RidgeOptimizer(lambda: Double, n: Int) extends SingleCoordinateOptimizerTrait {
  def optimize(pt: LabeledPoint, alpha: Double, w: DenseVector): Double = {

    val x = pt.features
    val y = pt.label

    (-dot(x,w) - alpha/2 + y) / (dot(x,x)/(lambda*n) + 1.0/2)
  }
}

object RidgeLoss extends RealFunction {
  def apply(y: Double, x: Double) = (x - y) * (x - y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugate extends RealFunction {
  def apply(y: Double, a: Double) = a*a/4 + a*y
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object RidgeLossConjugateDerivative extends RealFunction {
  def apply(y: Double, a: Double): Double = -a/2 - y
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}