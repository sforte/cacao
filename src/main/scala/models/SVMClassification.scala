package models

import distopt.utils.VectorOps._
import localsolvers.SingleCoordinateOptimizerTrait
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

case class SVMClassificationModel(lambda: Double) extends PrimalDualModel {

  def primalLoss = HingeLoss

  def dualLoss = HingeLossConjugate

  def initAlpha(y: Double) = 0.5*y
}

class SVMOptimizer(lambda: Double, n: Int) extends SingleCoordinateOptimizerTrait {
  def optimize(pt: LabeledPoint, alpha: Double, w: DenseVector): Double = {

    val x = pt.features
    val y = pt.label

    val norm = dot(x,x)

    var alphaNew = ((y - dot(x,w))*(lambda*n) + alpha*norm) / norm

    if (y == +1)
      alphaNew = math.min(math.max(alphaNew, 0), 1)
    else
      alphaNew = math.min(math.max(alphaNew, -1), 0)

    alphaNew - alpha
  }
}

object HingeLoss extends RealFunction {
  def apply(y: Double, x: Double) = math.max(0, 1 - x*y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object HingeLossConjugate extends RealFunction {
  def apply(y: Double, a: Double) = y*a
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object HingeLossConjugateDerivative extends RealFunction {
  def apply(y: Double, a: Double): Double = -y
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}