package models

import distopt.utils.VectorOps._
import localsolvers.SingleCoordinateOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/*
  SVM Classification model
 */
case class SVMClassificationModel(n: Long, lambda: Double) extends PrimalDualModelWithFirstDerivative with PrimalModel {

  def primalLoss = HingeLoss

  def dualLoss = HingeLossConjugate

  def primalLossGradient = null

  def initAlpha(y: Double) = 0.5*y
}

/*
  An ad-hoc single coordinate optimizer for SVM; the optimization is
  solvable in closed form.
 */
class SVMOptimizer extends SingleCoordinateOptimizer[SVMClassificationModel] {
  def optimize(model: SVMClassificationModel, pt: LabeledPoint, alpha: Double, w: Vector) = {

    val lambda = model.lambda
    val n = model.n

    val x = pt.features
    val y = pt.label

    val norm = dot(x,x)

    var alphaNew = ((y - dot(x,w))*(lambda*n) + alpha*norm) / norm

    if (y == +1)
      alphaNew = math.min(math.max(alphaNew, 0), 1)
    else
      alphaNew = math.min(math.max(alphaNew, -1), 0)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, times(x, deltaAlpha/(lambda*n)))
  }
}

object HingeLoss extends RealFunction {
  def apply(y: Double, x: Double) = math.max(0, 1 - x*y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object HingeLossConjugate extends DifferentiableRealFunction {
  def apply(y: Double, a: Double) = y*a
  def derivative = HingeLossConjugateDerivative
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object HingeLossConjugateDerivative extends RealFunction {
  def apply(y: Double, a: Double): Double = -y
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}