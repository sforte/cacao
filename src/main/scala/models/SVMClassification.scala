package models

import distopt.utils.VectorOps._
import localsolvers.SingleCoordinateOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/*
  SVM Classification model
 */
case class SVMClassificationModel(n: Long, var regularizer: Regularizer)
  extends DualModelWithFirstDerivative {

  def primalLoss = HingeLoss

  def dualLoss = HingeLossConjugate
}

/*
  An ad-hoc single coordinate optimizer for SVM; the optimization is
  solvable in closed form.
 */
class SVMOptimizer extends SingleCoordinateOptimizer[SVMClassificationModel] {
  def optimize(model: SVMClassificationModel, pt: LabeledPoint, alpha: Double, v: Vector, epsilon: Double = 0.0) = {

    val lambda = model.regularizer.lambda
    val n = model.n
    val w = model.regularizer.dualGradient(v)

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

object HingeLoss extends Loss {
  def apply(y: Double, x: Double) = math.max(0, 1 - x*y)
  def domain(y: Double) = (Double.NegativeInfinity, Double.PositiveInfinity)
}

object HingeLossConjugate extends DifferentiableLoss {
  def apply(y: Double, a: Double) = y*a
  def derivative = HingeLossConjugateDerivative
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}

object HingeLossConjugateDerivative extends Loss {
  def apply(y: Double, a: Double): Double = -y
  def domain(y: Double) = if (y == 1) (-1,0) else (0,1)
}