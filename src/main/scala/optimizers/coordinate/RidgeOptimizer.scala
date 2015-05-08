package optimizers.coordinate

import breeze.linalg.Vector
import optimizers.SingleCoordinateOptimizer
import vectors.{LazyScaledVector, LabelledPoint}
import models.RidgeRegressionModel

/*
  An ad-hoc single coordinate optimizer for Ridge; the optimization is
  solvable in closed form.
 */
class RidgeOptimizer extends SingleCoordinateOptimizer[RidgeRegressionModel] {
  def optimize(model: RidgeRegressionModel, pt: LabelledPoint, alpha: Double, v: Vector[Double], epsilon: Double = 0.0) = {

    val n = model.n
    val lambda = model.regularizer.lambda
    val w : Vector[Double] = model.regularizer.dualGradient(v)

    val x = pt.features
    val y = pt.label

    val (xx, xw) = (x dot x, x dot w)

    val deltaAlpha = (-xw - alpha/2 + y) / (xx/(lambda*n) + 1.0/2)

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}