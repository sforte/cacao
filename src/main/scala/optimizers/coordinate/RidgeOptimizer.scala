package optimizers.coordinate

import breeze.linalg.Vector
import models.loss.RidgeLoss
import optimizers.SingleCoordinateOptimizer
import vectors.{LazyScaledVector, LabelledPoint}
import models.Regularizer

/*
  An ad-hoc single coordinate optimizer for Ridge; the optimization is
  solvable in closed form.
 */
class RidgeOptimizer extends SingleCoordinateOptimizer[RidgeLoss] {
  def optimize(model: RidgeLoss, regularizer: Regularizer, n: Long,
               pt: LabelledPoint, alpha: Double, v: Vector[Double], epsilon: Double = 0.0) = {

    val lambda = regularizer.lambda
    val w : Vector[Double] = regularizer.dualGradient(v)

    val x = pt.features
    val y = pt.label

    val (xx, xw) = (x dot x, x dot w)

    val deltaAlpha = (-xw - alpha/2 + y) / (xx/(lambda*n) + 1.0/2)

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}