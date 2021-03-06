package optimizers.coordinate

import breeze.linalg.Vector
import models.RidgeLoss
import optimizers.SingleCoordinateOptimizer
import vectors.{LazyScaledVector, LabelledPoint}
import models.Model

/*
  An ad-hoc single coordinate optimizer for Ridge; the optimization is
  solvable in closed form.
 */
class RidgeOptimizer extends SingleCoordinateOptimizer[RidgeLoss] {
  def optimize(model: Model[RidgeLoss], pt: LabelledPoint, alpha: Double, v: Vector[Double]) = {

    val lambda = model.lambda
    val n = model.n
    val regularizer = model.regularizer

    val w : Vector[Double] = regularizer.dualGradient(v)

    val x = pt.features
    val y = pt.label

    val (xx, xw) = (x dot x, x dot w)

    val deltaAlpha = (-xw - alpha/2 + y) / (xx/(lambda*n) + 1.0/2)

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}