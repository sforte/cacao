package optimizers.coordinate

import breeze.linalg.Vector
import optimizers.SingleCoordinateOptimizer
import vectors.{LabelledPoint, LazyScaledVector}
import models.SVMClassificationModel

/**
 * Created by simone on 08/05/15.
 */
/*
  An ad-hoc single coordinate optimizer for SVM; the optimization is
  solvable in closed form.
 */
class SVMOptimizer extends SingleCoordinateOptimizer[SVMClassificationModel] {
  def optimize(model: SVMClassificationModel, pt: LabelledPoint, alpha: Double, v: Vector[Double], epsilon: Double = 0.0) = {

    val lambda = model.regularizer.lambda
    val n = model.n
    val w : Vector[Double] = model.regularizer.dualGradient(v)

    val x = pt.features
    val y = pt.label

    val (xx,xw) = (x dot x, x dot w)

    var alphaNew = ((y - xw)*(lambda*n) + alpha*xx) / xx

    if (y == +1)
      alphaNew = math.min(math.max(alphaNew, 0), 1)
    else
      alphaNew = math.min(math.max(alphaNew, -1), 0)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}
