package optimizers.coordinate

import breeze.linalg.Vector
import models.{Model, Regularizer, Loss, RealFunction}
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import optimizers.SingleCoordinateOptimizer
import vectors.{LabelledPoint, LazyScaledVector}

/**
 * Derivative free method to optimize to do ascent on a single coordinate.
 */
class BrentMethodOptimizer [-LossType<:Loss[_,RealFunction]] (numIter: Int = 100)
  extends SingleCoordinateOptimizer[LossType] {

  /**
   * @param pt Point of which we wish to do coordinate ascent
   * @param alpha Old value of alpha
   * @param v Old value of w
   * @return Delta alpha
   */

  override def optimize (model: Model[LossType], pt: LabelledPoint, alpha: Double, v: Vector[Double]) = {

    val lambda = model.lambda
    val n = model.n
    val regularizer = model.regularizer
    val loss = model.loss

    val y = pt.label
    val x = pt.features

    val dualLoss = model.loss.conjugate(y)

    val w = regularizer.dualGradient(v)

    val (xx, xw) = (x dot x, x dot w)

    // the function we wish to optimize on
    val func = new UnivariateRealFunction {
      def value(a: Double) =
        - (a-alpha)*xw - xx*math.pow(a-alpha,2)/(2*lambda*n) - dualLoss(-a)
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(numIter)
    brent.setRelativeAccuracy(0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    // the domain on which we optimize is determined by the domain of the conjugate loss function
    val domain = (-dualLoss.domain._2, -dualLoss.domain._1)
    val alphaNew = brent.optimize(func, GoalType.MAXIMIZE, domain._1, domain._2, alpha)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}