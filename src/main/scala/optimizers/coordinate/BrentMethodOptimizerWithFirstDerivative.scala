package optimizers.coordinate

import breeze.linalg.Vector
import models.DualModelWithFirstDerivative
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.analysis.solvers.BrentSolver
import optimizers.SingleCoordinateOptimizer
import vectors.{LabelledPoint, LazyScaledVector}

/**
 * Derivative free method to optimize to do ascent on a single coordinate.
 */
class BrentMethodOptimizerWithFirstDerivative [-ModelType<:DualModelWithFirstDerivative] (numIter: Int = 100)
  extends SingleCoordinateOptimizer[ModelType] {

  /**
   * @param pt Point of which we wish to do coordinate ascent
   * @param alpha Old value of alpha
   * @param v Old value of v
   * @return Delta alpha
   */

  override def optimize(model: ModelType, pt: LabelledPoint, alpha: Double, v: Vector[Double], epsilon: Double = 0.0) = {

    val n = model.n
    val lambda = model.regularizer.lambda
    val dualLossDer = model.dualLoss.derivative

    val x = pt.features
    val y = pt.label

    val w : Vector[Double] = model.regularizer.dualGradient(v)

    val (xx,wx) = (x dot x, x dot w)

    // the function we wish to optimize on
    val func = new UnivariateRealFunction {
      def value(a: Double) =
        - wx - xx*(a-alpha)/(lambda*n) + dualLossDer(y,-a)
    }

    val brent = new BrentSolver
    brent.setMaximalIterationCount(numIter)
    brent.setRelativeAccuracy(0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    // the domain on which we optimize is determined by the domain of the conjugate loss function
    val domain = (-dualLossDer.domain(y)._2, -dualLossDer.domain(y)._1)
    val alphaNew = brent.solve(func, domain._1, domain._2, alpha)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}
