package optimizers.coordinate

import breeze.linalg.Vector
import models._
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.analysis.solvers.BrentSolver
import optimizers.SingleCoordinateOptimizer
import vectors.{LabelledPoint, LazyScaledVector}

/**
 * Derivative free method to optimize to do ascent on a single coordinate.
 */
class BrentMethodOptimizerWithFirstDerivative
  [-LossType<:Loss[RealFunction,DifferentiableRealFunction]] (numIter: Int = 100)
  extends SingleCoordinateOptimizer[LossType] {

  /**
   * @param pt Point of which we wish to do coordinate ascent
   * @param alpha Old value of alpha
   * @param v Old value of v
   * @return Delta alpha
   */

  override def optimize(model: Model[LossType], pt: LabelledPoint, alpha: Double, v: Vector[Double]) = {

    val lambda = model.lambda
    val n = model.n
    val regularizer = model.regularizer
    val loss = model.loss

    val x = pt.features
    val y = pt.label

    val dualLossDer = loss.conjugate(y).derivative

    val w : Vector[Double] = regularizer.dualGradient(v)

    val (xx,wx) = (x dot x, x dot w)

    // the function we wish to optimize on
    val func = new UnivariateRealFunction {
      def value(a: Double) =
        - wx - xx*(a-alpha)/(lambda*n) + dualLossDer(-a)
    }

    val brent = new BrentSolver
    brent.setMaximalIterationCount(numIter)
    brent.setAbsoluteAccuracy(4*math.ulp(1d))

    // the domain on which we optimize is determined by the domain of the conjugate loss function
    val domain = (-dualLossDer.domain._2, -dualLossDer.domain._1)

    val alphaNew = if(math.abs(domain._1-domain._2) <= 10*math.ulp(1d))
      alpha
    else
      brent.solve(func, domain._1, domain._2, alpha)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}
