package optimizers.coordinate

import breeze.linalg.Vector
import models.{Model, Loss, RealFunction}
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import optimizers.SingleCoordinateOptimizer
import vectors.{LabelledPoint, LazyScaledVector}

/**
 * Derivative free method to do descent on a local primal problem
 */
class PrimalOptimizer [-LossType<:Loss[RealFunction,_]] (numIter: Int)
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

    val primalLoss = loss(y)
    val w : Vector[Double] = regularizer.dualGradient(v)

    val (xx,wx) = (x dot x, w dot x)

    val func = new UnivariateRealFunction {
      def value(deltaA: Double) =
        primalLoss(wx + deltaA/(lambda*n) * xx) + 1.0/(2*lambda*n)*math.pow(alpha + deltaA,2)*xx
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(numIter)
//    brent.setRelativeAccuracy(0.0001)
    brent.setRelativeAccuracy(0.0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    val deltaAlpha = brent.optimize(func, GoalType.MINIMIZE, -1000, 1000, 0)

    (deltaAlpha, new LazyScaledVector(x, deltaAlpha/(lambda*n)))
  }
}
