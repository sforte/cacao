package localsolvers

import distopt.utils.VectorOps._
import models.DualModel
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Derivative free method to do descent on a local primal problem
 */
class PrimalOptimizer [-ModelType<:DualModel] (numIter: Int)
  extends SingleCoordinateOptimizer[ModelType] {

  /**
   * @param pt Point of which we wish to do coordinate ascent
   * @param alpha Old value of alpha
   * @param w Old value of w
   * @return Delta alpha
   */

  override def optimize(model: ModelType, pt: LabeledPoint, alpha: Double, w: Vector) = {

    val n = model.n
    val lambda = model.lambda
    val primalLoss = model.primalLoss

    val x = pt.features
    val y = pt.label

    val (xx,wx) = (dot(x,x),dot(w,x))

    val func = new UnivariateRealFunction {
      def value(deltaA: Double) =
        primalLoss(y, wx + deltaA/(lambda*n) * xx) + 1.0/(2*lambda*n)*math.pow(alpha + deltaA,2)*xx
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(numIter)
    brent.setRelativeAccuracy(0.0001)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    val deltaAlpha = brent.optimize(func, GoalType.MINIMIZE, -1000, 1000, 0)

    (deltaAlpha, times(x, deltaAlpha/(lambda*n)))
  }
}
