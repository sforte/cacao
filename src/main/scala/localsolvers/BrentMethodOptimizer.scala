package localsolvers

import distopt.utils.VectorOps._
import models.{DualModel, RealFunction}
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Derivative free method to optimize to do ascent on a single coordinate.
 */
class BrentMethodOptimizer [-ModelType<:DualModel] (numIter: Int)
  extends SingleCoordinateOptimizer[ModelType] {

  /**
   * @param pt Point of which we wish to do coordinate ascent
   * @param alpha Old value of alpha
   * @param w Old value of w
   * @return Delta alpha
   */

  override def optimize(model: ModelType, pt: LabeledPoint, alpha: Double, v: Vector) = {

    val n = model.n
    val lambda = model.lambda
    val dualLoss = model.dualLoss

    val x = pt.features
    val y = pt.label

//    val w = Vectors.zeros(x.size)
    val w = v

    val (ww,xx,xw) = (dot(w,w),dot(x,x),dot(x,w))

    // the function we wish to optimize on
    val func = new UnivariateRealFunction {
      def value(a: Double) =
        (lambda*n)/2*ww + (a-alpha)*xw + math.pow(a-alpha,2)/(2*lambda*n)*xx + dualLoss(y,-a)
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(numIter)
    brent.setRelativeAccuracy(0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    // the domain on which we optimize is determined by the domain of the conjugate loss function
    val domain = dualLoss.domain(-y)
    val alphaNew = brent.optimize(func, GoalType.MINIMIZE, domain._1, domain._2, alpha)

    val deltaAlpha = alphaNew - alpha

    (deltaAlpha, times(x, deltaAlpha/(lambda*n)))
//    (deltaAlpha, null)
  }
}
