package localsolvers

import distopt.utils.VectorOps._
import models.RealFunction
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Created by simone on 22/03/15.
 */
class BrentMethodOptimizer(dualLoss: RealFunction, numIter: Int, lambda: Double, n: Long)
  extends SingleCoordinateOptimizerTrait {

  override def optimize(pt: LabeledPoint, alpha: Double, w: DenseVector): Double = {

    val x = pt.features
    val y = pt.label

    val (ww,xx,xw) = (dot(w,w),dot(x,x),dot(x,w))

    val func = new UnivariateRealFunction {
      def value(a: Double) =
        (lambda*n)/2*ww + (a-alpha)*xw + math.pow(a-alpha,2)/(2*lambda*n)*xx + dualLoss(y,-a)
    }

    val brent = new BrentOptimizer
    brent.setMaximalIterationCount(numIter)
    brent.setRelativeAccuracy(0)
    brent.setAbsoluteAccuracy(4*Math.ulp(1d))

    val domain = dualLoss.domain(-y)
    val alphaNew = brent.optimize(func, GoalType.MINIMIZE, domain._1, domain._2, alpha)

    alphaNew - alpha
  }
}
