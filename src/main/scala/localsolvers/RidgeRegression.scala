package localsolvers


import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 22/03/15.
 */
object RidgeRegression {

  /**
   * Created by simone on 20/03/15.
   */
  class RidgeOptimizer(lambda: Double, n: Int) extends SingleCoordinateOptimizerTrait {
    override def optimize(pt: SparseClassificationPoint, alpha: Double, w: Array[Double]): Double = {

      val x = pt.features
      val y = pt.label

      (- x.dot(w) - alpha/2 + y) / (x.dot(x)/(lambda*n) + 1.0/2)
    }
  }

  class RidgeLoss(y: Double) extends RealFunction {
    override def apply(x: Double): Double = (x - y) * (x - y)
  }

  class RidgeLossConjugate(y: Double) extends RealFunction {
    override def apply(a: Double): Double = a*a/4 + a*y
  }

  class RidgeLossConjugateDerivative(y: Double) extends RealFunction {
    override def apply(a: Double): Double = -a/2 - y
  }
}
