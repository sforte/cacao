package localsolvers

import distopt.utils.{SparseVector, SparseClassificationPoint}

/**
 * Created by simone on 22/03/15.
 */
object SVMClassification {
  class SVMOptimizer(lambda: Double, n: Int) extends SingleCoordinateOptimizerTrait {
    def optimize(pt: SparseClassificationPoint, alpha: Double, w: Array[Double]): Double = {

      val x = pt.features
      val y = pt.label

      val norm = x.dot(x)

      var alphaNew = ((y - x.dot(w))*(lambda*n) + alpha*norm) / norm

      if (y == +1)
        alphaNew = math.min(math.max(alphaNew, 0), 1)
      else
        alphaNew = math.min(math.max(alphaNew, -1), 0)

      alphaNew - alpha
    }
  }

  class HingeLoss(y: Double) extends RealFunction {
    override def apply(x: Double): Double = math.max(0, 1 - x*y)
  }

  class HingeLossGrad(y: Double, x: SparseClassificationPoint) {
    def apply(w: Array[Double]) {
      if (x.features.dot(w) <= 1)
        x.features.times(-y)
      else
        x.features.times(0)
    }
  }

  class HingeLossConjugate(y: Double) extends RealFunction {
    override def apply(a: Double): Double = {
      if(a >= -1 && a <= 1 && y*a <= 0)
        y*a
      else
        Double.PositiveInfinity
    }
  }

  class HingeLossConjugateDerivative(y: Double) extends RealFunction {
    override def apply(a: Double): Double = -y
  }
}
