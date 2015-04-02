package localsolvers

import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 22/03/15.
 */
object LogisticRegression {

  class LogitLoss(y: Double) extends RealFunction {
    override def apply(x: Double): Double = math.log(1 + math.exp(-x*y))
  }

  class LogitLossConjugate(y: Double) extends RealFunction {
    override def apply(a: Double): Double = {
      if(a >= -1 && a <= 1 && y*a <= 0) {
        -a*y*math.log(-a*y) + (1+a*y)*math.log(1+a*y)
      } else {
        Double.PositiveInfinity
      }
    }
  }

  class LogitLossConjugateDerivative(y: Double) extends RealFunction {
    override def apply(a: Double): Double = {
      y*(math.log(-a*y) - math.log(1+a*y))
    }
  }
}
