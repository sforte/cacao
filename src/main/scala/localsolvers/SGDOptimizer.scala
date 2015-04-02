package localsolvers

import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 22/03/15.
 */
class SGDOptimizer(dualLoss: RealFunction, dualLossGradient: RealFunction, numIter: Int, lambda: Double, n: Int)
  extends SingleCoordinateOptimizerTrait {

  override def optimize(pt: SparseClassificationPoint, alpha: Double, w: Array[Double]): Double = {

    val x = pt.features
    val y = pt.label

    var alphaNew = if (y == +1) 0.5 else -0.5

    val xx = x.dot(x)
    val xw = x.dot(w)

    for (i <- 1 to numIter) {

      val gradient = xw + xx*(alphaNew-alpha)/(lambda*n) + dualLossGradient(-alphaNew)

      val alphaNewNew = alphaNew - 1.0/(i) * gradient

      if (dualLoss(-alphaNewNew) < Double.PositiveInfinity) {
        alphaNew = alphaNewNew
      }

//      if (y == +1 && alphaNewNew > 0 && alphaNewNew < 1)
//        alphaNew = alphaNewNew
//      else if (y == -1 && alphaNewNew > -1 && alphaNewNew < 0)
//        alphaNew = alphaNewNew
    }

    alphaNew - alpha
  }
}
