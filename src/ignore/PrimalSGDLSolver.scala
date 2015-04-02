package localsolvers

import java.security.SecureRandom
import distopt.utils.SparseClassificationPoint
import distopt.utils.Implicits.arraytoDoubleArray

/**
 * Created by simone on 29/03/15.
 */
class PrimalSGDLSolver(n: Int, lambda: Double, numIter: Int) extends PrimalLocalSolverTrait {
  override def optimize(data: Array[SparseClassificationPoint], wOld: Array[Double], wBar: Array[Double], seed: Int): Array[Double] = {

    val nLocal = data.length
//    var w : Array[Double] = wBar.map(x => 0.0)
    var w : Array[Double] = wOld

    val rand = new SecureRandom
    rand.setSeed(seed)

    for (i <- 1 to 100000) {

      val idx = rand.nextInt(nLocal)

      val x = data(idx).features
      val y = data(idx).label

      val lossGrad = if (x.dot(w.plus(wBar)) * y <= 1) x.times(-y) else x.times(0)
//      val lossGrad = if (x.dot(w) * y <= 1) x.times(-y) else x.times(0)

//      val lossGrad = x.times(2*(x.dot(w.plus(wBar)) - y))

      val grad = lossGrad.times(1.0/n).plus(w.times(lambda/nLocal))

      w = w.plus(grad.times(-1.0/i))
    }
    w
  }
}
