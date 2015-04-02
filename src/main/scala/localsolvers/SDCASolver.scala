package localsolvers

import java.security.SecureRandom

import distopt.utils.SparseClassificationPoint
import distopt.utils.Implicits._

/**
 * Created by simone on 18/03/15.
 */
class SDCASolver(scOptimizers: Array[SingleCoordinateOptimizerTrait], localIters: Int, lambda: Double, n: Int)
  extends LocalSolverTrait {
  /**
   * @param localData the local data examples
   * @param wInit ...
   * @param alphaOld ...
   * @param seed ...
   * @return deltaAlpha and deltaW, summarizing the performed local changes, see paper
   */
  override def optimize(
    localData: Array[SparseClassificationPoint],
    wInit: Array[Double],
    alphaOld: Array[Double],
    seed : Int = 0) : (Array[Double],Array[Double]) = {

    val alpha = alphaOld.clone()

    var w = wInit
    val nLocal = localData.length
    val r = new SecureRandom
    r.setSeed(seed)

    var deltaW = Array.fill(wInit.length)(0.0)

    // perform local udpates
    for (i <- 1 to localIters) {
      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val pt = localData(idx)
      val x = pt.features

      val scDeltaAlpha = scOptimizers(idx).optimize(pt, alpha(idx), w)

      val update = x.times(scDeltaAlpha/(lambda*n))
//      w = update.plus(w)
//      deltaW = update.plus(deltaW)

      w += update
      deltaW += update
      alpha(idx) += scDeltaAlpha
    }

    val deltaAlpha = alphaOld.times(-1.0).plus(alpha)
    (deltaAlpha, deltaW)
  }
}
