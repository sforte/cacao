package localsolvers

import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 29/03/15.
 */
trait PrimalLocalSolverTrait extends Serializable {
  def optimize(
    data: Array[SparseClassificationPoint],
    wOld: Array[Double],
    wBar: Array[Double],
    seed : Int = 0): Array[Double]
}
