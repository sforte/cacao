package localsolvers

import distopt.utils.SparseClassificationPoint

/**
 * Created by simone on 18/03/15.
 */
trait LocalSolverTrait extends Serializable {

  def optimize(
    data: Array[SparseClassificationPoint],
    wInit: Array[Double],
    alpha: Array[Double],
    seed : Int = 0): (Array[Double],Array[Double])
}
