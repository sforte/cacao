package localsolvers

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * An abstraction for a local dual solver to be used by CoCoA
 */
trait LocalSolverTrait extends Serializable {

  /**
   * @param data Data in the local partition of the solver
   * @param wInit Old value of w
   * @param alpha Old value of the alphas for the points in the partition
   * @param seed ...
   * @return
   */
  def optimize(
    data: Array[LabeledPoint],
    wInit: DenseVector,
    alpha: DenseVector,
    seed : Int = 0): (DenseVector,DenseVector)
}
