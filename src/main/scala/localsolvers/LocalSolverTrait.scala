package localsolvers

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
/**
 * Created by simone on 18/03/15.
 */
trait LocalSolverTrait extends Serializable {

  def optimize(
    data: Array[LabeledPoint],
    wInit: DenseVector,
    alpha: DenseVector,
    seed : Int = 0): (DenseVector,DenseVector)
}
