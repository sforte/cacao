package localsolvers

import models.{PrimalModel, DualModel}
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint

trait PrimalDualSolver[-ModelType<:DualModel, -DataType, AlphasType, WType] extends Serializable {

  def optimize(model: ModelType, data: DataType, alpha: AlphasType, w: WType): (AlphasType,WType)
}

trait SingleCoordinateOptimizer[-ModelType<:DualModel] extends PrimalDualSolver[ModelType, LabeledPoint, Double, Vector]

trait LocalOptimizer [-ModelType<:DualModel] extends PrimalDualSolver[ModelType, Array[LabeledPoint], DenseVector, DenseVector]

trait PrimalSolver[-ModelType<:PrimalModel, -DataType, WType] extends Serializable