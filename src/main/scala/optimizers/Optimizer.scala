package optimizers

import breeze.linalg.{DenseVector, Vector}
import models.{Regularizer, RealFunction, Loss}
import org.apache.spark.rdd.RDD
import vectors.LabelledPoint

trait Optimizer[-LossType<:Loss[_,_], DataType, AlphasType, VType] extends Serializable {

  def optimize(loss: LossType, reg: Regularizer, n: Long,
               data: DataType, alpha: AlphasType, v: VType): (AlphasType,VType)
}

trait SingleCoordinateOptimizer[-LossType<:Loss[_,_]]
  extends Optimizer[LossType, LabelledPoint, Double, Vector[Double]]

trait LocalOptimizer[-LossType<:Loss[_,_]]
  extends Optimizer[LossType, Array[LabelledPoint], DenseVector[Double], DenseVector[Double]]

trait DistributedOptimizer[-LossType<:Loss[_,_]]
  extends Optimizer[LossType, RDD[Array[LabelledPoint]], RDD[DenseVector[Double]], DenseVector[Double]]