package oldstuff

import models.{Loss, RealFunction, Regularizer,Model}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.rdd.RDD
import utils.OptUtils
import vectors.{LabelledPoint, VectorOps}

object MllibLogisticWithL1 {

  def run (data: RDD[LabelledPoint], model: Model[Loss[RealFunction,RealFunction]], iter: Int) {

    val data01 = data.map(p => new LabelledPoint(if(p.label==1) 1 else 0,p.features))
                    .map(_.toMllib)

    val lambda = model.lambda

    var weights : Vector = null

    val steps = 100

    for (i <- 1 to iter/steps) {

      val logisticAlgo = new LogisticRegressionWithSGD().setIntercept(false).setValidateData(false)

      logisticAlgo.optimizer.setNumIterations(steps).setRegParam(lambda).setUpdater(new L1Updater)

      weights = if (weights == null)
        logisticAlgo.run(data01).weights
      else
        logisticAlgo.run(data01, weights).weights

      println(s"Iteration $i")
      OptUtils.printSummaryStatsFromW("Mllib", model,
        data.mapPartitions(x=>Iterator(x.toArray)), VectorOps.toBreeze(weights))
    }
  }
}
