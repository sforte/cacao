package distopt

import models.regularizer.L1Regularizer
import optimizers.coordinate.BrentMethodOptimizerWithFirstDerivative
import optimizers.local.SDCAOptimizer
import optimizers.distributed.{MllibLogisticWithL1, AccCoCoA}
import models._
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionWithSGD, SVMModel}
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.{LassoModel, LassoWithSGD}
import utils.OptUtils
import vectors.LabelledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object driver {

  def main(args: Array[String]) {

    //  parsing command-line options
    val options = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => opt -> v
        case Array(opt) => opt -> "true"
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits", "1").toInt
    val testFile = options.getOrElse("testFile", "")
    val lambda = options.getOrElse("lambda", "0.01").toDouble
    val numRounds = options.getOrElse("numRounds", "200").toInt
    val numPasses = options.getOrElse("numPasses", "1").toInt
    val beta = options.getOrElse("beta", "1.0").toDouble
    val sgdIterations = options.getOrElse("sgdIterations", "100").toInt
    val seed = options.getOrElse("seed", "0").toInt

    println("master:       " + master)
    println("trainFile:    " + trainFile)
    println("numFeatures:  " + numFeatures)
    println("numSplits:    " + numSplits)
    println("testfile:     " + testFile)
    println("lambda:       " + lambda)
    println("numRounds:    " + numRounds)
    println("numPasses:    " + numPasses)
    println("beta          " + beta)
    println("seed          " + seed)
    println("sgdIterations " + sgdIterations)

    // setting up Spark
    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    val data = OptUtils.loadLibSVMFile(sc, trainFile, numFeatures, numSplits).repartition(numSplits)
    val n = data.count()

//    val model = new LogisticRegressionModel(n, new L2Regularizer(lambda))
    val model = new LogisticRegressionModel(n, new L1Regularizer(lambda, 0.01))
//    val model = new SVMClassificationModel(n, new L2Regularizer(lambda))
//    val model = new SVMClassificationModel(n, new L1Regularizer(lambda, epsilon = 0.0001))
//    val model = new RidgeRegressionModel(n, new L2Regularizer(lambda))
//    val model = new RidgeRegressionModel(n, new L1Regularizer(lambda, epsilon = 0.0001))

//    val scOptimizer = new SVMOptimizer
    val scOptimizer = new BrentMethodOptimizerWithFirstDerivative(sgdIterations*10)
//    val scOptimizer = new RidgeOptimizer
//    val scOptimizer = new PrimalOptimizer(sgdIterations*10)
    val localSolver = new SDCAOptimizer(scOptimizer, numPasses)

//    CoCoA.runCoCoA(sc, data, model, localSolver, numRounds, beta, seed)
    AccCoCoA.runCoCoA(sc, data, model, localSolver, numRounds, beta, seed)

    MllibLogisticWithL1.run(data, lambda, model, 100000)

    sc.stop()
   }
}