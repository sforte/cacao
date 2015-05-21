package distopt

import breeze.linalg.DenseVector
import breeze.numerics.{abs, pow}
import models.loss.LogisticLoss
import models.regularizer. ElasticNet
import optimizers.coordinate.{BrentMethodOptimizer, BrentMethodOptimizerWithFirstDerivative}
import optimizers.local.SDCAOptimizer
import optimizers.distributed._
import models._
import org.apache.commons.math.analysis.UnivariateRealFunction
import org.apache.commons.math.optimization.GoalType
import org.apache.commons.math.optimization.univariate.BrentOptimizer
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

    /*
      Here on the dual we are solving:
        \sum (g_i(..)) + \lambda |w|_1
        and not:
        \frac{1}{d} \sum (g_i(..)) + \lambda |w|_1
        (so in other words the same as in the paper)
        where g_i is either the l2 loss or the logistic
        (you have to choose the right one inside the L1 class.
     */
    L1.optimize(sc, data, lambda)

    return

    val n = data.count()

    val partData = data.mapPartitions(x=>Iterator(x.toArray), preservesPartitioning = true)
    val partAlphas = data.map(_ => 0.0)
      .mapPartitions(x=>Iterator(new DenseVector(x.toArray)), preservesPartitioning = true)

    val v = DenseVector.zeros[Double](numFeatures)

    val loss = new LogisticLoss
//    val regularizer = new L1Regularizer(lambda, 0.01)
    val regularizer = new ElasticNet(lambda, 0.001)
    println(regularizer)

//    val scOptimizer = new BrentMethodOptimizerWithFirstDerivative(sgdIterations*10)

    val scOptimizer = new BrentMethodOptimizer(sgdIterations*10)
    val localSolver = new SDCAOptimizer(scOptimizer, numPasses)

    val cocoa = new CoCoA(sc, localSolver, numRounds, beta, seed)

    cocoa.optimize(loss, regularizer, n, partData, partAlphas, v, 0.0)
//    acccocoa.optimize(loss, regularizer, n, partData, partAlphas, v, 0.0)

    MllibLogisticWithL1.run(data, loss, regularizer, n, 100000)

    sc.stop()
   }
}