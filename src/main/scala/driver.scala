package distopt

import distopt.solvers._
import localsolvers._
import models.LogisticRegressionModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

object driver {

  def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => opt -> v
        case Array(opt) => opt -> "true"
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits","1").toInt
    val testFile = options.getOrElse("testFile", "")
    val lambda = options.getOrElse("lambda", "0.01").toDouble
    val numRounds = options.getOrElse("numRounds", "200").toInt
    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble
    val beta = options.getOrElse("beta","1.0").toDouble
    val sgdIterations = options.getOrElse("sgdIterations","100").toInt
    val seed = options.getOrElse("seed","0").toInt

    println("master:       " + master);          println("trainFile:    " + trainFile)
    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits)
    println("testfile:     " + testFile);        println("lambda:       " + lambda)
    println("numRounds:    " + numRounds);       println("localIterFrac:" + localIterFrac)
    println("beta          " + beta);            println("beta          " + beta)
    println("seed          " + seed);            println("sgdIterations " + sgdIterations)

    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, trainFile, numFeatures, numSplits).repartition(numSplits)
    val n = data.count()

    val model = new LogisticRegressionModel(lambda)

    val scOptimizer = new BrentMethodOptimizer(model.dualLoss, sgdIterations, lambda, n)

    val localIters = Math.max((localIterFrac * n / data.partitions.size).toInt,1)
    val localSolver = new SDCASolver(scOptimizer, localIters, lambda, n)

    CoCoA.runCoCoA(sc, data, model, localSolver, numRounds, beta, seed)

    sc.stop()
   }
}