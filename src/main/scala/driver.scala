package distopt

import optimizers.distributed._
import models.LogisticLoss
import models._
import utils.OptUtils
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
    val lambda = options.getOrElse("lambda", "0.01").toDouble
    val numRounds = options.getOrElse("numRounds", "200").toInt
    val numPasses = options.getOrElse("numPasses", "1").toInt

    println("master:       " + master)
    println("trainFile:    " + trainFile)
    println("numFeatures:  " + numFeatures)
    println("numSplits:    " + numSplits)
    println("lambda:       " + lambda)
    println("numRounds:    " + numRounds)
    println("numPasses:    " + numPasses)

    // setting up Spark
    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    implicit val sc = new SparkContext(conf)

    val data = OptUtils.loadLibSVMFile(sc, trainFile, numFeatures, numSplits).repartition(numSplits)

    val n = data.count()

    val model = new Model(n, lambda, new LogisticLoss, new L2Regularizer)

    CoCoA.optimize(model, data)

    sc.stop()
   }
}