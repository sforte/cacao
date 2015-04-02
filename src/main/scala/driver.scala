package distopt

import localsolvers.LogisticRegression.{LogitLoss, LogitLossConjugate, LogitLossConjugateDerivative}
import localsolvers.RidgeRegression.{RidgeLossConjugateDerivative, RidgeLossConjugate, RidgeLoss, RidgeOptimizer}
import localsolvers.SVMClassification.{HingeLossConjugateDerivative, SVMOptimizer, HingeLossConjugate, HingeLoss}
import localsolvers._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import distopt.utils._
import distopt.solvers._

object driver {

  def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => opt -> v
        case Array(opt) => opt -> "true"
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits","1").toInt
    val chkptDir = options.getOrElse("chkptDir","")
    var chkptIter = options.getOrElse("chkptIter","100").toInt
    val testFile = options.getOrElse("testFile", "")
    val justCoCoA = options.getOrElse("justCoCoA", "true").toBoolean

    val lambda = options.getOrElse("lambda", "0.01").toDouble
    val numRounds = options.getOrElse("numRounds", "200").toInt
    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble
    val beta = options.getOrElse("beta","1.0").toDouble
    val debugIter = options.getOrElse("debugIter","10").toInt
    val seed = options.getOrElse("seed","0").toInt

    println("master:       " + master);          println("trainFile:    " + trainFile)
    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits)
    println("chkptDir:     " + chkptDir);        println("chkptIter     " + chkptIter)
    println("testfile:     " + testFile);        println("justCoCoA     " + justCoCoA)
    println("lambda:       " + lambda);          println("numRounds:    " + numRounds)
    println("localIterFrac:" + localIterFrac);   println("beta          " + beta)
    println("debugIter     " + debugIter);       println("seed          " + seed)

    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)
    if (chkptDir != "") {
      sc.setCheckpointDir(chkptDir)
    } else {
      chkptIter = numRounds + 1
    }

    println("started reading file")
    val dataasdf = OptUtils.loadLIBSVMData(sc,trainFile,numSplits,numFeatures)
//      .map(x => new SparseClassificationPoint(x.index, if (x.label == 1) 1 else -1, x.features))

    val data = dataasdf.zipWithIndex.filter(_._2 < 50000).map(_._1).repartition(numSplits).cache()
    val testData = dataasdf.zipWithIndex.filter(_._2 >= 100000).map(_._1)

    println("finished reading file")
    val n = data.count().toInt

//    val testData = {
//      if (testFile != ""){ OptUtils.loadLIBSVMData(sc,testFile,numSplits,numFeatures).cache() }
//      else { null }
//    }

    var localIters = (localIterFrac * n / data.partitions.size).toInt
    localIters = Math.max(localIters,1)

    val wInit = Array.fill(numFeatures)(0.0)
    val dataArr = data.mapPartitions(x => Iterator(x.toArray))

    val primalLosses: RDD[RealFunction] = data.map(p => new RidgeLoss(p.label))
//    val primalLossesDerivative: RDD[RealFunction] = data.map(p => new LogitLossDerivative(p.label))
    val dualLosses: RDD[RealFunction] = data.map(p => new RidgeLossConjugate(p.label))
//    val dualLossesDerivative: RDD[RealFunction] = data.map(p => new LogitLossConjugateDerivative(p.label))

//    val scOptimizers : RDD[Array[SingleCoordinateOptimizerTrait]] =
//      (dualLosses zip dualLossesDerivative)
//        .map(l => new SGDOptimizer(l._1,l._2,1000,lambda,n))
//        .mapPartitions(x => Iterator(x.toArray))

    val scOptimizers : RDD[Array[SingleCoordinateOptimizerTrait]] =
      data.map(p => new RidgeOptimizer(lambda,n))
        .mapPartitions(x => Iterator(x.toArray))

    val localSolvers: RDD[LocalSolverTrait] =
      scOptimizers.map(new SDCASolver(_, localIters, lambda, n))

    val asdf = dataArr zip localSolvers

    val (finalwCoCoA, finalalphaCoCoA) =
      CoCoA.runCoCoA(sc, asdf, wInit, numRounds, beta, chkptIter, n, testData, debugIter, lambda, seed)
    OptUtils.printSummaryStatsPrimalDual("CoCoA", data, finalwCoCoA, finalalphaCoCoA, lambda, testData, primalLosses, dualLosses)

    sc.stop()
   }
}