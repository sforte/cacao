package solvers

import distopt.utils.Implicits._
import distopt.utils._
import localsolvers.{PrimalLocalSolverTrait, LocalSolverTrait}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object CaCaO {

  /**
   * CoCoA - Communication-efficient distributed dual Coordinate Ascent.
   * 
   * @param sc Spark context
   * @param data RDD of all data examples
   * @param wInit initial weight vector (has to be zero)
   * @param numRounds number of outer iterations T in the paper
   * @param beta scaling parameter. beta=1 gives averaging, beta=K=data.partitions.size gives (aggressive) adding
   * @param debugIter ...
   * @param seed ...
   * @return
   */
  def runCaCaO(
    sc: SparkContext, 
    data: RDD[(Array[SparseClassificationPoint], PrimalLocalSolverTrait)],
    wInit: Array[Double], 
    numRounds: Int,
    beta: Double,
    n: Int,
    seed: Int) : Array[Double] = {
    
    val parts = data.partitions.size
    var localWs = data.mapPartitions(x => Iterator(wInit.clone()))

    println("\nRunning CaCaO on "+n+" data examples, distributed over "+parts+" workers")

    val scaling = beta / parts
    var w = wInit

    for(t <- 1 to numRounds){

      val newlocalWs = (data zip localWs).mapPartitions(
        partitionUpdate(_,w,scaling,seed+t),preservesPartitioning=true).persist()

//      println(newlocalWs.map(_.mkString).toLocalIterator.mkString(" "), w(0))

      w = (localWs zip newlocalWs).map(x => x._2.plus(x._1.times(-1))).reduce(_ plus _).times(scaling).plus(w)

      localWs = newlocalWs

      println(s"round $t")
    }

    w
  }

  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   * @param data ...
   * @param wTot ...
   * @param scaling this is the scaling factor beta/K in the paper
   * @param seed ...
   * @return
   */
  private def partitionUpdate(
    data: Iterator[((Array[SparseClassificationPoint], PrimalLocalSolverTrait),Array[Double])],
    wTot: Array[Double],
    scaling: Double,
    seed: Int): Iterator[Array[Double]] = {

    val ((localData, solver), wOld) = data.next

    val wNew = solver.optimize(localData, wOld, wTot.plus(wOld.times(-1)), seed)

    Iterator(wNew)
  }
}

//package distopt
//
//import _root_.solvers.CaCaO
//import localsolvers.LogisticRegression.{LogitLoss, LogitLossConjugate, LogitLossConjugateDerivative}
//import localsolvers.RidgeRegression.{RidgeLossConjugate, RidgeLoss, RidgeOptimizer}
//import localsolvers.SVMClassification.{HingeLossConjugateDerivative, SVMOptimizer, HingeLossConjugate, HingeLoss}
//import localsolvers._
//import org.apache.spark.rdd.RDD
//import org.apache.spark.{SparkContext, SparkConf}
//import distopt.utils._
//import distopt.solvers._
//
//object driver {
//
//  def main(args: Array[String]) {
//
//    val options =  args.map { arg =>
//      arg.dropWhile(_ == '-').split('=') match {
//        case Array(opt, v) => opt -> v
//        case Array(opt) => opt -> "true"
//        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
//      }
//    }.toMap
//
//    // read in inputs
//    val master = options.getOrElse("master", "local[4]")
//    val trainFile = options.getOrElse("trainFile", "")
//    val numFeatures = options.getOrElse("numFeatures", "0").toInt
//    val numSplits = options.getOrElse("numSplits","1").toInt
//    val chkptDir = options.getOrElse("chkptDir","")
//    var chkptIter = options.getOrElse("chkptIter","100").toInt
//    val testFile = options.getOrElse("testFile", "")
//    val justCoCoA = options.getOrElse("justCoCoA", "true").toBoolean
//
//    val lambda = options.getOrElse("lambda", "0.01").toDouble
//    val numRounds = options.getOrElse("numRounds", "200").toInt
//    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble
//    val beta = options.getOrElse("beta","1.0").toDouble
//    val debugIter = options.getOrElse("debugIter","10").toInt
//    val seed = options.getOrElse("seed","0").toInt
//
//    println("master:       " + master);          println("trainFile:    " + trainFile)
//    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits)
//    println("chkptDir:     " + chkptDir);        println("chkptIter     " + chkptIter)
//    println("testfile:     " + testFile);        println("justCoCoA     " + justCoCoA)
//    println("lambda:       " + lambda);          println("numRounds:    " + numRounds)
//    println("localIterFrac:" + localIterFrac);   println("beta          " + beta)
//    println("debugIter     " + debugIter);       println("seed          " + seed)
//
//    val conf = new SparkConf().setMaster(master)
//      .setAppName("demoCoCoA")
//      .setJars(SparkContext.jarOfObject(this).toSeq)
//    val sc = new SparkContext(conf)
//    if (chkptDir != "") {
//      sc.setCheckpointDir(chkptDir)
//    } else {
//      chkptIter = numRounds + 1
//    }
//
//    val data = OptUtils.loadLIBSVMData(sc,trainFile,numSplits,numFeatures).cache()
//    val n = data.count().toInt
//
//    val testData = {
//      if (testFile != ""){ OptUtils.loadLIBSVMData(sc,testFile,numSplits,numFeatures).cache() }
//      else { null }
//    }
//
//    var localIters = (localIterFrac * n / data.partitions.size).toInt
//    localIters = Math.max(localIters,1)
//
//    val wInit = Array.fill(numFeatures)(0.0)
//    val dataArr : RDD[(Array[SparseClassificationPoint], PrimalLocalSolverTrait)] =
//      data.mapPartitions(x => Iterator((x.toArray, new PrimalSGDLSolver(n,lambda,1000))))
//
//    val primalLosses: RDD[RealFunction] = data.map(p => new HingeLoss(p.label))
//
//    val finalw = CaCaO.runCaCaO(sc, dataArr, wInit, numRounds, beta, n, seed)
//
//    println("objective value: "+OptUtils.computePrimalObjective(data, finalw, lambda, primalLosses))
//
//    sc.stop()
//  }
//}
