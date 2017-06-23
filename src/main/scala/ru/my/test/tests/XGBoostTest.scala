package ru.my.test.tests

import ml.dmlc.xgboost4j.scala.spark.{TrackerConf, XGBoost}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession


object XGBoostTest {

  val sparkSession = SparkSession.builder().master("local").appName("testApp").getOrCreate()
  val sparkCtx = sparkSession.sparkContext
  val risingArray = Array(2.0, 3.0, 4.0)
  val plainArray = Array(5.0, 5.0, 5.0)
  val trainSeq = Seq(
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(1.0, new DenseVector(risingArray)),
    LabeledPoint(0.0, new DenseVector(plainArray))
  )
  val trainRDD = sparkCtx.parallelize(trainSeq)

  def trainAndSave(path: String): Unit = {
    val paramMap = List(
      "eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "binary:logistic"
    ).toMap

    val xgboostModel = XGBoost.train(trainRDD, paramMap, 1, 1, useExternalMemory = false)

    xgboostModel.saveModelAsHadoopFile(path)(sparkCtx)
  }

  def loadAndPredict(path: String): Unit = {

    val loadedModel = XGBoost.loadModelFromHadoopFile(path)(sparkCtx)

    val xgboostPrediction = loadedModel.predict(trainRDD.map(x => x.features))

    println("===============prediction===============")

    xgboostPrediction.collect().flatten.flatten.foreach(println)
  }

}
