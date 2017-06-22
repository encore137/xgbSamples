package ru.megafon.smartReaction

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.classification.LogisticRegression
import ru.megafon.smartReaction.tests.{MLPipelineTest, XGBoostTest}
import ru.megafon.smartReaction.utils.Benchmarks
import ru.megafon.smartReaction.tests.MLTuningTest
object Runner {

  def main(args: Array[String]): Unit = {

    /*val modelPath = "/home/osboxes/Documents/work/model/"
    //val pipelineTestPath = "/home/osboxes/Documents/work/pipelineTest"
    //XGBoostTest.trainAndSave(modelPath)
    //XGBoostTest.loadAndPredict(modelPath)
    //MLPipelineTest.basicConceptTest()

    //create estimators
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val paramMap = List(
      "num_rounds" -> 100).toMap
    val xgbEstimator = new XGBoostEstimator(paramMap)

    Benchmarks.simpleBenchmark("xgboost") {
      MLPipelineTest.pipelineTest(modelPath, xgbEstimator, false)
    }

    Benchmarks.simpleBenchmark("logistic regression") {
      MLPipelineTest.pipelineTest(modelPath, lr, false)
    }*/
    MLTuningTest.crossValidationTest()

  }

}
