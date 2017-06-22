package ru.megafon.smartReaction.tests

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

object MLPipelineTest {

  val sparkSession = SparkSession.builder().master("local").appName("testApp").getOrCreate()
  val sparkCtx = sparkSession.sparkContext

  val training = sparkSession.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")

  val test = sparkSession.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "spark hadoop spark"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  def basicConceptTest(): Unit = {
    //prepare training data - list of (label, features)
    val training = sparkSession.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    //create a logistic regression - estimator
    val lr = new LogisticRegression()

    //print the params of lr
    println("lr parameters:\n" + lr.explainParams() + "\n")

    //set lr params using setters
    lr.setMaxIter(10).setRegParam(0.01)

    //learn lr model
    val model1 = lr.fit(training)

    //print parameters of model1 during fit()
    println("model1 was fit using parameters: " + model1.parent.extractParamMap)

    //create a paramMap for model
    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter, 30)
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55)

    //paramMap can also be combined
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability")
    val paramMapCombined = paramMap ++ paramMap2

    //learn a new model with combinedMap
    val model2 = lr.fit(training, paramMapCombined)
    println("model2 was fit using parameters: " + model2.parent.extractParamMap)

    //now prepare test data
    val test = sparkSession.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    //make predictions on test data using transform()
    model2.transform(test)
    .select("features", "label", "myProbability", "prediction")
    .collect()
    .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob = $prob, prediction = $prediction")
    }
  }

  def pipelineTest(modelPath: String, estimator: PipelineStage, reloadFlag: Boolean): Unit = {

    //create pipeline elements
    val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")

    val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, estimator))

    //train the model using pipeline
    val model = pipeline.fit(training)

    //now make prediction on testDF
    println("------------original model------------")
    model.transform(test).show()
    /*.select("id", "text", "probability", "prediction")
    .collect()
    .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
      println(s"($id, $text) --> prob = $prob, prediction = $prediction")
    }*/

    if(reloadFlag) {
      //save the model to disk
      model.write.overwrite().save(modelPath)

      //and load it again
      val loadedModel = PipelineModel.load(modelPath)

      //predict to make sure the result is the same
      println("------------loaded model------------")
      loadedModel.transform(test).show()
    }

  }



}
