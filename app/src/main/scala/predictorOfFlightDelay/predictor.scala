package predictorOfFlightDelay

import org.apache.spark._
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressor}//GBTRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._

import scala.io._

object Flight {

  def main(args: Array[String]) {
    print("\n")
    print("Where is your dataset located? (provide full path on disk) \n")

    val dataPath = readLine()

    var mlTechnique: Int = 0

    while(mlTechnique != 1 && mlTechnique != 2){ //&&mlTechnique != 3
      print("\n")
      print("Which machine learning technique do you want to use? \n")
      print("[1] Linear Regression \n")
      print("[2] Random Forest Trees \n")
      //print("[3] Gradient-Boosted Trees \n")
      mlTechnique = readInt()
    }

    print("\n")
    print("Use categorical features? (yes/no) \n")
    val useCategorical = readBoolean()
    
    val conf = new SparkConf().setAppName("predictor")
    val sparkCtx = new SparkContext(conf)
    val sqlContext = new SQLContext(sparkCtx)

    val rawData = sqlContext.read.format("com.databricks.spark.csv")
                .option("header", "true")
                .option("inferSchema", "true")
                .load(dataPath)
                .withColumn("DelayOutputVar", col("ArrDelay").cast("double"))
                .withColumn("DepDelayDouble", col("DepDelay").cast("double"))
                .withColumn("TaxiOutDouble", col("TaxiOut").cast("double"))
                .cache()

    val data2 = rawData
                .drop("ActualElapsedTime") // Forbidden
                .drop("ArrTime") // Forbidden
                .drop("AirTime") // Forbidden
                .drop("TaxiIn") // Forbidden
                .drop("Diverted") // Forbidden
                .drop("CarrierDelay") // Forbidden
                .drop("WeatherDelay") // Forbidden
                .drop("NASDelay") // Forbidden
                .drop("SecurityDelay") // Forbidden
                .drop("LateAircraftDelay") // Forbidden
                .drop("DepDelay") // Casted to double in a new variable called DepDelayDouble
                .drop("TaxiOut") // Casted to double in a new variable called TaxiOutDouble
                .drop("UniqueCarrier") // Always the same value // Remove correlated variables
                .drop("CancellationCode") // Cancelled flights don't count
                .drop("DepTime") // Highly correlated to CRSDeptime
                .drop("CRSArrTime") // Highly correlated to CRSDeptime
                .drop("CRSElapsedTime") // Highly correlated to Distance
                .drop("Distance") // Remove uncorrelated variables to the arrDelay
                .drop("FlightNum") // Remove uncorrelated variables to the arrDelay
                .drop("CRSDepTime") // Remove uncorrelated variables to the arrDelay
                .drop("Year") // Remove uncorrelated variables to the arrDelay
                .drop("Month") // Remove uncorrelated variables to the arrDelay
                .drop("DayofMonth") // Remove uncorrelated variables to the arrDelay
                .drop("DayOfWeek") // Remove uncorrelated variables to the arrDelay
                .drop("TailNum")

    // remove cancelled flights
    val data = data2.filter("DelayOutputVar is not null")
                
    val assembler = if(useCategorical){
      new VectorAssembler()
        .setInputCols(Array("OriginVec", "DestVec", "DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
        .setHandleInvalid("skip")
    }else{
      new VectorAssembler()
        .setInputCols(Array("DepDelayDouble", "TaxiOutDouble"))
        .setOutputCol("features")
        .setHandleInvalid("skip")
      
    }

    val categoricalVariables = if(useCategorical){
                                Array("Origin", "Dest")
                              }else{
                                null
                              }
   
    val categoricalIndexers = if(useCategorical){
                                categoricalVariables.map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index").setHandleInvalid("skip"))
                              }else{
                                null
                              }
    val categoricalEncoders = if(useCategorical){
                                categoricalVariables.map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec").setDropLast(false))
                              }else{
                                null
                              }
    
    

    mlTechnique match {
      case 1 => 
        val lr = new LinearRegression()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
        val paramGrid = new ParamGridBuilder()
          .addGrid(lr.regParam, Array(0.1, 0.01))
          .addGrid(lr.fitIntercept)
          .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
          .build()

        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)
                                                              }else{
                                                                Array(assembler, lr)
                                                              }

        val pipeline = new Pipeline().setStages(steps)

        val tvs = new TrainValidationSplit()
          .setEstimator(pipeline)
          .setEvaluator(new RegressionEvaluator().setLabelCol("DelayOutputVar"))
          .setEstimatorParamMaps(paramGrid)
          .setTrainRatio(0.7)

        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = tvs.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")

      case 2 =>
        val rf = new RandomForestRegressor()
          .setNumTrees(10)
          .setMaxDepth(10)
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")


        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, rf)
                                                              }else{
                                                                Array(assembler, rf)
                                                              }

        val pipeline = new Pipeline().setStages(steps)

        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = pipeline.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("sqrt(MSE): " + Math.sqrt(rm.meanSquaredError))
        println("mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")
      /*
      case _ =>
        val gbt = new GBTRegressor()
          .setLabelCol("DelayOutputVar")
          .setFeaturesCol("features")
          .setMaxIter(10)

        val steps:Array[org.apache.spark.ml.PipelineStage] = if(useCategorical){
                                                                categoricalIndexers ++ categoricalEncoders ++ Array(assembler, gbt)
                                                              }else{
                                                                Array(assembler, gbt)
                                                              }

        val pipeline = new Pipeline().setStages(steps)


        val Array(training, test) = data.randomSplit(Array(0.70, 0.30), seed = 12345)

        val model = pipeline.fit(training)

        val holdout = model.transform(test).select("prediction", "DelayOutputVar")

        val rm = new RegressionMetrics(holdout.rdd.map(x =>
          (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

        println("sqrt(Mean squared error): " + Math.sqrt(rm.meanSquaredError))
        println("Mean absolute error: " + 	rm.meanAbsoluteError)
        println("R Squared: " + rm.r2)                         
        println("Explained Variance: " + rm.explainedVariance + "\n")
        */

    }
    sparkCtx.stop()
  }

}
