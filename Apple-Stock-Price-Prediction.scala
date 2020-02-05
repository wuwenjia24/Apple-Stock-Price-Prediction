// Databricks notebook source
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, GBTRegressor, DecisionTreeRegressor,RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, TrainValidationSplit}
import org.apache.spark.ml.regression.RandomForestRegressionModel


// COMMAND ----------

val aapl = sql("SELECT * FROM aapl_csv as aapl WHERE YEAR(aapl.Date) BETWEEN 2008 AND 2018")
val sp = sql("SELECT * FROM gspc_csv as sp WHERE YEAR(sp.Date) BETWEEN 2008 AND 2018")


// COMMAND ----------

//Prediction
val w = Window.orderBy("date")
val df = aapl.withColumn("Log_Adj_Close", log("Adj close"))
val df1 = df.withColumn("pred_daily", lead("Log_Adj_Close", 1).over(w))
val df2 = df1.withColumn("pred_weekly", lead("Log_Adj_Close", 5).over(w))
val df3 = df2.withColumn("pred_biweekly", lead("Log_Adj_Close", 10).over(w))
val df4 = df3.withColumn("pred_monthly", lead("Log_Adj_Close", 21).over(w))
val df5 = df4.withColumn("pred_quarterly", lead("Log_Adj_Close", 84).over(w))
val df6 = df5.withColumn("pred_annually", lead("Log_Adj_Close", 252).over(w))

//difference
val dif = df6.withColumn("dif_daily", (col("pred_daily") - col("Log_Adj_Close")))
val dif1 = dif.withColumn("dif_weekly", (col("pred_weekly") - col("Log_Adj_Close")))
val dif2 = dif1.withColumn("dif_biweekly", (col("pred_biweekly") - col("Log_Adj_Close")))
val dif3 = dif2.withColumn("dif_monthly", (col("pred_monthly") - col("Log_Adj_Close")))
val dif4 = dif3.withColumn("dif_quarterly", (col("pred_quarterly") - col("Log_Adj_Close")))
val dif5 = dif4.withColumn("dif_annually", (col("pred_annually") - col("Log_Adj_Close")))

//lag difference
val dfl = dif5.withColumn("dif_daily_lag", lag("dif_daily", 1).over(w))
val dfl1 = dfl.withColumn("dif_weekly_lag", lag("dif_weekly", 1).over(w))
val dfl2 = dfl1.withColumn("dif_biweekly_lag", lag("dif_biweekly", 1).over(w))
val dfl3 = dfl2.withColumn("dif_monthly_lag", lag("dif_monthly", 1).over(w))
val dfl4 = dfl3.withColumn("dif_quarterly_lag", lag("dif_quarterly", 1).over(w))
val dfl5 = dfl4.withColumn("dif_annually_lag", lag("dif_annually", 1).over(w))

//Normalize data
val nom = dfl5.withColumn("log_open", log("Open"))
val nom1 = nom.withColumn("log_high", log("High"))
val nom2 = nom1.withColumn("log_low", log("Low"))
val nom3 = nom2.withColumn("log_volume", log("Volume"))

//Returns
val re = nom3.withColumn("log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 1).over(w)) / lag("Log_Adj_Close", 5).over(w))
val re1 = re.withColumn("weekly_log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 5).over(w)) / lag("Log_Adj_Close", 10).over(w))
val re2 = re1.withColumn("biweekly_log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 10).over(w)) / lag("Log_Adj_Close", 21).over(w))
val re3 = re2.withColumn("monthly_log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 21).over(w)) / lag("Log_Adj_Close", 42).over(w))
val re4 = re3.withColumn("quarterly_log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 84).over(w)) / lag("Log_Adj_Close", 84).over(w))
val re5 = re4.withColumn("annual_log_return", -(col("Log_Adj_Close") - lag("Log_Adj_Close", 252).over(w)) / lag("Log_Adj_Close", 252).over(w))

//Volume
val v = re5.withColumn("daily_volume_diff", (col("log_volume") - lag("log_volume", 1).over(w))) 
val v1 = v.withColumn("weekly_volume_diff", (col("log_volume") - lag("log_volume", 5).over(w))) 
val v2 = v1.withColumn("biweekly_volume_diff", (col("log_volume") - lag("log_volume", 10).over(w))) 
val v3 = v2.withColumn("monthly_volume_diff", (col("log_volume") - lag("log_volume", 21).over(w))) 
val v4 = v3.withColumn("quarterly_volume_diff", (col("log_volume") - lag("log_volume", 84).over(w))) 
val v5 = v4.withColumn("annual_volume_diff", (col("log_volume") - lag("log_volume", 252).over(w))) 

//Avg & std
val w1 = Window.orderBy("Date").rowsBetween(-5, 0)
val wk = v5.withColumn("weekly_mean", avg("log_return").over(w1))
val wk1 = wk.withColumn("weekly_std", stddev("log_return").over(w1))
val wk2 = wk1.withColumn("weekly_volume_mean", avg("log_volume").over(w1))
val wk3 = wk2.withColumn("weekly_volume_std", stddev("log_volume").over(w1))

val w2 = Window.orderBy("Date").rowsBetween(-10, 0)
val bw = wk3.withColumn("biweekly_mean", avg("log_return").over(w2))
val bw1 = bw.withColumn("biweekly_std", stddev("log_return").over(w2))
val bw2 = bw1.withColumn("biweekly_volume_mean", avg("log_volume").over(w2))
val bw3 = bw2.withColumn("biweekly_volume_std", stddev("log_volume").over(w2))

val w3 = Window.orderBy("Date").rowsBetween(-21, 0)
val mon = bw3.withColumn("monthly_mean", avg("log_return").over(w3))
val mon1 = mon.withColumn("monthly_std", stddev("log_return").over(w3))
val mon2 = mon1.withColumn("monthly_volume_mean", avg("log_volume").over(w3))
val mon3 = mon2.withColumn("monthly_volume_std", stddev("log_volume").over(w3))

val w4 = Window.orderBy("Date").rowsBetween(-84, 0)
val qt = mon3.withColumn("quarterly_mean", avg("log_return").over(w4))
val qt1 = qt.withColumn("quarterly_std", stddev("log_return").over(w4))
val qt2 = qt1.withColumn("quarterly_volume_mean", avg("log_volume").over(w4))
val qt3 = qt2.withColumn("quarterly_volume_std", stddev("log_volume").over(w4))

val w5 = Window.orderBy("Date").rowsBetween(-126, 0)
val sa = qt3.withColumn("semiannual_mean", avg("log_return").over(w5))
val sa1 = sa.withColumn("semiannual_std", stddev("log_return").over(w5))
val sa2 = sa1.withColumn("semiannual_volume_mean", avg("log_volume").over(w5))
val sa3 = sa2.withColumn("semiannual_volume_std", stddev("log_volume").over(w5))

val w6 = Window.orderBy("Date").rowsBetween(-252, 0)
val an = sa3.withColumn("annual_mean", avg("log_return").over(w6))
val an1 = an.withColumn("annual_std", stddev("log_return").over(w6))
val an2 = an1.withColumn("annual_volume_mean", avg("log_volume").over(w6))
val an3 = an2.withColumn("annual_volume_std", stddev("log_volume").over(w6))

//Custom
val w7 = Window.orderBy("Date").rowsBetween(-1, 0)
val dm = an3.withColumn("daily_mean", avg("log_return").over(w7))

val w8 = Window.orderBy("Date").rowsBetween(-84, 0)
val qm = dm.withColumn("quarterly_mean", avg("log_return").over(w8))

val data = qm.drop("Close", "Adj close", "Open", "High", "Low", "Volume").na.fill(0)

// COMMAND ----------

//Normalize data
val nomsp = sp.withColumn("log_open_sp", log("Open"))
val nomsp1 = nomsp.withColumn("log_high_sp", log("High"))
val nomsp2 = nomsp1.withColumn("log_low_sp", log("Low"))
val nomsp3 = nomsp2.withColumn("log_volume_sp", log("Volume"))
val nomsp4 = nomsp3.withColumn("Log_Adj_Close_sp", log("Adj close"))

//Returns
val resp = nomsp4.withColumn("log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 1).over(w)) / lag("Log_Adj_Close_sp", 5).over(w))
val resp1 = resp.withColumn("weekly_log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 5).over(w)) / lag("Log_Adj_Close_sp", 10).over(w))
val resp2 = resp1.withColumn("biweekly_log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 10).over(w)) / lag("Log_Adj_Close_sp", 21).over(w))
val resp3 = resp2.withColumn("monthly_log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 21).over(w)) / lag("Log_Adj_Close_sp", 42).over(w))
val resp4 = resp3.withColumn("quarterly_log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 84).over(w)) / lag("Log_Adj_Close_sp", 84).over(w))
val resp5 = resp4.withColumn("annual_log_return_sp", -(col("Log_Adj_Close_sp") - lag("Log_Adj_Close_sp", 252).over(w)) / lag("Log_Adj_Close_sp", 252).over(w))

//Volume
val vsp = resp5.withColumn("daily_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 1).over(w))) 
val vsp1 = vsp.withColumn("weekly_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 5).over(w))) 
val vsp2 = vsp1.withColumn("biweekly_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 10).over(w))) 
val vsp3 = vsp2.withColumn("monthly_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 21).over(w))) 
val vsp4 = vsp3.withColumn("quarterly_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 84).over(w))) 
val vsp5 = vsp4.withColumn("annual_volume_diff_sp", (col("log_volume_sp") - lag("log_volume_sp", 252).over(w))) 

//Avg & std
//val w1 = Window.orderBy("Date").rowsBetween(-5, 0)
val wksp = vsp5.withColumn("weekly_mean_sp", avg("log_return_sp").over(w1))
val wksp1 = wksp.withColumn("weekly_std_sp", stddev("log_return_sp").over(w1))
val wksp2 = wksp1.withColumn("weekly_volume_mean_sp", avg("log_volume_sp").over(w1))
val wksp3 = wksp2.withColumn("weekly_volume_std_sp", stddev("log_volume_sp").over(w1))

//val w2 = Window.orderBy("Date").rowsBetween(-10, 0)
val bwsp = wksp3.withColumn("biweekly_mean_sp", avg("log_return_sp").over(w2))
val bwsp1 = bwsp.withColumn("biweekly_std_sp", stddev("log_return_sp").over(w2))
val bwsp2 = bwsp1.withColumn("biweekly_volume_mean_sp", avg("log_volume_sp").over(w2))
val bwsp3 = bwsp2.withColumn("biweekly_volume_std_sp", stddev("log_volume_sp").over(w2))

//val w3 = Window.orderBy("Date").rowsBetween(-21, 0)
val monsp = bwsp3.withColumn("monthly_mean_sp", avg("log_return_sp").over(w3))
val monsp1 = monsp.withColumn("monthly_std_sp", stddev("log_return_sp").over(w3))
val monsp2 = monsp1.withColumn("monthly_volume_mean_sp", avg("log_volume_sp").over(w3))
val monsp3 = monsp2.withColumn("monthly_volume_std_sp", stddev("log_volume_sp").over(w3))

//val w4 = Window.orderBy("Date").rowsBetween(-84, 0)
val qtsp = monsp3.withColumn("quarterly_mean_sp", avg("log_return_sp").over(w4))
val qtsp1 = qtsp.withColumn("quarterly_std_sp", stddev("log_return_sp").over(w4))
val qtsp2 = qtsp1.withColumn("quarterly_volume_mean_sp", avg("log_volume_sp").over(w4))
val qtsp3 = qtsp2.withColumn("quarterly_volume_std_sp", stddev("log_volume_sp").over(w4))

//val w5 = Window.orderBy("Date").rowsBetween(-126, 0)
val sasp = qtsp3.withColumn("semiannual_mean_sp", avg("log_return_sp").over(w5))
val sasp1 = sasp.withColumn("semiannual_std_sp", stddev("log_return_sp").over(w5))
val sasp2 = sasp1.withColumn("semiannual_volume_mean_sp", avg("log_volume_sp").over(w5))
val sasp3 = sasp2.withColumn("semiannual_volume_std_sp", stddev("log_volume_sp").over(w5))

//val w6 = Window.orderBy("Date").rowsBetween(-252, 0)
val ansp = sasp3.withColumn("annual_mean_sp", avg("log_return_sp").over(w6))
val ansp1 = ansp.withColumn("annual_std_sp", stddev("log_return_sp").over(w6))
val ansp2 = ansp1.withColumn("annual_volume_mean_sp", avg("log_volume_sp").over(w6))
val ansp3 = ansp2.withColumn("annual_volume_std_sp", stddev("log_volume_sp").over(w6))

//Custom
//val w7 = Window.orderBy("Date").rowsBetween(-1, 0)
val dmsp = ansp3.withColumn("daily_mean_sp", avg("log_return_sp").over(w7))

//val w8 = Window.orderBy("Date").rowsBetween(-84, 0)
val qmsp = dmsp.withColumn("quarterly_mean_sp", avg("log_return_sp").over(w8))

val datasp = qmsp.drop("Close", "Adj close", "Open", "High", "Low", "Volume").na.fill(0)

// COMMAND ----------

val data2 = datasp.withColumnRenamed("Date", "Date_temp")
val data_combined = data.as("data").join(data2.as("data2"), data("Date") === data2("Date_temp"), "inner").drop("Date_temp")

// COMMAND ----------

val featureCols = Array("Log_Adj_Close",
                     "dif_daily_lag",
                     "dif_weekly_lag",
                     "dif_biweekly_lag",
                     "dif_monthly_lag",
                     "dif_quarterly_lag",
                     "dif_annually_lag",
                     "log_open",
                     "log_high",
                     "log_low",
                     "log_return",
                     "weekly_log_return",
                     "biweekly_log_return",
                     "monthly_log_return",
                     "quarterly_log_return",
                     "annual_log_return",
                     "daily_volume_diff",
                     "weekly_volume_diff",
                     "biweekly_volume_diff",
                     "monthly_volume_diff",
                     "quarterly_volume_diff",
                     "annual_volume_diff",
                     "weekly_mean",
                     "weekly_std",
                     "weekly_volume_mean",
                     "weekly_volume_std",
                     "biweekly_mean",
                     "biweekly_std",
                     "biweekly_volume_mean",
                     "biweekly_volume_std",
                     "monthly_mean",
                     "monthly_std",
                     "monthly_volume_mean",
                     "monthly_volume_std",
                     "quarterly_mean",
                     "quarterly_std",
                     "quarterly_volume_mean",
                     "quarterly_volume_std",
                     "semiannual_mean",
                     "semiannual_std",
                     "semiannual_volume_mean",
                     "semiannual_volume_std",
                     "annual_mean",
                     "annual_std",
                     "annual_volume_mean",
                     "annual_volume_std",
                     "daily_mean",
                     "Log_Adj_Close_sp",
                     "log_open_sp",
                     "log_high_sp",
                     "log_low_sp",
                     "log_return_sp",
                     "weekly_log_return_sp",
                     "biweekly_log_return_sp",
                     "monthly_log_return_sp",
                     "quarterly_log_return_sp",
                     "annual_log_return_sp",
                     "daily_volume_diff_sp",
                     "weekly_volume_diff_sp",
                     "biweekly_volume_diff_sp",
                     "monthly_volume_diff_sp",
                     "quarterly_volume_diff_sp",
                     "annual_volume_diff_sp",
                     "weekly_mean_sp",
                     "weekly_std_sp",
                     "weekly_volume_mean_sp",
                     "weekly_volume_std_sp",
                     "biweekly_mean_sp",
                     "biweekly_std_sp",
                     "biweekly_volume_mean_sp",
                     "biweekly_volume_std_sp",
                     "monthly_mean_sp",
                     "monthly_std_sp",
                     "monthly_volume_mean_sp",
                     "monthly_volume_std_sp",
                     "quarterly_mean_sp",
                     "quarterly_std_sp",
                     "quarterly_volume_mean_sp",
                     "quarterly_volume_std_sp",
                     "semiannual_mean_sp",
                     "semiannual_std_sp",
                     "semiannual_volume_mean_sp",
                     "semiannual_volume_std_sp",
                     "annual_mean_sp",
                     "annual_std_sp",
                     "annual_volume_mean_sp",
                     "annual_volume_std_sp",
                     "daily_mean_sp"
                      )
val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

val output = assembler.transform(data_combined)

val vec = output.select("Date",
                   "features",
                   "dif_daily",
                   "dif_weekly",
                   "dif_biweekly",
                   "dif_monthly",
                   "dif_quarterly",
                   "dif_annually",
                   "pred_daily",
                   "pred_weekly",
                   "pred_biweekly",
                   "pred_monthly",
                   "pred_quarterly",
                   "pred_annually")

vec.show()

// COMMAND ----------

featureCols.length

// COMMAND ----------

//only keep the rows with no 0s
val raw = vec.filter($"dif_quarterly" =!= 0).withColumn("rank", percent_rank().over(w))



//split data set 70/30 for model train and test
val test = raw.where("rank >= .7").drop("rank")

//split train set 80/20 for model validation
val train = raw.where("rank < .56").drop("rank")

val valid = raw.where("rank < .7").where("rank >= .56").drop("rank")

// COMMAND ----------

val train_rf_daily = train.withColumnRenamed("dif_daily", "label").select("features", "label")
val test_rf_daily = test.withColumnRenamed("dif_daily", "label").select("features", "label")
val valid_rf_daily = valid.withColumnRenamed("dif_daily", "label").select("features", "label")

val train_rf_weekly = train.withColumnRenamed("dif_weekly", "label").select("features", "label")
val test_rf_weekly = test.withColumnRenamed("dif_weekly", "label").select("features", "label")
val valid_rf_weekly = valid.withColumnRenamed("dif_weekly", "label").select("features", "label")

val train_rf_biweekly = train.withColumnRenamed("dif_biweekly", "label").select("features", "label")
val test_rf_biweekly = test.withColumnRenamed("dif_biweekly", "label").select("features", "label")
val valid_rf_biweekly = valid.withColumnRenamed("dif_biweekly", "label").select("features", "label")

val train_rf_monthly = train.withColumnRenamed("dif_monthly", "label").select("features", "label")
val test_rf_monthly = test.withColumnRenamed("dif_monthly", "label").select("features", "label")
val valid_rf_monthly = valid.withColumnRenamed("dif_monthly", "label").select("features", "label")

val train_rf_quarterly = train.withColumnRenamed("dif_quarterly", "label").select("features", "label")
val test_rf_quarterly = test.withColumnRenamed("dif_quarterly", "label").select("features", "label")
val valid_rf_quarterly = valid.withColumnRenamed("dif_quarterly", "label").select("features", "label")

val rf  = new RandomForestRegressor()

val rf_evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("label")
      .setPredictionCol("prediction")

val test_date = test.select("Date", "pred_daily").withColumn("rowId2", monotonically_increasing_id())



// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 3 to 12){
  var rf_model = rf.setMaxDepth(i)
  var model = rf_model.fit(train_rf_daily)
  var pred_train = model.transform(train_rf_daily)
  var pred_valid = model.transform(valid_rf_daily)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 1 to 10){
  var rf_model = rf.setNumTrees(i).setMaxDepth(11)
  var model = rf_model.fit(train_rf_daily)
  var pred_train = model.transform(train_rf_daily)
  var pred_valid = model.transform(test_rf_daily)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

val rf_d  = new RandomForestRegressor().setMaxDepth(11).setNumTrees(6) //9 10

val rf_daily = rf_d.fit(train_rf_daily)

//Make predictions on test data with the best set of parameters.

val rf_pred_daily_train = rf_daily.transform(train_rf_daily)
val rf_pred_daily_test = rf_daily.transform(test_rf_daily)


val rmse_daily_train = rf_evaluator.evaluate(rf_pred_daily_train)
val rmse_daily_test = rf_evaluator.evaluate(rf_pred_daily_test)

println(s"Daily Train RMSE: ${rmse_daily_train}")
println(s"Daily Test RMSE: ${rmse_daily_test}")

val rf_1 = rf_pred_daily_test.withColumn("rowId1", monotonically_increasing_id())

val rf_combine = rf_1.as("rf_1").join(test_date.as("test_date"), rf_1("rowId1") === test_date("rowId2"), "inner").drop("rowId1","rowId2")

val rf_result_daily = rf_combine.withColumn("pred_added", (col("prediction") + lag("pred_daily", 1).over(w)))
.withColumn("label_added", (col("label") + lag("pred_daily", 1).over(w)))

display(rf_result_daily)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 3 to 12){
  var rf_model = rf.setMaxDepth(i)
  var model = rf_model.fit(train_rf_weekly)
  var pred_train = model.transform(train_rf_weekly)
  var pred_valid = model.transform(valid_rf_weekly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 3 to 12){
  var rf_model = rf.setNumTrees(i).setMaxDepth(8)
  var model = rf_model.fit(train_rf_weekly)
  var pred_train = model.transform(train_rf_weekly)
  var pred_valid = model.transform(valid_rf_weekly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

val rf_w  = new RandomForestRegressor().setMaxDepth(8).setNumTrees(5)//7 9
val rf_weekly = rf_w.fit(train_rf_weekly)

//Make predictions on test data with the best set of parameters.

val rf_pred_weekly_train = rf_weekly.transform(train_rf_weekly)
val rf_pred_weekly_test = rf_weekly.transform(test_rf_weekly)

val rmse_weekly_train = rf_evaluator.evaluate(rf_pred_weekly_train)
val rmse_weekly_test = rf_evaluator.evaluate(rf_pred_weekly_test)

println(s"Weekly Train RMSE: ${rmse_weekly_train}")
println(s"Weekly Test RMSE: ${rmse_weekly_test}")

val rf_1 = rf_pred_weekly_test.withColumn("rowId1", monotonically_increasing_id())

val rf_combine = rf_1.as("rf_1").join(test_date.as("test_date"), rf_1("rowId1") === test_date("rowId2"), "inner").drop("rowId1","rowId2")

val rf_result_weekly = rf_combine.withColumn("pred_added", (col("prediction") + lag("pred_daily", 1).over(w)))
.withColumn("label_added", (col("label") + lag("pred_daily", 1).over(w)))

display(rf_result_weekly)

// COMMAND ----------

val rf  = new RandomForestRegressor()
var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 1 to 10){
  var rf_model = rf.setMaxDepth(i)
  var model = rf_model.fit(train_rf_biweekly)
  var pred_train = model.transform(train_rf_biweekly)
  var pred_valid = model.transform(valid_rf_biweekly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 3 to 12){
  var rf_model = rf.setNumTrees(i).setMaxDepth(6)
  var model = rf_model.fit(train_rf_biweekly)
  var pred_train = model.transform(train_rf_biweekly)
  var pred_valid = model.transform(valid_rf_biweekly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

val rf_bw  = new RandomForestRegressor().setMaxDepth(6).setNumTrees(9) //5 8
val rf_biweekly = rf_bw.fit(train_rf_biweekly)

//Make predictions on test data with the best set of parameters.

val rf_pred_biweekly_train = rf_biweekly.transform(train_rf_biweekly)
val rf_pred_biweekly_test = rf_biweekly.transform(test_rf_biweekly)

val rmse_biweekly_train = rf_evaluator.evaluate(rf_pred_biweekly_train)
val rmse_biweekly_test = rf_evaluator.evaluate(rf_pred_biweekly_test)

println(s"biweekly Train RMSE: ${rmse_biweekly_train}")
println(s"biweekly Test RMSE: ${rmse_biweekly_test}")

val rf_1 = rf_pred_biweekly_test.withColumn("rowId1", monotonically_increasing_id())

val rf_combine = rf_1.as("rf_1").join(test_date.as("test_date"), rf_1("rowId1") === test_date("rowId2"), "inner").drop("rowId1","rowId2")

val rf_result_biweekly = rf_combine.withColumn("pred_added", (col("prediction") + lag("pred_daily", 1).over(w)))
.withColumn("label_added", (col("label") + lag("pred_daily", 1).over(w)))

display(rf_result_biweekly)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 1 to 10){
  var rf_model = rf.setMaxDepth(i)
  var model = rf_model.fit(train_rf_monthly)
  var pred_train = model.transform(train_rf_monthly)
  var pred_valid = model.transform(valid_rf_monthly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 5 to 14){
  var rf_model = rf.setNumTrees(i).setMaxDepth(4)
  var model = rf_model.fit(train_rf_monthly)
  var pred_train = model.transform(train_rf_monthly)
  var pred_valid = model.transform(valid_rf_monthly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

val rf_m  = new RandomForestRegressor().setMaxDepth(4).setNumTrees(10) //4 8
val rf_monthly = rf_m.fit(train_rf_monthly)

//Make predictions on test data with the best set of parameters.

val rf_pred_monthly_train = rf_monthly.transform(train_rf_monthly)
val rf_pred_monthly_test = rf_monthly.transform(test_rf_monthly)

val rmse_monthly_train = rf_evaluator.evaluate(rf_pred_monthly_train)
val rmse_monthly_test = rf_evaluator.evaluate(rf_pred_monthly_test)

println(s"monthly Train RMSE: ${rmse_monthly_train}")
println(s"monthly Test RMSE: ${rmse_monthly_test}")

val rf_1 = rf_pred_monthly_test.withColumn("rowId1", monotonically_increasing_id())

val rf_combine = rf_1.as("rf_1").join(test_date.as("test_date"), rf_1("rowId1") === test_date("rowId2"), "inner").drop("rowId1","rowId2")

val rf_result_monthly = rf_combine.withColumn("pred_added", (col("prediction") + lag("pred_daily", 1).over(w)))
.withColumn("label_added", (col("label") + lag("pred_daily", 1).over(w)))

display(rf_result_monthly)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 1 to 10){
  var rf_model = rf.setMaxDepth(i)
  var model = rf_model.fit(train_rf_quarterly)
  var pred_train = model.transform(train_rf_quarterly)
  var pred_valid = model.transform(valid_rf_quarterly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

var result_train = new Array[Double](11)
var result_valid = new Array[Double](11)

var i = 0
var j = 1
for (i <- 3 to 12){
  var rf_model = rf.setNumTrees(i).setMaxDepth(3)
  var model = rf_model.fit(train_rf_quarterly)
  var pred_train = model.transform(train_rf_quarterly)
  var pred_valid = model.transform(valid_rf_quarterly)
  var rmse_train = rf_evaluator.evaluate(pred_train)
  var rmse_valid = rf_evaluator.evaluate(pred_valid)
  result_train(j) = rmse_train
  result_valid(j) = rmse_valid
  j+=1
}
var rf_tune = sc.parallelize(result_train zip result_valid).toDF("train","valid")
display(rf_tune)

// COMMAND ----------

val rf_q  = new RandomForestRegressor().setMaxDepth(3).setNumTrees(7)
val rf_quarterly = rf_q.fit(train_rf_quarterly)

//Make predictions on test data with the best set of parameters.

val rf_pred_quarterly_train = rf_quarterly.transform(train_rf_quarterly)
val rf_pred_quarterly_test = rf_quarterly.transform(test_rf_quarterly)

val rmse_quarterly_train = rf_evaluator.evaluate(rf_pred_quarterly_train)
val rmse_quarterly_test = rf_evaluator.evaluate(rf_pred_quarterly_test)

println(s"quarterly Train RMSE: ${rmse_quarterly_train}")
println(s"quarterly Test RMSE: ${rmse_quarterly_test}")

val rf_1 = rf_pred_quarterly_test.withColumn("rowId1", monotonically_increasing_id())

val rf_combine = rf_1.as("rf_1").join(test_date.as("test_date"), rf_1("rowId1") === test_date("rowId2"), "inner").drop("rowId1","rowId2")

val rf_result_quarterly = rf_combine.withColumn("pred_added", (col("prediction") + lag("pred_daily", 1).over(w)))
.withColumn("label_added", (col("label") + lag("pred_daily", 1).over(w)))

display(rf_result_quarterly)

// COMMAND ----------

display(rf_result_daily)

// COMMAND ----------

display(rf_result_weekly)

// COMMAND ----------

display(rf_result_biweekly)

// COMMAND ----------

display(rf_result_monthly)

// COMMAND ----------

display(rf_result_quarterly)

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql._
val schema = StructType(
    StructField("Features", StringType, false) ::
    StructField("Importances", DoubleType, false) :: Nil)

val res = featureCols.zip(rf_daily.featureImportances.toArray).sortBy(-_._2).take(10)

val rdd = sc.parallelize (res).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))

val feature_imp = sqlContext.createDataFrame(rdd, schema)

display(feature_imp)

// COMMAND ----------

val res = featureCols.zip(rf_weekly.featureImportances.toArray).sortBy(-_._2).take(10)
val rdd = sc.parallelize (res).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
val feature_imp = sqlContext.createDataFrame(rdd, schema)
display(feature_imp)

// COMMAND ----------

val res = featureCols.zip(rf_biweekly.featureImportances.toArray).sortBy(-_._2).take(10)
val rdd = sc.parallelize (res).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
val feature_imp = sqlContext.createDataFrame(rdd, schema)
display(feature_imp)

// COMMAND ----------

val res = featureCols.zip(rf_monthly.featureImportances.toArray).sortBy(-_._2).take(10)
val rdd = sc.parallelize (res).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
val feature_imp = sqlContext.createDataFrame(rdd, schema)
display(feature_imp)

// COMMAND ----------

val res = featureCols.zip(rf_quarterly.featureImportances.toArray).sortBy(-_._2).take(10)
val rdd = sc.parallelize (res).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
val feature_imp = sqlContext.createDataFrame(rdd, schema)
display(feature_imp)

// COMMAND ----------

///SMAPE Function
val results = Array(rf_result_daily, rf_result_weekly, rf_result_biweekly, rf_result_monthly, rf_result_quarterly)
for (x <- results){
  val n = x.count().toInt
  val df_new = x.withColumn("diff", exp(x("prediction")) - exp(x("label")))

  val df_new_abs = df_new.select($"diff",$"prediction",$"label", abs($"diff").as("abs_diff"),abs(exp($"prediction")).as("abs_prediction"),abs(exp($"label")).as("abs_label"))
  val df_new_abs2 = df_new_abs.withColumn("denum", ($"abs_prediction"+$"abs_label")/2).withColumn("ratio",$"abs_diff"/$"denum")
  val df_smape_ratio = df_new_abs2.select("ratio")
  val sum_cols = df_new_abs2.columns.tail.map(x => sum(col(x)))
  val df_smape_sum = df_new_abs2.agg(sum_cols.head, sum_cols.tail: _*).select("sum(ratio)").withColumnRenamed("sum(ratio)","sum_ratio")
  val sum_s = df_smape_sum.select("sum_ratio").first.getDouble(0)
  val smape = 100*sum_s /n
  println(smape)
}


// COMMAND ----------

println(s"daily Test RMSE: ${rmse_daily_train}")
println(s"daily Test RMSE: ${rmse_daily_test}")
println(s"Weekly Test RMSE: ${rmse_weekly_train}")
println(s"Weekly Test RMSE: ${rmse_weekly_test}")
println(s"bieekly Test RMSE: ${rmse_biweekly_train}")
println(s"bieekly Test RMSE: ${rmse_biweekly_test}")
println(s"monthly Test RMSE: ${rmse_monthly_train}")
println(s"monthly Test RMSE: ${rmse_monthly_test}")
println(s"quarterly Test RMSE: ${rmse_quarterly_train}")
println(s"quarterly Test RMSE: ${rmse_quarterly_test}")

// COMMAND ----------

///SMAPE Function
val results = Array(rf_result_daily,  rf_result_weekly, rf_result_biweekly, rf_result_monthly, rf_result_quarterly)
for (x <- results){
  val n = x.count().toInt
  val df_new = x.withColumn("diff", exp(x("pred_added")) - exp(x("label_added")))

  val df_new_abs = df_new.select($"diff",$"pred_added",$"label_added", abs($"diff").as("abs_diff"),abs(exp($"pred_added")).as("abs_prediction"),abs(exp($"label_added")).as("abs_label"))
  val df_new_abs2 = df_new_abs.withColumn("denum", ($"abs_prediction"+$"abs_label")/2).withColumn("ratio",$"abs_diff"/$"denum")
  val df_smape_ratio = df_new_abs2.select("ratio")
  val sum_cols = df_new_abs2.columns.tail.map(x => sum(col(x)))
  val df_smape_sum = df_new_abs2.agg(sum_cols.head, sum_cols.tail: _*).select("sum(ratio)").withColumnRenamed("sum(ratio)","sum_ratio")
  val sum_s = df_smape_sum.select("sum_ratio").first.getDouble(0)
  val smape = 100*sum_s /n
  println(smape)
}


// COMMAND ----------


