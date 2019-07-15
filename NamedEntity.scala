// Databricks notebook source
//CS6350 Assignment 1
//Nikhil Kalekar nlk180002
//Term: Summer 2019

//  Question 1 generating the count of named entities on the book: Count of Monte Cristo

// COMMAND ----------

// Importing the John Snow NLP libraries
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

// COMMAND ----------

// creating pre-trained model to fetch the Named-entities (like: Names, Location, sentiment/action words, etc.) 
val pipeline = PretrainedPipeline("entity_recognizer_dl", "en")

// COMMAND ----------

//  reading our book as a DF. We picked Count of Monte Christo by Leo Tolsoty 
val input = sc.textFile("/FileStore/tables/CountofMonteChristo.txt").filter(x=>x.length>5).toDS.toDF("text")

// COMMAND ----------

//  checking the input
input.take(10)

// COMMAND ----------

//  applying the pre trained NLP model on the book to generate all the Named Entities line by line and storing it to Named Entity Recognizer column in the form of array<String> or sql.Row types
pipeline.transform(input).select("ner_converter").show(false)

// COMMAND ----------

//  cleaning the output. We can see the result below has generated "[]" as there are no named entities in some lines, thats why we are getting the blank array 
val input2 =pipeline.transform(input).select("ner_converter.result").drop()

// COMMAND ----------

// val input3 = input2.drop()
input2.show(5)

// COMMAND ----------

//  getting rid of the blank/ Null arrays: and exploding the arrays 
import org.apache.spark.sql.functions.{concat,concat_ws, size}

val input3= input2.withColumn("length", size($"result")).filter($"length"=!=0).drop("length") 

val solution1 = input3.withColumn("result", concat_ws(", ", $"result"))

solution1.show()


// COMMAND ----------

//  Since there were multiple named entities on 1 row( representing number of named entities in a single line) I have splitted it and made a new row for each words so I can easily read it to an RDD. then I can apply MapReduce on it!

import org.apache.spark.sql.functions._
val solution2 = solution1.withColumn("new", explode(split($"result", "[,]")))
val solution = solution2.select("new")
solution.show(false)

// COMMAND ----------

//  Reading the entities into an RDD to apply MapReduce


val rows= solution.rdd
// val charRemove
// val rows1=rows.flatMap(x=>x)
import spark.implicits._
// val rows1=rows.flatMap(x=>x(0).asInstanceOf[String].stripPrefix(" ").trim)
// import spark.implicits._
val sourceDS = solution.as[String]
val rows1 = sourceDS.rdd
val rows2 = rows1.map(x=>x.stripPrefix(" ").trim)
// val sourceRdd = solution.rdd.map { case x : Row => x(0).asInstanceOf[String] }.flatMap(s => s.split(","))
// val rows= solution.rdd
// val rows1 = rows.flatMap(_=_)
// val charRemove
// val rows1=rows.flatMap(_.getString(_.length))
// val rows2=rows1.flatMap(_.stripPrefix("\"").stripSuffix("\"").trim)

// COMMAND ----------

rows2.take(20)

// COMMAND ----------

rows2.count()

// COMMAND ----------

// Map function generating the K,V pair where K = Named Entitiy and V=1

val finalResult=rows2.map(x=>(x,1))

// COMMAND ----------

finalResult.collect()

// COMMAND ----------

//  getting the total count of each named entitiy
val finalResult2 = finalResult.reduceByKey((x,y)=>x+y)

// COMMAND ----------

//  Sorting it to show the required result.
finalResult2.sortBy(-_._2).collect()

// COMMAND ----------


