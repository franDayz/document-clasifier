package org.frandayz.tweets.classifier

import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.io.File

/**
 * Created by frandayz on 15.03.16.
 */
object DocumentClassifier {
  def main(args: Array[String]): Unit = {
    val numClusters = args(0).toInt
    val basePath = args(1)
    val stopWordsFile = args(2)
    val inputFiles = getInputFiles(args, numClusters)

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("TweetClassifier")

    val sc = SparkContext.getOrCreate(conf)
    val documents = getDocuments(sc, basePath, inputFiles)
    val first4 = documents.take(4)
  }

  def getInputFiles(args: Array[String], n: Int): Array[String] =
    Range(3, 3 + n).map(i => args(i)).toArray

  def getDocuments(sc: SparkContext,
                   basePath: String,
                   fileNames: Array[String]) =
    fileNames.map(fn => basePath + File.separator + fn)
      .map(path => sc.textFile(path))
      .reduce((rdd1, rdd2) => rdd1.union(rdd2))
}
