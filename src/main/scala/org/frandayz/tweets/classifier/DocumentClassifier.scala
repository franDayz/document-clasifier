package org.frandayz.tweets.classifier

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map
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
      .setAppName("DocumentClassifier")

    val sc = SparkContext.getOrCreate(conf)
    val documents = getDocuments(sc, basePath, inputFiles)
    val tokenizedDocs = documents.map(doc => doc.split(","))
    val stopWords = sc.textFile(getFullPath(stopWordsFile, basePath))
    val dictionary = buildDictionary(sc, tokenizedDocs, stopWords)

    val tokensFeatures = buildTokensFeatures(tokenizedDocs, dictionary)
    val features = tokensFeatures.map(tokFeat => tokFeat._2)
    features.cache()

    val idf = new IDF().fit(features)
    val tfidf = idf.transform(features)

    val numIterations = 10
    val numRuns = 100

    val clusterModel = KMeans.train(tfidf, numClusters, numIterations, numRuns)

    val textAndPredictions = tokensFeatures.map {
      case (tokens, features) =>
        val pred = clusterModel.predict(features)
        (tokens.mkString(", "), pred)
    }

    val clusters = textAndPredictions.groupBy(p => p._2)

    clusters.foreach {
      case(cluster, tweets) =>
        println("========")
        println(cluster)
        tweets.map(tweet => tweet._1).foreach(println)
    }
  }

  def getInputFiles(args: Array[String], n: Int): Array[String] =
    Range(3, 3 + n).map(i => args(i)).toArray

  def getDocuments(sc: SparkContext,
                   basePath: String,
                   fileNames: Array[String]) =
    fileNames.map(fn => getFullPath(fn, basePath))
      .map(path => sc.textFile(path))
      .reduce((rdd1, rdd2) => rdd1.union(rdd2))

  def getFullPath(fileName: String, basePath: String) =
    basePath + File.separator + fileName

  def buildDictionary(sc: SparkContext,
                      tokenizedDocs: RDD[Array[String]],
                      stopWords: RDD[String]): Broadcast[Map[String, Long]] = {
    val allWords = tokenizedDocs.flatMap(w => w)
    val stopWordsSet = stopWords.collect().toSet
    val withoutStopWords = allWords.filter(w => !stopWordsSet.contains(w))
    val wordCounts = withoutStopWords.map(w => (w, 1)).reduceByKey(_ + _)

    sc.broadcast(
      wordCounts.filter(pair => pair._2 > 1)
        .map(pair => pair._1)
        .zipWithIndex().collectAsMap())
  }

  def buildTokensFeatures(tokenizedDocs: RDD[Array[String]],
                    dictionary: Broadcast[Map[String, Long]]) =
    tokenizedDocs.map(tokens => {
      val dict = dictionary.value
      val features = Array.fill(dictionary.value.size) { 0.0 }
      tokens.foreach(
        token => {
          val someIndex = dict.get(token)
          if (someIndex.isDefined) {
            features(someIndex.get.toInt) = 1.0
          }
        }
      )
      (tokens, Vectors.dense(features))
    })
}
