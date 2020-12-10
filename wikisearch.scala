// We need the movie name and details like genre, language and so on, so we read it from the movie_metadata file while we get the plot summaries from the plot_summaries.txt files
val plot_sum = sc.textFile("/FileStore/tables/plot_summaries.txt")
val mov_name = sc.textFile("/FileStore/tables/movie_metadata.tsv")

// We are going to map the movie details to the plot summaries first : 
val movies = mov_name.map(mov => (mov.split("\t")(0), mov.split("\t")(2)))

// We need to remove the stopwords from the movie summaries. We can do this by using a set of available stop words and try to cross reference them with the words in the movie summaries. 
// Got this list from https://gist.github.com/sebleier/554280
val stop_words = sc.textFile("/FileStore/tables/NLTK_s_list_of_english_stopwords")

// removing the punctuations and converting all the characters into lower case.
val rem_punctuation = plot_sum.map(mov => mov.replaceAll("""[\p{Punct}]""", " "))
val lowercase_movie = rem_punctuation.map(mov => mov.toLowerCase.split("""\s+"""))

// now we are going to remove the stop words from the movie summaries 
val stop_words_set = stop_words.flatMap(mov => mov.split(",")).collect().toSet
val final_movies = lowercase_movie.map((x => x.map(y => y).filter(stop_word => stop_words_set.contains(stop_word) == false)))

// We start with the search term. Since we can have a single word as well as multiple word queries, we will assume multiple word queries as that case also covers the single word queries.
val query = "Artificial Intelligence"
val search = query.toLowerCase().split(" ")

// we need the tf-idf to proceed with the search to find the search results 
def tf(y: String) = final_movies.map(x => (x(0), x.count(_.contains(y)).toDouble/x.size)).filter(x=>x._2!=0.0)
val term_frequency = search.map(y => tf(y).collect().toMap)

def DF(x: String) = mov_name.flatMap(y => y.split("\n").filter(t => t.contains(x))).map(y => ("t", 1)).reduceByKey(_ + _).collect()(0)._2
val document_frequency = search.map(y => DF(y))
val inverse_document_frequency = document_frequency.map(y => (1+ math.log(mov_name.count()/y)))

def compute_tfidf(x: Int) = term_frequency(x).map(a=>(a._1,a._2*inverse_document_frequency(x))).toMap
val TF_IDF = term_frequency.zipWithIndex.map{ case (e, i) =>compute_tfidf(i) }

// since we have the tf-idf values, we go for searching the results of the query using the tf-idf values 
val search_tf =  search.map(y => search.count(_.contains(y)).toDouble/search.size)
val search_tf_idf = search_tf.zipWithIndex.map{case (e, i) => e * inverse_document_frequency(i)}
val query_ans = math.sqrt(search_tf_idf.reduce((x,y) => x * x + y * y))
val distinct_movies = TF_IDF.flatMap(x => x.map(y=>y._1)).toList.distinct.toArray

// we now find the tf-idf values for the movies that are obtained as result of the query to determine the most relevant movies
def document_function(x:String)= search.zipWithIndex.map{case (e, i) => (TF_IDF(i).get(x).getOrElse(0.0).asInstanceOf[Double]).toDouble }.reduce((x,y)=>x*x+y*y)
val document = distinct_movies.map(x =>  (x, math.sqrt(document_function(x) ))).toMap

//we now calculate the cosine similarity which is the number of times that the query results overlap divided by the dot product of the number of occurances in the individual documents
def dot_function(x:String)= search.zipWithIndex.map{case (e, i) => (search_tf_idf(i) * TF_IDF(i).get(x).getOrElse(0.0).asInstanceOf[Double]).toDouble }.reduce((x,y)=>x+y)
val dot_product = distinct_movies.map(x =>  (x, dot_function(x))).toMap
val cosine = distinct_movies.map( x=> (x, dot_product.get(x).getOrElse(0.0).asInstanceOf[Double] / (document.get(x).getOrElse(0.0).asInstanceOf[Double] * query_ans)))
val cosine_similarity= sc.parallelize(cosine)
val finalresult = movies.join(cosine_similarity).map(x=>(x._2._1,x._2._2)).sortBy(_._2).map(_._1).take(10)

val result = sc.parallelize(finalresult)
result.collect()